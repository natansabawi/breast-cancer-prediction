"""
image_features.py
==================
Estimates the 8 key tumor-cell features (radius, texture, perimeter, area,
concavity, concave points -- mean and worst) directly from an uploaded
microscopy image, using classical image processing (OpenCV).

IMPORTANT — read this before trusting the numbers:
This is NOT the same pipeline used to build the original Wisconsin dataset,
which relied on specialized, calibrated boundary-fitting software applied to
digitized fine-needle-aspirate slides. Here we detect nucleus-like blobs with
thresholding + watershed, measure their pixel geometry, and then quantile-map
those pixel measurements onto the *statistical distribution* of the original
training data (since we have no physical microns-per-pixel calibration from
a plain uploaded image). This keeps the numbers in a sane range for the
model, but it is an educational approximation, not a diagnostic-grade
measurement. Always show the disclaimer in the UI next to these results.
"""

import json
import os
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

QUANTILE_PATH = os.path.join(os.path.dirname(__file__), "feature_quantiles.json")

KEY_FEATURES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "concavity_mean",
    "concave points_mean",
    "radius_worst",
    "concave points_worst",
]


def _load_quantiles():
    with open(QUANTILE_PATH, "r") as f:
        return json.load(f)


_QUANTILES = None


def _quantile_map(raw_value, feature_name, raw_min, raw_max):
    """Maps a raw pixel-based measurement onto the training data's scale.

    We don't know the true microns-per-pixel of an arbitrary uploaded image,
    so instead of pretending physical calibration, we rank the raw value
    against the *other detected nuclei in this same image* (raw_min/raw_max)
    and place it at the equivalent percentile of the real dataset's
    distribution for that feature. A nucleus that is relatively large/rough
    compared to its neighbors in the image ends up mapped to a relatively
    large/rough value in the model's expected range.
    """
    global _QUANTILES
    if _QUANTILES is None:
        _QUANTILES = _load_quantiles()

    if raw_max <= raw_min:
        percentile = 50.0
    else:
        percentile = 100.0 * (raw_value - raw_min) / (raw_max - raw_min)
        percentile = float(np.clip(percentile, 0, 100))

    table = _QUANTILES[feature_name]
    idx = percentile  # table is indexed 0..100
    lo = int(np.floor(idx))
    hi = int(np.ceil(idx))
    if lo == hi:
        return table[lo]
    frac = idx - lo
    return table[lo] * (1 - frac) + table[hi] * frac


def _segment_nuclei(gray):
    """Threshold + watershed segmentation. Returns a labeled image."""
    # Contrast boost so faint stains still separate from background
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Otsu threshold; try both polarities and keep whichever gives a more
    # plausible foreground fraction (nuclei are usually the minority class)
    _, th1 = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th2 = cv2.bitwise_not(th1)
    frac1 = np.mean(th1 > 0)
    frac2 = np.mean(th2 > 0)
    binary = th1 if frac1 < frac2 else th2

    # Clean small noise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Watershed to split touching nuclei
    distance = ndi.distance_transform_edt(binary)
    coords = peak_local_max(distance, min_distance=8, labels=binary)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=binary)
    return labels


def extract_features_from_image(image_bgr, min_nucleus_px=40, max_nucleus_frac=0.2):
    """
    Runs the full pipeline on a BGR OpenCV image and returns:
      - a dict of the 8 key features (quantile-mapped, ready for the model)
      - an annotated preview image (BGR) with detected nuclei outlined
      - the number of nuclei detected
    Raises ValueError if too few nuclei are detected to produce a result.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    img_area = h * w

    labels = _segment_nuclei(gray)

    radii, textures, perimeters, areas, concavities, concave_pts = [], [], [], [], [], []
    annotated = image_bgr.copy()

    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        nucleus_mask = (labels == label_id).astype(np.uint8) * 255
        area_px = float(np.sum(nucleus_mask > 0))

        if area_px < min_nucleus_px or area_px > max_nucleus_frac * img_area:
            continue  # too small (noise) or too large (background blob)

        contours, _ = cv2.findContours(nucleus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5:
            continue

        perimeter_px = cv2.arcLength(contour, True)
        radius_px = np.sqrt(area_px / np.pi)

        # Texture: intensity variation inside the nucleus
        texture_val = float(np.std(gray[nucleus_mask > 0]))

        # Concavity via convex hull comparison
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        concavity_val = max(0.0, (hull_area - cv2.contourArea(contour)) / hull_area) if hull_area > 0 else 0.0

        # Concave points via convexity defects
        hull_idx = cv2.convexHull(contour, returnPoints=False)
        n_defects = 0
        if hull_idx is not None and len(hull_idx) > 3:
            try:
                hull_idx_sorted = np.sort(hull_idx.flatten())[:, None]
                defects = cv2.convexityDefects(contour, hull_idx_sorted)
                if defects is not None:
                    defects = np.asarray(defects).reshape(-1, 4)
                    depth_thresh = 0.02 * radius_px * 256  # scaled depth units used by OpenCV
                    n_defects = int(np.sum(defects[:, 3] > depth_thresh))
            except (cv2.error, IndexError, ValueError):
                n_defects = 0

        radii.append(radius_px)
        textures.append(texture_val)
        perimeters.append(perimeter_px)
        areas.append(area_px)
        concavities.append(concavity_val)
        concave_pts.append(n_defects)

        cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)

    n_nuclei = len(radii)
    if n_nuclei < 3:
        raise ValueError(
            f"Only detected {n_nuclei} clear nucleus-like region(s) in this image. "
            "Need at least 3 for a reliable estimate — try a clearer, higher-contrast "
            "microscopy image with visible individual cells."
        )

    def mean_and_worst(values):
        values = np.array(values, dtype=float)
        mean_v = float(np.mean(values))
        k = max(1, min(3, len(values)))
        worst_v = float(np.mean(np.sort(values)[-k:]))
        return mean_v, worst_v, float(np.min(values)), float(np.max(values))

    radius_mean_raw, radius_worst_raw, r_min, r_max = mean_and_worst(radii)
    texture_mean_raw, _, t_min, t_max = mean_and_worst(textures)
    perimeter_mean_raw, _, p_min, p_max = mean_and_worst(perimeters)
    area_mean_raw, _, a_min, a_max = mean_and_worst(areas)
    concavity_mean_raw, _, c_min, c_max = mean_and_worst(concavities)
    concave_pts_mean_raw, concave_pts_worst_raw, cp_min, cp_max = mean_and_worst(concave_pts)

    result = {
        "radius_mean": _quantile_map(radius_mean_raw, "radius_mean", r_min, r_max),
        "texture_mean": _quantile_map(texture_mean_raw, "texture_mean", t_min, t_max),
        "perimeter_mean": _quantile_map(perimeter_mean_raw, "perimeter_mean", p_min, p_max),
        "area_mean": _quantile_map(area_mean_raw, "area_mean", a_min, a_max),
        "concavity_mean": _quantile_map(concavity_mean_raw, "concavity_mean", c_min, c_max),
        "concave points_mean": _quantile_map(concave_pts_mean_raw, "concave points_mean", cp_min, cp_max),
        "radius_worst": _quantile_map(radius_worst_raw, "radius_worst", r_min, r_max),
        "concave points_worst": _quantile_map(concave_pts_worst_raw, "concave points_worst", cp_min, cp_max),
    }

    return result, annotated, n_nuclei
