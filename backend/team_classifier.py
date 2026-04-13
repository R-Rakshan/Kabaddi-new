"""
team_classifier.py  — v4
========================
Robust jersey-based team classification using:

  1. HSV TORSO COLOR — Extract dominant HSV color from the torso region
     (middle 30%-70% of bbox height) to avoid skin/hair contamination.
     K-means clusters per-crop, then temporal smoothing over sliding window.

  2. SPATIAL X — X-centroid as secondary signal for anchor alignment.

  3. HARD LOCK — Once a track's team is assigned, it NEVER changes.

Strategy
--------
- Accumulate HSV color and spatial X in per-track sliding windows.
- Global K-Means anchors established once from accumulated data and frozen.
- Temporal averaging: team label is decided from the MEAN of the entire
  observation history, not a single frame. This guarantees zero flicker.
- Once locked, the label is permanent even if the player changes court half.
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Tuple, Optional

# ── Config ─────────────────────────────────────────────────────────────────
OBS_WINDOW = 30  # frames to keep per track
MIN_OBS = 8  # minimum observations before classifying (reduced)
COLOR_WEIGHT = 0.80  # weight for HSV color signal (Primary)
SPATIAL_WEIGHT = 0.20  # weight for X-position signal (Secondary)

# ── Per-track observation buffers ─────────────────────────────────────────
_x_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=OBS_WINDOW))
_hsv_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=OBS_WINDOW))

# ── Locked assignments ─────────────────────────────────────────────────────
_locked: Dict[int, int] = {}  # track_id -> team (0 or 1)

# ── Global cluster anchors (established once and never reset) ──────────────
_x_anchors: Optional[Tuple[float, float]] = None
_hsv_anchors: Optional[Tuple[np.ndarray, np.ndarray]] = None  # Two 3D HSV centers


def _extract_torso_hsv(
    frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
) -> Optional[np.ndarray]:
    """
    Extract dominant HSV color from the TORSO region (30%-70% of bbox height).
    This avoids hair, face skin, and shoes/feet which contaminate jersey color.
    Returns a 3-element HSV mean array, or None if crop is too small.
    """
    h = y2 - y1
    torso_y1 = y1 + max(1, int(h * 0.25))
    torso_y2 = y1 + max(2, int(h * 0.65))

    # Slight horizontal inset to avoid background edges
    w = x2 - x1
    inset = max(1, int(w * 0.15))
    crop_x1 = x1 + inset
    crop_x2 = x2 - inset

    if crop_x2 <= crop_x1 or torso_y2 <= torso_y1:
        return None

    crop = frame[torso_y1:torso_y2, crop_x1:crop_x2]

    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None

    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Downsample pixels to speed up K-means: take every 4th pixel
    pixels = hsv_crop[::4, ::4, :].reshape(-1, 3)

    if len(pixels) < 10:
        return None

    # K-means with 2 clusters to find the dominant jersey color
    # (ignoring minor logo/stripe secondary color)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 2, None, crit, 3, cv2.KMEANS_RANDOM_CENTERS)

    counts = np.bincount(labels.flatten())
    dominant_idx = np.argmax(counts)

    return centers[dominant_idx]  # Shape: (3,) -> [H, S, V]


def update_observations(
    frame: np.ndarray, track_boxes: Dict[int, Tuple], frame_idx: int = 0
) -> None:
    """Accumulate HSV color and spatial X observations per track."""
    if frame_idx % 10 != 0:  # Sample every 10 frames for speed
        return

    frame_w = frame.shape[1]
    for tid, box_data in track_boxes.items():
        if tid in _locked:
            continue

        x1, y1, x2, y2 = box_data[:4]
        cx = ((x1 + x2) / 2.0) / max(1, frame_w)
        _x_history[tid].append(cx)

        hsv = _extract_torso_hsv(frame, x1, y1, x2, y2)
        if hsv is not None:
            _hsv_history[tid].append(hsv)


def _build_or_update_anchors(
    means_x: Dict[int, float], means_hsv: Dict[int, np.ndarray]
) -> None:
    """Establish global K-Means anchors once from accumulated data."""
    global _x_anchors, _hsv_anchors

    tids = [t for t in means_x if len(_x_history[t]) >= MIN_OBS and t in means_hsv]
    if len(tids) < 4:
        return

    # ── Spatial X anchors ──
    xs = np.array([means_x[t] for t in tids], dtype=np.float32).reshape(-1, 1)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    _, _, centers_x = cv2.kmeans(xs, 2, None, crit, 10, cv2.KMEANS_RANDOM_CENTERS)
    c0x, c1x = float(centers_x[0][0]), float(centers_x[1][0])

    if _x_anchors is None:
        _x_anchors = (min(c0x, c1x), max(c0x, c1x))

    # ── HSV color anchors ──
    hsv_arr = np.array([means_hsv[t] for t in tids], dtype=np.float32)
    _, labels_c, centers_hsv = cv2.kmeans(
        hsv_arr, 2, None, crit, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    if _hsv_anchors is None:
        # Align color clusters with spatial clusters for consistency
        # Team 0 = left side, Team 1 = right side
        c0_tids = [tids[i] for i in range(len(tids)) if labels_c[i][0] == 0]
        c1_tids = [tids[i] for i in range(len(tids)) if labels_c[i][0] == 1]

        mean_x_c0 = np.mean([means_x[t] for t in c0_tids]) if c0_tids else 0.5
        mean_x_c1 = np.mean([means_x[t] for t in c1_tids]) if c1_tids else 0.5

        if mean_x_c0 <= mean_x_c1:
            _hsv_anchors = (centers_hsv[0].copy(), centers_hsv[1].copy())
        else:
            _hsv_anchors = (centers_hsv[1].copy(), centers_hsv[0].copy())


def _hsv_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute perceptual HSV distance. Hue is circular (0-180 in OpenCV),
    so we handle wraparound. S and V are linear.
    """
    dh = min(abs(a[0] - b[0]), 180 - abs(a[0] - b[0]))  # Circular hue distance
    ds = abs(a[1] - b[1])
    dv = abs(a[2] - b[2])
    # Weighted: Hue is most discriminative for jerseys
    return float(dh * 2.0 + ds * 1.0 + dv * 0.5)


def _classify_single(
    tid: int, mean_x: float, mean_hsv: Optional[np.ndarray]
) -> Optional[int]:
    """Classify a single track using fused HSV color + spatial signal."""
    if _hsv_anchors is None or _x_anchors is None:
        return None

    a0h, a1h = _hsv_anchors
    a0x, a1x = _x_anchors

    # HSV color probability
    if mean_hsv is not None:
        d0c = _hsv_distance(mean_hsv, a0h)
        d1c = _hsv_distance(mean_hsv, a1h)
        tot_c = d0c + d1c
        prob_c1 = d0c / tot_c if tot_c > 0 else 0.5
    else:
        prob_c1 = 0.5

    # Spatial X probability
    d0x = abs(mean_x - a0x)
    d1x = abs(mean_x - a1x)
    tot_x = d0x + d1x
    prob_x1 = d0x / tot_x if tot_x > 0 else 0.5

    final_prob_1 = (COLOR_WEIGHT * prob_c1) + (SPATIAL_WEIGHT * prob_x1)

    return 0 if final_prob_1 < 0.5 else 1


def classify_teams() -> Dict[int, int]:
    """
    Classify all known tracks into team 0 or team 1.
    Uses temporally averaged HSV + spatial signals.
    """
    means_x: Dict[int, float] = {}
    means_hsv: Dict[int, np.ndarray] = {}

    for tid, buf in _x_history.items():
        if len(buf) >= MIN_OBS:
            means_x[tid] = float(np.mean(buf))

    for tid in means_x:
        h_buf = _hsv_history[tid]
        if len(h_buf) >= MIN_OBS:
            arr = np.array(h_buf)
            means_hsv[tid] = np.mean(arr, axis=0)  # Temporally averaged HSV

    _build_or_update_anchors(means_x, means_hsv)

    result = dict(_locked)

    for tid, mx in means_x.items():
        if tid in _locked:
            continue
        label = _classify_single(tid, mx, means_hsv.get(tid))
        if label is not None:
            result[tid] = label

    return result


def lock_team(track_id: int, team: int) -> None:
    """Permanently lock a track's team. Once locked, NEVER changes."""
    if track_id not in _locked:
        _locked[track_id] = team


def get_locked_teams() -> Dict[int, int]:
    return dict(_locked)


def reset() -> None:
    global _x_history, _hsv_history, _locked, _x_anchors, _hsv_anchors
    _x_history = defaultdict(lambda: deque(maxlen=OBS_WINDOW))
    _hsv_history = defaultdict(lambda: deque(maxlen=OBS_WINDOW))
    _locked = {}
    _x_anchors = None
    _hsv_anchors = None
