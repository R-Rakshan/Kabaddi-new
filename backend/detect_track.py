"""
detect_track.py
YOLOv8-Pose detection + ByteTrack multi-object tracking.
Integrated Full-Body structural validation and head estimation.
"""

from ultralytics import YOLO, RTDETR
import numpy as np
import court_detector
from reid_model import get_reid_extractor
import cv2
from typing import Tuple, List, Dict, Optional
import torch

# Shared model instances (lazy-loaded)

MAX_VELOCITY = 150  # Max pixels a player center can move in 1 frame
MAX_ACCEL = 80  # Max acceleration (velocity change) per frame in pixels
MAX_SIZE_CHANGE = 0.4  # Max fractional change in bbox area per frame (40%)
EMA_ALPHA = 0.6  # EMA alpha for bounding box smoothing
HEAD_EMA_ALPHA = 0.35

# Keypoint indices (COCO 17-point format)
KP_UPPER = [0, 1, 2, 3, 4, 5, 6]  # Nose, Eyes, Ears, Shoulders
KP_LOWER = [11, 12, 13, 14, 15, 16]  # Hips, Knees, Ankles

KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_HIP = 11
KP_R_HIP = 12

_model_pose = None
_model_rtdetr = None


def _get_model():
    global _model_pose, _model_rtdetr
    if _model_pose is None:
        _model_pose = YOLO("yolov8n-pose.pt")
    if _model_rtdetr is None:
        _model_rtdetr = RTDETR("rtdetr-l.pt")
    return _model_pose, _model_rtdetr


def calculate_iou(box1, box2):
    """Calculates IOU between two boxes [x1, y1, x2, y2]."""
    xi1, yi1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    xi2, yi2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    b1_a = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_a = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (b1_a + b2_a - inter + 1e-6)


def is_touching_edge(x1, y1, x2, y2, w, h, margin=5):
    return x1 <= margin or y1 <= margin or x2 >= w - margin or y2 >= h - margin


class KalmanBoxTracker:
    """
    Constant velocity Kalman Filter for a bounding box: [x, y, a, r]
    x, y: center, a: area, r: aspect ratio.
    """

    def __init__(self, box: Tuple[float, float, float, float]):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            np.float32,
        )

        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            np.float32,
        )

        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        self.kf.processNoiseCov[4:, 4:] *= 0.1  # Velocity noise

        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)

        # Initial state
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2
        self.kf.statePost = np.array(
            [cx, cy, w * h, w / h, 0, 0, 0, 0], np.float32
        ).reshape(-1, 1)

    def update(self, box: Tuple[float, float, float, float]) -> float:
        """Updates KF and returns the innovation (distance between prediction and measurement)."""
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2

        # Predict to get innovation
        pred = self.kf.predict()
        px, py = pred[0, 0], pred[1, 0]
        innovation = float(np.hypot(cx - px, cy - py))

        self.kf.correct(np.array([cx, cy, w * h, w / h], np.float32).reshape(-1, 1))
        return innovation

    def predict(self) -> Tuple[int, int, int, int]:
        pred = self.kf.predict()
        cx, cy, a, r = pred[0, 0], pred[1, 0], pred[2, 0], pred[3, 0]
        w = np.sqrt(max(1, a * r))
        h = np.sqrt(max(1, a / r))
        return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


def calculate_track_confidence(
    det_conf: float, reid_sim: float, innovation: float, max_inv: float = 150.0
) -> float:
    """Calculates a weighted confidence score for the track."""
    motion_score = max(0.0, 1.0 - (innovation / max_inv))
    score = (0.5 * det_conf) + (0.3 * reid_sim) + (0.2 * motion_score)
    return float(score)


class SceneAnalyzer:
    """Tracks temporal scene metrics to drive adaptive thresholding."""

    def __init__(self):
        self.density_ema = 0.5
        self.motion_ema = 0.2
        self.occlusion_ema = 0.0
        self.alpha = 0.1  # Slow adaptation

    def update(self, num_tracks: int, velocities: List[float], num_occluded: int):
        # Density (out of 14 players)
        curr_density = min(1.0, num_tracks / 14.0)
        self.density_ema = (
            1 - self.alpha
        ) * self.density_ema + self.alpha * curr_density

        # Motion (normalized by 50px/frame)
        curr_motion = min(1.0, np.mean(velocities) / 50.0) if velocities else 0.0
        self.motion_ema = (1 - self.alpha) * self.motion_ema + self.alpha * curr_motion

        # Occlusion (number of missing tracks being predicted)
        curr_occlusion = min(1.0, num_occluded / 14.0)
        self.occlusion_ema = (
            1 - self.alpha
        ) * self.occlusion_ema + self.alpha * curr_occlusion

    def get_thresholds(self) -> Dict[str, float]:
        """Returns adaptive YOLO and tracking thresholds."""
        # Lower confidence during high occlusion to catch partially visible players
        conf = 0.6 - (self.occlusion_ema * 0.15) - (self.density_ema * 0.05)
        conf = np.clip(conf, 0.45, 0.7)

        # Stricter IOU during high density to prevent ID swaps
        iou = 0.5 + (self.density_ema * 0.2)
        iou = np.clip(iou, 0.4, 0.7)

        # Adjust suppression: be stricter if density is high (noise pool is larger),
        # but more lenient if occlusion is high (to keep ghost tracks)
        suppress = 0.45 + (self.density_ema * 0.1) - (self.occlusion_ema * 0.1)
        suppress = np.clip(suppress, 0.4, 0.6)

        return {"conf": float(conf), "iou": float(iou), "suppress": float(suppress)}


def is_valid_court_transition(
    tid: int, curr_m: Tuple[float, float], tracker_state: dict
) -> bool:
    """Checks if a transition in meter space is physically possible in Kabaddi."""
    mx, my = curr_m
    if mx < -0.5 or mx > 13.5 or my < -0.5 or my > 10.5:
        return False
    if tid in tracker_state.get("last_m_pos", {}):
        pmx, pmy = tracker_state["last_m_pos"][tid]
        if (pmx < 6.5 and mx > 6.5) or (pmx > 6.5 and mx < 6.5):
            if abs(mx - pmx) > 3.0:
                return False
    return True


class GlobalMotionEstimator:
    """
    Estimates camera motion between frames using ORB keypoint matching and RANSAC.
    Ensures tracking is invariant to camera shake, pan, and zoom.
    """

    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_gray = None
        self.prev_kps = None
        self.prev_des = None
        self.last_transform = np.eye(2, 3, dtype=np.float32)

    def update(
        self, frame: np.ndarray, masks: List[Tuple[int, int, int, int]] = []
    ) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Mask out moving players to find static background features
        mask = np.ones_like(gray) * 255
        for x1, y1, x2, y2 in masks:
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 0, -1)

        kps, des = self.orb.detectAndCompute(gray, mask=mask)

        transform = np.eye(2, 3, dtype=np.float32)
        if self.prev_gray is not None and des is not None and self.prev_des is not None:
            matches = self.matcher.match(self.prev_des, des)
            if len(matches) > 10:
                src_pts = np.float32(
                    [self.prev_kps[m.queryIdx].pt for m in matches]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32([kps[m.trainIdx].pt for m in matches]).reshape(
                    -1, 1, 2
                )

                # Use Affine Transform for Pan/Zoom/Rotate/Shake robustness
                M, inliers = cv2.estimateAffinePartial2D(
                    src_pts, dst_pts, method=cv2.RANSAC
                )
                if M is not None:
                    transform = M

        self.prev_gray = gray
        self.prev_kps = kps
        self.prev_des = des
        self.last_transform = transform
        return transform


def detect_and_track(
    frame: np.ndarray,
    tracker_state: dict,
    frame_idx: int = 0,
    active_raider_tid: Optional[int] = None,
) -> tuple:
    """
    Run YOLOv8-Pose + ByteTrack on a single frame.
    Supports interval-based detection skipping to boost performance.

    Returns:
        active_boxes: {track_id: (x1, y1, x2, y2, conf, emb)}
        head_positions_output: {track_id: (head_x, head_y)}
        foot_positions_output: {track_id: (foot_x, foot_y)}
        court_positions: {track_id: bool} - True if player is within court boundaries
    """
    model_pose, model_rtdetr = _get_model()

    # ── State Initialization ──────────────────────────────────────
    if "tracker" not in tracker_state:
        tracker_state["tracker"] = None
    if "box_ema" not in tracker_state:
        tracker_state["box_ema"] = {}
    if "last_pos" not in tracker_state:
        tracker_state["last_pos"] = {}
    if "velocity" not in tracker_state:
        tracker_state["velocity"] = {}
    if "miss_count" not in tracker_state:
        tracker_state["miss_count"] = {}
    if "last_emb" not in tracker_state:
        tracker_state["last_emb"] = {}
    if "head_ema" not in tracker_state:
        tracker_state["head_ema"] = {}
    if "last_area" not in tracker_state:
        tracker_state["last_area"] = {}
    if "kf" not in tracker_state:
        tracker_state["kf"] = {}
    if "track_conf" not in tracker_state:
        tracker_state["track_conf"] = {}
    if "scene_analyzer" not in tracker_state:
        tracker_state["scene_analyzer"] = SceneAnalyzer()
    if "last_m_pos" not in tracker_state:
        tracker_state["last_m_pos"] = {}
    if "ensemble_confirmed" not in tracker_state:
        tracker_state["ensemble_confirmed"] = set()
    if "ensemble_frame" not in tracker_state:
        tracker_state["ensemble_frame"] = 0
    if "gme" not in tracker_state:
        tracker_state["gme"] = GlobalMotionEstimator()

    tracker_state["ensemble_frame"] += 1
    sa = tracker_state["scene_analyzer"]
    frame_h, frame_w = frame.shape[:2]

    # ── 1. Global Motion Compensation (GMC) ───────────────────────
    prev_boxes = [tracker_state["box_ema"][tid] for tid in tracker_state["box_ema"]]
    gmc_M = tracker_state["gme"].update(frame, prev_boxes)
    dx, dy = gmc_M[0, 2], gmc_M[1, 2]

    # ── 2. Scene Analysis & Adaptive Thresholding ──────────────────
    active_velocities = [
        np.hypot(v[0], v[1]) for v in tracker_state["velocity"].values()
    ]
    num_occluded = sum(1 for m in tracker_state["miss_count"].values() if m > 0)
    sa.update(len(tracker_state["box_ema"]), active_velocities, num_occluded)
    thresholds = sa.get_thresholds()

    # ── 2. ROI Optimization ──────────────────────────────────────
    if court_detector._court_mask is None:
        court_detector.init_court_mask(frame)

    roi_box = None
    if court_detector._court_mask is not None:
        # Get bounding box of the court mask
        coords = np.argwhere(court_detector._court_mask > 0)
        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            # Add margin
            margin = 20
            roi_box = (
                max(0, x_min - margin),
                max(0, y_min - margin),
                min(frame_w, x_max + margin),
                min(frame_h, y_max + margin),
            )

    # ── 3. Optimized Interval Inference (Every 5 frames) ───────────
    # Skip frames unless scene is complex or motion spike.
    # In between frames, rely on Kalman prediction + GMC.
    is_complex = (sa.density_ema > 0.8) or (sa.occlusion_ema > 0.15)
    is_motion_spike = abs(dx) > 30 or abs(dy) > 30

    should_run_yolo = (frame_idx % 5 == 0) or is_complex or is_motion_spike
    should_run_rtdetr = False  # Disabled - too slow for CPU

    results_pose = None
    if should_run_yolo:
        # Downscale for faster CPU inference
        inf_frame = frame
        h, w = frame.shape[:2]
        if h > 480:
            inf_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

        # If ROI available, crop to smaller region
        if roi_box:
            rx1, ry1, rx2, ry2 = roi_box
            # Scale ROI to downscaled coords
            scale_x, scale_y = 640 / w, 480 / h
            inf_frame = inf_frame[
                int(ry1 * scale_y) : int(ry2 * scale_y),
                int(rx1 * scale_x) : int(rx2 * scale_x),
            ]

        # CPU optimization: half=False, no augment
        results_pose = model_pose.track(
            inf_frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=max(0.4, thresholds["conf"]),  # Lower conf for CPU
            iou=thresholds["iou"],
            verbose=False,
            half=False,  # CPU doesn't benefit from FP16
            augment=False,
        )

    rtdetr_boxes = []
    if should_run_rtdetr:
        results_rtdetr = model_rtdetr(
            frame, classes=[0], conf=0.45, verbose=False, half=True
        )
        rtdetr_boxes = (
            results_rtdetr[0].boxes.xyxy.cpu().numpy()
            if (results_rtdetr and len(results_rtdetr) > 0)
            else []
        )
        tracker_state["ensemble_confirmed"] = set()

    active_boxes = {}
    head_positions_output = {}
    foot_positions_output = {}
    court_positions = {}
    current_tids = set()

    # ── 4. Process Active Detections ─────────────────────────────
    if results_pose and len(results_pose) > 0 and results_pose[0].boxes is not None:
        r = results_pose[0].boxes
        if results_pose[0].keypoints is not None:
            kps_conf = results_pose[0].keypoints.conf.cpu().numpy()
            kps_xy = results_pose[0].keypoints.xy.cpu().numpy()

            if r.id is not None:
                for i in range(len(r)):
                    tid = int(r.id[i].item())
                    yolo_box = r.xyxy[i].cpu().numpy().astype(float)

                    # Apply ROI shift to box
                    if roi_box:
                        yolo_box[0] += roi_box[0]
                        yolo_box[2] += roi_box[0]
                        yolo_box[1] += roi_box[1]
                        yolo_box[3] += roi_box[1]

                    # Consensus Check: Raider bypasses part of it to ensure continuity
                    is_confirmed = tid == active_raider_tid
                    if not is_confirmed:
                        if should_run_rtdetr:
                            for rbox in rtdetr_boxes:
                                if calculate_iou(yolo_box, rbox) > 0.4:
                                    is_confirmed = True
                                    tracker_state["ensemble_confirmed"].add(tid)
                                    break
                        else:
                            if tid in tracker_state["ensemble_confirmed"]:
                                is_confirmed = True

                    if not is_confirmed:
                        continue

                    # Pose-Based Validation
                    k_conf = kps_conf[i]
                    if not (
                        any(k_conf[idx] > 0.4 for idx in KP_UPPER)
                        and any(k_conf[idx] > 0.4 for idx in KP_LOWER)
                    ):
                        continue

                    # Apply ROI shift to keypoints
                    curr_kps_xy = kps_xy[i].copy()
                    if roi_box:
                        curr_kps_xy[:, 0] += roi_box[0]
                        curr_kps_xy[:, 1] += roi_box[1]

                    x1, y1, x2, y2 = yolo_box
                    conf = float(r.conf[i].item())
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    curr_m = court_detector.image_to_court(cx, cy)

                    # Court Constraint
                    if not is_valid_court_transition(tid, curr_m, tracker_state):
                        continue

                    # Check if player is within court boundaries
                    in_court = court_detector.is_in_court(int(cx), int(cy))
                    court_positions[tid] = in_court

                    # Update State
                    current_tids.add(tid)
                    tracker_state["miss_count"][tid] = 0
                    tracker_state["last_m_pos"][tid] = curr_m

                    # Kalman Update
                    if tid not in tracker_state["kf"]:
                        tracker_state["kf"][tid] = KalmanBoxTracker((x1, y1, x2, y2))
                    tracker_state["kf"][tid].update((x1, y1, x2, y2))

                    # Smooth BBox (Ultra-Fine for Raider)
                    curr_ema = 0.3 if tid == active_raider_tid else EMA_ALPHA
                    if tid in tracker_state["box_ema"]:
                        px1, py1, px2, py2 = tracker_state["box_ema"][tid]
                        x1 = px1 * curr_ema + x1 * (1 - curr_ema)
                        y1 = py1 * curr_ema + y1 * (1 - curr_ema)
                        x2 = px2 * curr_ema + x2 * (1 - curr_ema)
                        y2 = py2 * curr_ema + y2 * (1 - curr_ema)
                    tracker_state["box_ema"][tid] = (x1, y1, x2, y2)

                    # Head Position (Ultra-Fine for Raider)
                    curr_head_ema = 0.6 if tid == active_raider_tid else HEAD_EMA_ALPHA
                    ls, rs = curr_kps_xy[KP_L_SHOULDER], curr_kps_xy[KP_R_SHOULDER]
                    lh, rh = curr_kps_xy[KP_L_HIP], curr_kps_xy[KP_R_HIP]
                    shoulder_mid = (ls + rs) / 2.0
                    hip_mid = (lh + rh) / 2.0
                    head = shoulder_mid + 0.55 * (shoulder_mid - hip_mid)

                    if tid in tracker_state["head_ema"]:
                        hx, hy = tracker_state["head_ema"][tid]
                        hx = hx * curr_head_ema + head[0] * (1 - curr_head_ema)
                        hy = hy * curr_head_ema + head[1] * (1 - curr_head_ema)
                        tracker_state["head_ema"][tid] = (hx, hy)
                    else:
                        tracker_state["head_ema"][tid] = (head[0], head[1])

                    head_positions_output[tid] = (
                        int(tracker_state["head_ema"][tid][0]),
                        int(tracker_state["head_ema"][tid][1]),
                    )

                    # Foot Position
                    l_ank, r_ank = curr_kps_xy[15], curr_kps_xy[16]
                    foot = l_ank if l_ank[1] > r_ank[1] else r_ank
                    foot_positions_output[tid] = (int(foot[0]), int(foot[1]))

                    # Embedding
                    x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
                    tx1, ty1 = x1_i + (x2_i - x1_i) // 4, y1_i + (y2_i - y1_i) // 8
                    tx2, ty2 = x2_i - (x2_i - x1_i) // 4, y1_i + (y2_i - y1_i) // 2
                    torso = frame[
                        max(0, ty1) : min(frame_h, ty2), max(0, tx1) : min(frame_w, tx2)
                    ]
                    emb = (
                        torso.mean(axis=(0, 1)) / 255.0
                        if torso.size > 0
                        else np.zeros(3)
                    )
                    tracker_state["last_emb"][tid] = emb

                    # Confidence Score
                    c_score = calculate_track_confidence(conf, 1.0, 0.0)
                    tracker_state["track_conf"][tid] = c_score

                    active_boxes[tid] = (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                        c_score,
                        emb,
                    )

    # ── 5. Occlusion Recovery (GMC-Aware) ─────────────────────────
    is_tackle = (sa.density_ema > 0.8) or (sa.occlusion_ema > 0.15)
    max_miss_frames = 60 if is_tackle else 40

    for tid in list(tracker_state["miss_count"].keys()):
        if tid in current_tids:
            continue

        miss = tracker_state["miss_count"][tid] + 1
        tracker_state["miss_count"][tid] = miss

        if miss > max_miss_frames:
            for key in [
                "box_ema",
                "last_pos",
                "velocity",
                "last_emb",
                "miss_count",
                "head_ema",
                "last_area",
                "kf",
                "track_conf",
                "last_m_pos",
                "ensemble_confirmed",
            ]:
                if tid in tracker_state.get(key, {}):
                    del tracker_state[key][tid]
            continue

        pred_box = tracker_state["kf"][tid].predict()
        x1_p, y1_p, x2_p, y2_p = pred_box
        pred_box_gmc = (x1_p + dx, y1_p + dy, x2_p + dx, y2_p + dy)

        cx_p, cy_p = (
            (pred_box_gmc[0] + pred_box_gmc[2]) / 2,
            (pred_box_gmc[1] + pred_box_gmc[3]) / 2,
        )
        curr_m_p = court_detector.image_to_court(cx_p, cy_p)

        if not is_valid_court_transition(tid, curr_m_p, tracker_state):
            tracker_state["miss_count"][tid] = max_miss_frames + 1
            continue

        v_conf = tracker_state["track_conf"].get(tid, 0.4) * 0.95
        tracker_state["track_conf"][tid] = v_conf

        if v_conf >= 0.3 and tid in tracker_state["last_emb"]:
            active_boxes[tid] = (
                int(pred_box_gmc[0]),
                int(pred_box_gmc[1]),
                int(pred_box_gmc[2]),
                int(pred_box_gmc[3]),
                v_conf,
                tracker_state["last_emb"][tid],
            )
            tracker_state["box_ema"][tid] = pred_box_gmc
            tracker_state["last_pos"][tid] = (cx_p, cy_p)
            tracker_state["last_m_pos"][tid] = curr_m_p
            if tid in tracker_state["head_ema"]:
                head_positions_output[tid] = (
                    int(tracker_state["head_ema"][tid][0]),
                    int(tracker_state["head_ema"][tid][1]),
                )

            # Court position for recovered track
            in_court_p = court_detector.is_in_court(int(cx_p), int(cy_p))
            court_positions[tid] = in_court_p

    return active_boxes, head_positions_output, foot_positions_output, court_positions
