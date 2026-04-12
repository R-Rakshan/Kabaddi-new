"""
pose_estimator.py
YOLOv8-Pose keypoint extraction + head position estimation with EMA smoothing.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional

_pose_model = None

# EMA smoothed head positions per track_id
_head_ema: Dict[int, np.ndarray] = {}
HEAD_EMA_ALPHA = 0.35

# Keypoint indices (COCO 17-point format)
KP_NOSE       = 0
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_HIP      = 11
KP_R_HIP      = 12


def _get_pose_model():
    global _pose_model
    if _pose_model is None:
        from ultralytics import YOLO
        _pose_model = YOLO("yolov8n-pose.pt")
    return _pose_model


def _iou(a: Tuple, b: Tuple) -> float:
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


def estimate_heads(
    frame: np.ndarray,
    track_boxes: Dict[int, Tuple]
) -> Dict[int, Tuple[int, int]]:
    """
    Run YOLOv8-Pose on the frame, match predictions to tracked boxes via IoU,
    compute head position for each matched track, apply EMA smoothing.

    Returns:
        dict: track_id -> (head_x, head_y) in pixel coords
    """
    model = _get_pose_model()
    results = model(frame, verbose=False, conf=0.3)

    # Build pose detections list: (box, keypoints array)
    pose_dets = []
    if results and results[0].boxes is not None and results[0].keypoints is not None:
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        kps_xy = results[0].keypoints.xy.cpu().numpy()   # (N, 17, 2)
        for i in range(len(boxes_xyxy)):
            pose_dets.append((boxes_xyxy[i], kps_xy[i]))

    head_positions: Dict[int, Tuple[int, int]] = {}

    for tid, box_data in track_boxes.items():
        tx1, ty1, tx2, ty2 = box_data[:4]
        best_iou = 0.3  # minimum IoU threshold
        best_kps = None

        for (pb, kps) in pose_dets:
            score = _iou((tx1, ty1, tx2, ty2), tuple(pb.astype(int)))
            if score > best_iou:
                best_iou = score
                best_kps = kps

        if best_kps is None:
            # Fallback: estimate head as top-center of bounding box
            head = np.array([(tx1 + tx2) / 2.0, ty1 + (ty2 - ty1) * 0.1])
        else:
            ls = best_kps[KP_L_SHOULDER]
            rs = best_kps[KP_R_SHOULDER]
            lh = best_kps[KP_L_HIP]
            rh = best_kps[KP_R_HIP]

            shoulder_mid = (ls + rs) / 2.0
            hip_mid = (lh + rh) / 2.0

            # Project upward from shoulder using torso vector
            torso = shoulder_mid - hip_mid
            head = shoulder_mid + 0.55 * torso

        # EMA smoothing
        if tid in _head_ema:
            head = HEAD_EMA_ALPHA * head + (1 - HEAD_EMA_ALPHA) * _head_ema[tid]
        _head_ema[tid] = head

        head_positions[tid] = (int(head[0]), int(head[1]))

    return head_positions


def reset():
    global _head_ema
    _head_ema = {}
