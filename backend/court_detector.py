"""
court_detector.py  — v2
=======================
Homography-based tracking in real-world court coordinates.

All player positions are transformed from image pixels to real-world
13m x 10m Kabaddi court coordinates via a perspective homography.
Trajectory smoothing, jitter rejection, and velocity validation all
operate in this meter-based coordinate system, completely eliminating
perspective distortion artifacts.

Radar output: 650px x 500px  (1 pixel ≈ 2cm)
"""

import cv2
import numpy as np
import json
from collections import defaultdict, deque
from typing import Dict, Tuple, List, Optional

# ── Real-world court dimensions (meters) ──────────────────────────────────
COURT_LENGTH_M = 13.0  # X-axis
COURT_WIDTH_M  = 10.0  # Y-axis

# Radar pixel dimensions
RADAR_W = 650
RADAR_H = 500

# Court lines in meters
MIDLINE_M       = COURT_LENGTH_M / 2.0          # 6.5m
BAULK_LEFT_M    = MIDLINE_M - 3.75              # 2.75m
BAULK_RIGHT_M   = MIDLINE_M + 3.75              # 10.25m
BONUS_LEFT_M    = BAULK_LEFT_M - 1.0            # 1.75m
BONUS_RIGHT_M   = BAULK_RIGHT_M + 1.0           # 11.25m

# Colors
C_BG    = (230, 240, 230)
C_LINES = (50, 150, 50)
C_MID   = (0, 0, 200)
C_TEXT  = (30, 30, 30)

TEAM_COLORS = {
    0: (0, 60, 230),    # Team 1 (red)
    1: (220, 100, 0),   # Team 2 (blue)
}

# ── Real-world smoothing thresholds ───────────────────────────────────────
MAX_SPEED_M_PER_FRAME = 0.8  # Max realistic player speed per frame (~20m/s at 25fps)

# ── Module state ──────────────────────────────────────────────────────────
# Trajectories stored in METERS (real-world court coords)
_trajectories: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
_official_ids: Dict[int, str] = {}
_homography_matrix: np.ndarray = None
_court_mask: np.ndarray = None

# Per-track EMA positions in meter space for smooth radar dots
_pos_ema: Dict[int, np.ndarray] = {}
POS_EMA_ALPHA = 0.5

_cached_radar_bg = None


def init_court_mask(frame: np.ndarray):
    global _court_mask
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sample = hsv[int(h*0.5):int(h*0.8), int(w*0.3):int(w*0.7)]
    pixels = sample.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(pixels, 1, None, crit, 3, cv2.KMEANS_RANDOM_CENTERS)
    dom_hsv = centers[0]
    
    lower_bound = np.array([max(0, dom_hsv[0] - 15), max(0, dom_hsv[1] - 50), max(0, dom_hsv[2] - 50)], dtype=np.uint8)
    upper_bound = np.array([min(180, dom_hsv[0] + 15), min(255, dom_hsv[1] + 50), min(255, dom_hsv[2] + 50)], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        _court_mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(largest_contour)
        cv2.drawContours(_court_mask, [hull], -1, 255, thickness=cv2.FILLED)
    else:
        _court_mask = np.ones((h, w), dtype=np.uint8) * 255

def is_in_court(x: int, y: int) -> bool:
    global _court_mask
    if _court_mask is None:
        return True
    h, w = _court_mask.shape
    if 0 <= y < h and 0 <= x < w:
        return _court_mask[y, x] > 0
    return False

def init_homography(frame_w: int, frame_h: int):
    """
    Initialize homography: image pixels -> real-world meters (13m x 10m).
    Source: approximate court trapezoid in the broadcast frame.
    Destination: actual court rectangle in meters.
    """
    global _homography_matrix
    src_pts = np.array([
        [int(frame_w * 0.15), int(frame_h * 0.35)],  # Top-left
        [int(frame_w * 0.85), int(frame_h * 0.35)],  # Top-right
        [int(frame_w * 0.95), int(frame_h * 0.95)],  # Bottom-right
        [int(frame_w * 0.05), int(frame_h * 0.95)]   # Bottom-left
    ], dtype=np.float32)

    # Map directly to real-world court meters
    dst_pts = np.array([
        [0.0,            COURT_WIDTH_M],  # Top-left  (0m, 10m)
        [COURT_LENGTH_M, COURT_WIDTH_M],  # Top-right (13m, 10m)
        [COURT_LENGTH_M, 0.0],            # Bottom-right (13m, 0m)
        [0.0,            0.0]             # Bottom-left  (0m, 0m)
    ], dtype=np.float32)

    _homography_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)


def image_to_court(x: float, y: float) -> Tuple[float, float]:
    """
    Transform a point from image pixel coordinates to real-world
    court coordinates in meters. Returns (court_x_m, court_y_m).
    """
    if _homography_matrix is None:
        return (COURT_LENGTH_M / 2.0, COURT_WIDTH_M / 2.0)
        
    pt = np.array([[[x, y]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, _homography_matrix)
    cx_m, cy_m = mapped[0][0]
    
    # Clamp to court boundaries
    cx_m = max(0.0, min(COURT_LENGTH_M, float(cx_m)))
    cy_m = max(0.0, min(COURT_WIDTH_M, float(cy_m)))
    return (cx_m, cy_m)


def _meters_to_radar(cx_m: float, cy_m: float) -> Tuple[int, int]:
    """Convert meter coordinates to radar pixel coordinates."""
    rx = int((cx_m / COURT_LENGTH_M) * RADAR_W)
    rx = max(0, min(RADAR_W - 1, rx))
    
    # Invert Y: court 0m=bottom, radar 0px=top
    ry = int((1.0 - cy_m / COURT_WIDTH_M) * RADAR_H)
    ry = max(0, min(RADAR_H - 1, ry))
    return (rx, ry)


def _draw_court() -> np.ndarray:
    """Draw the static 2D radar background with real court line positions."""
    global _cached_radar_bg
    if _cached_radar_bg is not None:
        return _cached_radar_bg.copy()
        
    radar = np.ones((RADAR_H, RADAR_W, 3), dtype=np.uint8) * np.array(C_BG, dtype=np.uint8)
    
    # Border
    cv2.rectangle(radar, (0, 0), (RADAR_W - 1, RADAR_H - 1), C_LINES, 4)
    
    # Midline
    mx, _ = _meters_to_radar(MIDLINE_M, 0)
    cv2.line(radar, (mx, 0), (mx, RADAR_H), C_MID, 4)
    
    # Baulk lines
    blx, _ = _meters_to_radar(BAULK_LEFT_M, 0)
    brx, _ = _meters_to_radar(BAULK_RIGHT_M, 0)
    cv2.line(radar, (blx, 0), (blx, RADAR_H), C_LINES, 2)
    cv2.line(radar, (brx, 0), (brx, RADAR_H), C_LINES, 2)
    
    # Bonus lines
    bolx, _ = _meters_to_radar(BONUS_LEFT_M, 0)
    borx, _ = _meters_to_radar(BONUS_RIGHT_M, 0)
    cv2.line(radar, (bolx, 0), (bolx, RADAR_H), C_LINES, 2)
    # Cache background once complete
    _cached_radar_bg = radar.copy()
    return radar


def _smooth_trajectory_meters(path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Smooth trajectory in METER space. All distance checks and filtering
    operate on real-world distances, completely free of perspective distortion.
    """
    if len(path) < 10:
        return path
        
    # Step 1: Jitter rejection in meter space
    filtered = [path[0]]
    for i in range(1, len(path)):
        dx = filtered[-1][0] - path[i][0]
        dy = filtered[-1][1] - path[i][1]
        dist_m = np.hypot(dx, dy)
        
        # Reject impossible jumps (>0.8m in a single frame at 25fps = 20m/s)
        if dist_m < MAX_SPEED_M_PER_FRAME:
            filtered.append(path[i])
            
    if len(filtered) < 11:
        return filtered
        
    # Step 2: Hanning window low-pass in meter space
    xs = [p[0] for p in filtered]
    ys = [p[1] for p in filtered]
    
    window = np.hanning(11)
    window = window / window.sum()
    
    smooth_xs = np.convolve(xs, window, mode='valid')
    smooth_ys = np.convolve(ys, window, mode='valid')
    
    pad_start = 5
    pad_end = max(0, len(filtered) - len(smooth_xs) - pad_start)
    
    final_xs = [xs[0]] * pad_start + list(smooth_xs) + [xs[-1]] * pad_end
    final_ys = [ys[0]] * pad_start + list(smooth_ys) + [ys[-1]] * pad_end
    
    return list(zip(final_xs, final_ys))


def update_and_draw(
    frame_w: int, 
    frame_h: int, 
    track_boxes: Dict[int, Tuple], 
    team_labels: Dict[int, int],
    official_ids: Dict[int, str],
    active_raider_tid: Optional[int] = None,
    foot_positions: Dict[int, Tuple] = {}
) -> np.ndarray:
    """
    Transform player positions to real-world meters, update trajectories,
    and render the radar diagram. Uses exact foot placement for raider near bonus lines.
    """
    if _homography_matrix is None:
        init_homography(frame_w, frame_h)
        
    radar = _draw_court()
    
    # Transform and record player positions in meter space
    for tid, box_data in track_boxes.items():
        if tid not in official_ids:
            continue
            
        is_raider = (tid == active_raider_tid)
        
        # Determine anchor point (Exact Feet for Raider near Bonus Line)
        if is_raider and tid in foot_positions:
            fx, fy = foot_positions[tid]
            cx_m_raw, cy_m_raw = image_to_court(fx, fy)
            
            # Use exact foot only if near bonus lines (1.75m or 11.25m)
            dist_to_bonus = min(abs(cx_m_raw - 1.75), abs(cx_m_raw - 11.25))
            if dist_to_bonus < 0.6: # Within 60cm of bonus line
                court_x_m, court_y_m = cx_m_raw, cy_m_raw
            else:
                x1, y1, x2, y2 = box_data[:4]
                court_x_m, court_y_m = image_to_court((x1+x2)/2.0, float(y2))
        else:
            x1, y1, x2, y2 = box_data[:4]
            court_x_m, court_y_m = image_to_court((x1+x2)/2.0, float(y2))
        
        # EMA smooth (Responsive for Raider)
        curr_alpha = 0.8 if is_raider else POS_EMA_ALPHA
        pt = np.array([court_x_m, court_y_m])
        if tid in _pos_ema:
            pt = curr_alpha * pt + (1 - curr_alpha) * _pos_ema[tid]
        _pos_ema[tid] = pt
        
        _trajectories[tid].append((float(pt[0]), float(pt[1])))
        _official_ids[tid] = official_ids[tid]
        
    # Render trajectories (non-raiders first, raider on top)
    draw_order = sorted(list(_trajectories.keys()), key=lambda t: t == active_raider_tid)
    
    for tid in draw_order:
        if tid not in track_boxes:
            continue
            
        meter_path = _trajectories[tid]
        smoothed = _smooth_trajectory_meters(meter_path[-150:])
        if len(smoothed) < 2:
            continue
            
        team = team_labels.get(tid, -1)
        color = TEAM_COLORS.get(team, (120, 120, 120))
        is_raider = (tid == active_raider_tid)
        
        if active_raider_tid is not None and not is_raider:
            color = (color[0]//2 + 100, color[1]//2 + 100, color[2]//2 + 100)
            thick = 1
        elif is_raider:
            color = (0, 210, 255) # Raider Gold/Cyan
            thick = 3
        else:
            thick = 2
        
        pts = [_meters_to_radar(mx, my) for mx, my in smoothed]
        pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(radar, [pts_arr], isClosed=False, color=color, thickness=thick, lineType=cv2.LINE_AA)
        
        curr_pos = pts[-1]
        cv2.circle(radar, curr_pos, 8, color, -1)
        cv2.circle(radar, curr_pos, 8, (255, 255, 255), 1)
        
        label = _official_ids.get(tid, f"#{tid}")
        if is_raider: label = "R"
            
        cv2.putText(radar, label, (curr_pos[0] - 15, curr_pos[1] - 12),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, C_TEXT, 1, cv2.LINE_AA)
    return radar


def export_data(out_png: str, out_json: str, team_labels: Dict[int, int]):
    """Export trajectory data and professional metrics as JSON and a plot PNG."""
    export_dict = {}
    FPS = 25.0
    
    for tid, meter_path in _trajectories.items():
        path = _smooth_trajectory_meters(meter_path)
        if len(path) < 2:
            continue
            
        oid = _official_ids.get(tid, f"ID#{tid}")
        
        # ── Professional Metrics Calculation ────────────────────────────
        dist_total_m = 0.0
        crossings = 0
        for i in range(1, len(path)):
            p1, p2 = path[i-1], path[i]
            dist_total_m += np.hypot(p2[0] - p1[0], p2[1] - p1[1])
            
            # Midline crossing (6.5m)
            if (p1[0] < 6.5 and p2[0] > 6.5) or (p1[0] > 6.5 and p2[0] < 6.5):
                crossings += 1
                
        duration_s = len(path) / FPS
        avg_speed_m_s = dist_total_m / duration_s if duration_s > 0 else 0.0
        # ────────────────────────────────────────────────────────────

        export_dict[oid] = {
            "metrics": {
                "total_distance_m": round(dist_total_m, 2),
                "avg_speed_m_s": round(avg_speed_m_s, 2),
                "midline_crossings": crossings
            },
            "trajectory": [{"x_m": round(p[0], 3), "y_m": round(p[1], 3)} for p in path]
        }
        
    with open(out_json, 'w') as f:
        json.dump(export_dict, f, indent=2)

    # Render full trajectory plot
    radar = _draw_court()
    for tid, meter_path in _trajectories.items():
        path = _smooth_trajectory_meters(meter_path)
        if len(path) < 2:
            continue
        
        team = team_labels.get(tid, -1)
        color = TEAM_COLORS.get(team, (120, 120, 120))
        
        pts = [_meters_to_radar(mx, my) for mx, my in path]
        pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(radar, [pts_arr], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
        
        curr_pos = pts[-1]
        cv2.circle(radar, curr_pos, 8, color, -1)
        cv2.circle(radar, curr_pos, 8, (255, 255, 255), 1)
        
        oid = _official_ids.get(tid, f"ID#{tid}")
        cv2.putText(radar, oid, (curr_pos[0] - 15, curr_pos[1] - 12),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, C_TEXT, 1, cv2.LINE_AA)
                    
    cv2.imwrite(out_png, radar)


def reset():
    global _trajectories, _official_ids, _homography_matrix, _court_mask, _pos_ema
    _trajectories = defaultdict(list)
    _official_ids = {}
    _homography_matrix = None
    _court_mask = None
    _pos_ema = {}

