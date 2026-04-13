"""
annotator.py  — v1
==================
Video annotation module for Kabaddi player tracking.

Draws:
  - Bounding boxes (colored by team, gray for out-of-court)
  - Head label pills with official IDs (T1#1, T2#3, etc.)
  - Raider highlight indicators
  - Scene metrics overlay
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List


# ── Colors ────────────────────────────────────────────────────────────────────
TEAM_COLORS = {
    0: (50, 80, 220),  # Team 1 - Red-ish (BGR)
    1: (200, 120, 50),  # Team 2 - Blue-ish (BGR)
}

OUT_OF_COURT_COLOR = (100, 100, 100)  # Gray
UNKNOWN_COLOR = (150, 150, 150)  # Light gray

RAIDER_COLOR = (0, 230, 255)  # Gold/Cyan for raider

TEXT_COLOR = (255, 255, 255)  # White text
LABEL_BG_TEAM1 = (60, 50, 180)  # Dark red background
LABEL_BG_TEAM2 = (120, 80, 40)  # Dark blue background
LABEL_BG_UNKNOWN = (80, 80, 80)  # Gray background

# ── Label dimensions ──────────────────────────────────────────────────────────
LABEL_PADDING = 4
LABEL_HEIGHT = 18
LABEL_MIN_WIDTH = 40


# ── State ─────────────────────────────────────────────────────────────────────
_state = {
    "frame_idx": 0,
}


def reset():
    """Reset annotation state for new video processing."""
    global _state
    _state = {"frame_idx": 0}


def _draw_rounded_rect(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    radius: int = 8,
    thickness: int = 2,
):
    """
    Draw a rounded rectangle on the image.
    """
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)

    cv2.ellipse(
        img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness
    )
    cv2.ellipse(
        img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness
    )
    cv2.ellipse(
        img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness
    )
    cv2.ellipse(
        img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness
    )


def _draw_filled_rounded_rect(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    radius: int = 8,
):
    """
    Draw a filled rounded rectangle on the image.
    """
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return

    img[y1:y2, x1:x2] = color

    mask = np.zeros((h + 2 * radius, w + 2 * radius), dtype=np.uint8)
    roi = mask[radius : radius + h, radius : radius + w]

    cv2.rectangle(
        mask, (x1 + radius, y1 + radius), (x2 + radius - 1, y2 + radius - 1), 255, -1
    )
    mask_circle = np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    cv2.circle(mask_circle, (radius, radius), radius, 255, -1)

    mask[radius : radius + h, radius : radius + w] = cv2.bitwise_and(
        mask[radius : radius + h, radius : radius + w].astype(np.uint8),
        cv2.bitwise_not(mask_circle).astype(np.uint8),
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(img, contours, -1, color, -1, offset=(-radius, -radius))


def _get_label_text(
    official_id: Optional[str],
    team: Optional[int],
    is_in_court: bool,
    has_official_id: bool,
) -> Tuple[str, Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Determine label text, text color, and background color based on player state.

    Returns: (text, text_color, bg_color)
    """
    if has_official_id and official_id:
        text = official_id
    elif team is not None:
        if is_in_court:
            text = f"T{team + 1}?"
        else:
            text = "?"
    else:
        text = "?"

    if has_official_id:
        if team == 0:
            return text, TEXT_COLOR, LABEL_BG_TEAM1
        elif team == 1:
            return text, TEXT_COLOR, LABEL_BG_TEAM2
    elif team is not None:
        return text, TEXT_COLOR, LABEL_BG_UNKNOWN

    return text, TEXT_COLOR, LABEL_BG_UNKNOWN


def _draw_head_label(
    img: np.ndarray,
    x: int,
    y: int,
    official_id: Optional[str],
    team: Optional[int],
    is_in_court: bool,
    is_raider: bool,
    has_official_id: bool,
):
    """
    Draw a pill-shaped label above the player's head position.

    Args:
        img: Image to draw on
        x, y: Head position (center of label)
        official_id: Official ID string (e.g., "T1#1")
        team: Team number (0 or 1) or None
        is_in_court: Whether player is within court boundaries
        is_raider: Whether this player is the active raider
        has_official_id: Whether official ID has been assigned
    """
    text, text_color, bg_color = _get_label_text(
        official_id, team, is_in_court, has_official_id
    )

    if is_raider:
        bg_color = (0, 60, 80)  # Darker cyan for raider

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thickness = 1

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

    actual_w = max(LABEL_MIN_WIDTH, text_w + LABEL_PADDING * 2)
    actual_h = LABEL_HEIGHT

    label_x1 = x - actual_w // 2
    label_y1 = y - actual_h - 5

    label_x2 = label_x1 + actual_w
    label_y2 = label_y1 + actual_h

    h, w = img.shape[:2]
    if label_y1 < 0:
        label_y1 = y + 5
        label_y2 = label_y1 + actual_h

    label_x1 = max(0, label_x1)
    label_x2 = min(w, label_x2)
    label_y1 = max(0, label_y1)
    label_y2 = min(h, label_y2)

    _draw_filled_rounded_rect(
        img, label_x1, label_y1, label_x2, label_y2, bg_color, radius=4
    )

    text_x = label_x1 + (actual_w - text_w) // 2
    text_y = label_y1 + (actual_h + text_h) // 2 - baseline // 2

    if is_raider:
        outline_color = RAIDER_COLOR
        cv2.putText(
            img,
            text,
            (text_x, text_y),
            font,
            font_scale,
            outline_color,
            font_thickness + 1,
            cv2.LINE_AA,
        )

    cv2.putText(
        img,
        text,
        (text_x, text_y),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )

    if is_raider:
        tip_size = 5
        tip_x = x
        tip_y = label_y2 + 1
        if tip_y + tip_size < h:
            triangle = np.array(
                [
                    [tip_x, tip_y + tip_size],
                    [tip_x - tip_size, tip_y],
                    [tip_x + tip_size, tip_y],
                ],
                np.int32,
            )
            cv2.fillConvexPoly(img, triangle, RAIDER_COLOR)


def _draw_bbox(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    team: Optional[int],
    is_in_court: bool,
    is_raider: bool,
    track_id: int,
    has_official_id: bool,
):
    """
    Draw a bounding box around the player.

    Args:
        img: Image to draw on
        x1, y1, x2, y2: Bounding box coordinates
        team: Team number (0 or 1) or None
        is_in_court: Whether player is within court boundaries
        is_raider: Whether this player is the active raider
        track_id: YOLO track ID for debug
        has_official_id: Whether official ID has been assigned
    """
    if is_raider:
        color = RAIDER_COLOR
        thickness = 3
    elif team is not None and is_in_court:
        color = TEAM_COLORS[team]
        thickness = 2
    else:
        color = OUT_OF_COURT_COLOR
        thickness = 1

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if is_raider:
        corner_len = 15
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), RAIDER_COLOR, 4)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), RAIDER_COLOR, 4)

        cv2.line(img, (x2, y1), (x2 - corner_len, y1), RAIDER_COLOR, 4)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), RAIDER_COLOR, 4)

        cv2.line(img, (x1, y2), (x1 + corner_len, y2), RAIDER_COLOR, 4)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), RAIDER_COLOR, 4)

        cv2.line(img, (x2, y2), (x2 - corner_len, y2), RAIDER_COLOR, 4)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), RAIDER_COLOR, 4)


def _draw_scene_info(
    img: np.ndarray,
    frame_idx: int,
    num_players: int,
    num_team1: int,
    num_team2: int,
    scene_metrics: Optional[Dict] = None,
):
    """
    Draw scene information overlay in top-left corner.
    """
    h, w = img.shape[:2]

    info_lines = [
        f"Frame: {frame_idx}",
        f"Players: {num_players} (T1:{num_team1}, T2:{num_team2})",
    ]

    if scene_metrics:
        density = scene_metrics.get("density", 0)
        motion = scene_metrics.get("motion", 0)
        info_lines.append(f"Density: {density:.2f}, Motion: {motion:.2f}")

    x_base = 10
    y_base = 25

    for i, line in enumerate(info_lines):
        y = y_base + i * 20
        cv2.putText(
            img,
            line,
            (x_base, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            line,
            (x_base - 1, y - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def draw_frame(
    frame: np.ndarray,
    track_boxes: Dict[int, Tuple],
    head_positions: Dict[int, Tuple[int, int]],
    current_teams: Dict[int, int],
    official_ids: Dict[int, str],
    frame_idx: int,
    active_raider_tid: Optional[int] = None,
    court_positions: Optional[Dict[int, bool]] = None,
    sa_thresholds: Optional[Dict] = None,
    scene_metrics: Optional[Dict] = None,
) -> np.ndarray:
    """
    Main annotation function. Draws all overlays on the frame.

    Args:
        frame: Input frame (BGR)
        track_boxes: {track_id: (x1, y1, x2, y2, conf, emb)}
        head_positions: {track_id: (head_x, head_y)}
        current_teams: {track_id: team_num (0 or 1)}
        official_ids: {track_id: official_id_string (e.g., "T1#1")}
        frame_idx: Current frame number
        active_raider_tid: Track ID of the active raider
        court_positions: {track_id: bool (True if in court)}
        sa_thresholds: Scene analyzer thresholds (unused but kept for compatibility)
        scene_metrics: Scene analyzer metrics for overlay

    Returns:
        Annotated frame
    """
    img = frame.copy()
    h, w = img.shape[:2]

    if court_positions is None:
        court_positions = {}

    num_team1 = 0
    num_team2 = 0

    sorted_tids = sorted(track_boxes.keys())

    for tid in sorted_tids:
        if tid == active_raider_tid:
            continue

        box_data = track_boxes[tid]
        x1, y1, x2, y2 = [int(v) for v in box_data[:4]]

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        team = current_teams.get(tid)
        if team == 0:
            num_team1 += 1
        elif team == 1:
            num_team2 += 1

        is_in_court = court_positions.get(tid, True)
        has_official_id = tid in official_ids

        _draw_bbox(img, x1, y1, x2, y2, team, is_in_court, False, tid, has_official_id)

        head_x, head_y = head_positions.get(tid, ((x1 + x2) // 2, y1))
        head_x = max(0, min(w - 1, head_x))
        head_y = max(0, min(h - 1, head_y))

        _draw_head_label(
            img,
            head_x,
            head_y,
            official_ids.get(tid),
            team,
            is_in_court,
            False,
            has_official_id,
        )

    if active_raider_tid is not None and active_raider_tid in track_boxes:
        box_data = track_boxes[active_raider_tid]
        x1, y1, x2, y2 = [int(v) for v in box_data[:4]]

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        team = current_teams.get(active_raider_tid)
        if team == 0:
            num_team1 += 1
        elif team == 1:
            num_team2 += 1

        is_in_court = court_positions.get(active_raider_tid, True)
        has_official_id = active_raider_tid in official_ids

        _draw_bbox(
            img,
            x1,
            y1,
            x2,
            y2,
            team,
            is_in_court,
            True,
            active_raider_tid,
            has_official_id,
        )

        head_x, head_y = head_positions.get(active_raider_tid, ((x1 + x2) // 2, y1))
        head_x = max(0, min(w - 1, head_x))
        head_y = max(0, min(h - 1, head_y))

        _draw_head_label(
            img,
            head_x,
            head_y,
            official_ids.get(active_raider_tid),
            team,
            is_in_court,
            True,
            has_official_id,
        )

    _draw_scene_info(
        img, frame_idx, len(track_boxes), num_team1, num_team2, scene_metrics
    )

    _state["frame_idx"] = frame_idx

    return img
