"""
pipeline.py  — v2
==================
Full orchestration pipeline: reads video, runs all CV modules,
writes annotated video, logs progress to SQLite.
"""

import cv2
import os
import time
import threading
import queue
import subprocess
from typing import Optional, Callable

import detect_track
import team_classifier
import pose_estimator
import id_manager
import annotator
import database
import court_detector
import numpy as np


def find_active_raider(track_boxes: dict, current_teams: dict) -> int:
    team_x_sums = {0: 0.0, 1: 0.0}
    team_counts = {0: 0, 1: 0}

    player_court_x = {}
    for tid, box in track_boxes.items():
        feet_x = (box[0] + box[2]) / 2.0
        feet_y = float(box[3])
        cx_m, cy_m = court_detector.image_to_court(feet_x, feet_y)
        player_court_x[tid] = cx_m

        team = current_teams.get(tid)
        if team is not None:
            team_x_sums[team] += cx_m
            team_counts[team] += 1

    if team_counts[0] == 0 or team_counts[1] == 0:
        return None

    team0_center = team_x_sums[0] / team_counts[0]
    team1_center = team_x_sums[1] / team_counts[1]

    team0_is_left = team0_center < team1_center
    midline = (team0_center + team1_center) / 2.0

    raiders = []
    for tid, cx_m in player_court_x.items():
        team = current_teams.get(tid)
        if team is None:
            continue

        if team == 0 and team0_is_left and cx_m > midline:
            raiders.append(tid)
        elif team == 0 and not team0_is_left and cx_m < midline:
            raiders.append(tid)
        elif team == 1 and not team0_is_left and cx_m > midline:
            raiders.append(tid)
        elif team == 1 and team0_is_left and cx_m < midline:
            raiders.append(tid)

    if len(raiders) == 1:
        return raiders[0]

    if len(raiders) > 1:
        return max(raiders, key=lambda t: abs(player_court_x[t] - midline))

    return None


def process_single_frame(
    frame: np.ndarray, tracker_state: dict, frame_num: int = 0
) -> tuple:
    """
    Processes a single frame through the entire CV stack.
    Returns:
        track_boxes, head_positions, current_teams, official_ids, court_positions
    """
    # 1. Detection + ByteTrack + Pose + Head
    track_boxes, head_positions, foot_positions, court_positions = (
        detect_track.detect_and_track(frame, tracker_state, frame_idx=frame_num)
    )

    # 2. Team Classification
    team_classifier.update_observations(frame, track_boxes, frame_idx=frame_num)
    current_teams = team_classifier.classify_teams()

    # 3. Identity Management (Re-ID + Stable Locks)
    id_manager.record_frame(track_boxes, current_teams)
    official_ids = id_manager.try_assign_ids()

    # 4. Team Locking for assigned IDs
    for tid, oid in official_ids.items():
        team_num = 0 if oid.startswith("T1") else 1
        team_classifier.lock_team(tid, team_num)
        current_teams[tid] = team_num

    return track_boxes, head_positions, current_teams, official_ids, court_positions


def run_pipeline(
    input_path: str,
    output_path: str,
    job_id: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    # ── Reset all module states ──────────────────────────────────────────
    team_classifier.reset()
    pose_estimator.reset()
    id_manager.reset()
    annotator.reset()
    court_detector.reset()

    # Get video properties using ffprobe
    import subprocess

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames",
            "-of",
            "default=noprint_wrappers=1",
            input_path,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    width, height, fps_str, total_frames = 640, 480, "25/1", 0
    for line in probe.stdout.split("\n"):
        if line.startswith("width="):
            width = int(line.split("=")[1])
        elif line.startswith("height="):
            height = int(line.split("=")[1])
        elif line.startswith("r_frame_rate="):
            fps_str = line.split("=")[1]
        elif line.startswith("nb_frames="):
            total_frames = int(line.split("=")[1])

    fps = eval(fps_str) if fps_str else 25.0
    if total_frames == 0:
        # Estimate from duration if nb_frames not available
        total_frames = 500  # Will be updated as we process

    # Extract frames using ffmpeg to temp directory
    frames_dir = os.path.join(os.path.dirname(output_path), f"frames_{job_id}")
    os.makedirs(frames_dir, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-i",
            input_path,
            "-vf",
            "scale=640:480",
            "-q:v",
            "2",
            "-fps_mode",
            "vfr",
            f"{frames_dir}/frame_%05d.jpg",
        ],
        capture_output=True,
        timeout=120,
    )

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    total_frames = len(frame_files)

    # Radar dimensions
    radar_w = int(court_detector.RADAR_W)
    comb_w = 640 + radar_w
    comb_h = max(480, court_detector.RADAR_H)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── FFmpeg Writer Setup with Fallback ─────────────────────────────────
    ffmpeg_proc = None
    writer = None
    try:
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{comb_w}x{comb_h}",
            "-pix_fmt",
            "bgr24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except Exception:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (comb_w, comb_h))

    # ── Parallel Queues ──────────────────────────────────────────────────
    input_queue = queue.Queue(maxsize=32)
    output_queue = queue.Queue(maxsize=32)

    tracker_state = {}
    stop_event = threading.Event()

    def reader_loop():
        for i, fname in enumerate(sorted(frame_files)):
            if stop_event.is_set():
                break
            frame_idx = i + 1

            # Frame Skipping - process every 5th frame for CPU speed
            if frame_idx % 5 != 0:
                continue

            frame = cv2.imread(os.path.join(frames_dir, fname))
            if frame is not None:
                input_queue.put((frame_idx, frame))

        input_queue.put((None, None))

    def processor_loop():
        while not stop_event.is_set():
            frame_idx, frame = input_queue.get()
            if frame_idx is None:
                break

            # Downscale to 480p for faster CPU processing
            h, w = frame.shape[:2]
            if h > 480:
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

            # CV Processing
            (
                track_boxes,
                head_positions,
                current_teams,
                official_ids,
                court_positions,
            ) = process_single_frame(frame, tracker_state, frame_idx)

            # Annotation logic
            active_raider_tid = find_active_raider(track_boxes, current_teams)
            sa = tracker_state.get("scene_analyzer")
            sa_th = sa.get_thresholds() if sa else None
            sa_metrics = (
                {
                    "density": sa.density_ema,
                    "motion": sa.motion_ema,
                    "occlusion": sa.occlusion_ema,
                }
                if sa
                else None
            )

            annotated = annotator.draw_frame(
                frame,
                track_boxes,
                head_positions,
                current_teams,
                official_ids,
                frame_idx,
                active_raider_tid,
                court_positions=court_positions,
                sa_thresholds=sa_th,
                scene_metrics=sa_metrics,
            )

            # Annotation logic
            active_raider_tid = find_active_raider(track_boxes, current_teams)
            sa = tracker_state.get("scene_analyzer")
            sa_th = sa.get_thresholds() if sa else None
            sa_metrics = (
                {
                    "density": sa.density_ema,
                    "motion": sa.motion_ema,
                    "occlusion": sa.occlusion_ema,
                }
                if sa
                else None
            )

            annotated = annotator.draw_frame(
                frame,
                track_boxes,
                head_positions,
                current_teams,
                official_ids,
                frame_idx,
                active_raider_tid,
                sa_thresholds=sa_th,
                scene_metrics=sa_metrics,
            )

            radar_img = court_detector.update_and_draw(
                width,
                height,
                track_boxes,
                current_teams,
                official_ids,
                active_raider_tid,
            )

            # Combined Layout logic
            target_h = max(annotated.shape[0], radar_img.shape[0])
            if annotated.shape[0] < target_h:
                pad = np.zeros(
                    (target_h - annotated.shape[0], annotated.shape[1], 3),
                    dtype=np.uint8,
                )
                annotated = np.vstack((annotated, pad))
            if radar_img.shape[0] < target_h:
                pad = np.zeros(
                    (target_h - radar_img.shape[0], radar_img.shape[1], 3),
                    dtype=np.uint8,
                )
                radar_img = np.vstack((radar_img, pad))

            combined = np.hstack((annotated, radar_img))
            output_queue.put((frame_idx, combined))
        output_queue.put((None, None))

    # ── Start Threads ────────────────────────────────────────────────────
    reader_thread = threading.Thread(target=reader_loop)
    processor_thread = threading.Thread(target=processor_loop)
    reader_thread.start()
    processor_thread.start()

    start_time = time.time()
    database.update_job(job_id, status="processing", total_frames=total_frames)

    frame_count = 0
    try:
        while True:
            frame_idx, combined_frame = output_queue.get()
            if frame_idx is None:
                break

            if ffmpeg_proc and ffmpeg_proc.poll() is None:
                ffmpeg_proc.stdin.write(combined_frame.tobytes())
            elif writer:
                writer.write(combined_frame)

            frame_count += 1

            if progress_callback:
                if progress_callback(frame_count, total) is False:
                    stop_event.set()
                    break

            if frame_count % 30 == 0:
                pct = int(frame_count / total * 100) if total > 0 else 0
                database.update_job(job_id, progress=pct, processed_frames=frame_count)
    finally:
        stop_event.set()
        reader_thread.join()
        processor_thread.join()
        if ffmpeg_proc:
            if ffmpeg_proc.stdin:
                ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
        if writer:
            writer.release()

    # Cleanup extracted frames
    try:
        import shutil

        shutil.rmtree(frames_dir)
    except:
        pass

    elapsed = round(time.time() - start_time, 2)
    locked_map = id_manager.try_assign_ids()

    for tid, oid in locked_map.items():
        team_num = 1 if oid.startswith("T1") else 2
        database.save_player_record(job_id, tid, oid, team_num)

    out_dir = os.path.dirname(output_path)
    png_path = os.path.join(out_dir, f"{job_id}_trajectory_graph.png")
    json_path = os.path.join(out_dir, f"{job_id}_player_trajectories.json")
    court_detector.export_data(png_path, json_path, team_classifier.get_locked_teams())

    database.update_job(
        job_id,
        status="done",
        progress=100,
        processed_frames=frame_count,
        elapsed_seconds=elapsed,
        output_path=output_path,
    )

    return {
        "job_id": job_id,
        "frames_processed": frame_count,
        "elapsed_seconds": elapsed,
        "players_assigned": len(locked_map),
    }
