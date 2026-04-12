import os
import cv2
import time
import json
import psutil
import pipeline
import id_manager
import court_detector
import numpy as np

def run_stress_test(video_path, num_loops=50):
    print(f"--- STARTING LONG DURATION STRESS TEST (Simulation: {num_loops} loops) ---")
    
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    process = psutil.Process(os.getpid())
    start_time = time.time()
    total_frames = 0
    tracker_state = {}

    for loop in range(num_loops):
        print(f"\n[Loop {loop+1}/{num_loops}] Processing segment...")
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        
        while cap.isOpened() and frame_idx < 100: # Process 100 frames per loop
            ret, frame = cap.read()
            if not ret: break
            
            # Run the actual pipeline
            track_boxes, head_positions, current_teams, official_ids = pipeline.process_single_frame(frame, tracker_state)
            
            total_frames += 1
            frame_idx += 1
            
            if total_frames % 200 == 0:
                mem_mb = process.memory_info().rss / (1024 * 1024)
                fps = total_frames / (time.time() - start_time)
                print(f"  > Status: {total_frames} frames | Mem: {mem_mb:.1f} MB | FPS: {fps:.1f} | Active IDs: {len(id_manager._locked)}")

        cap.release()
        
    end_time = time.time()
    final_mem = process.memory_info().rss / (1024 * 1024)
    avg_fps = total_frames / (end_time - start_time)
    
    report = {
        "total_frames": total_frames,
        "duration_sec": end_time - start_time,
        "avg_fps": avg_fps,
        "peak_mem_mb": final_mem,
        "final_id_count": len(id_manager._locked),
        "status": "PASS" if final_mem < 4000 else "FAIL (Memory Leak)"
    }
    
    with open("stress_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print("\n--- STRESS TEST COMPLETE ---")
    print(json.dumps(report, indent=4))

if __name__ == "__main__":
    vid = r"c:\Users\raksh\Downloads\newww\newww\newww\backend\input\23066f49-e4fe-4b21-b2df-db1493cdf6af_Kabaddi_video.mp4"
    run_stress_test(vid, num_loops=20)
