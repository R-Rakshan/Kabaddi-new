import sys
import os
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline
import database


def main():
    database.init_db()

    # Use converted video for better compatibility
    input_video = "input/video_converted.mp4"
    if not os.path.exists(input_video):
        input_video = "input/WhatsApp Video 2026-04-12 at 10.48.03 PM.mp4"
    if not os.path.exists(input_video):
        print(f"Error: {input_video} not found.")
        return

    output_video = "output/test_output.mp4"
    os.makedirs("output", exist_ok=True)
    job_id = "test_kabaddi_001"

    print(f"Starting test run on {input_video}...")
    print("=" * 60)
    start = time.time()
    last_pct = -1

    try:

        def check_progress(f, t):
            nonlocal last_pct
            pct = int(f / max(t, 1) * 100)
            if pct != last_pct:
                bar_len = 40
                filled = int(bar_len * pct / 100)
                bar = "█" * filled + "░" * (bar_len - filled)
                elapsed = time.time() - start
                fps = f / elapsed if elapsed > 0 else 0
                eta = (t - f) / fps if fps > 0 else 0
                sys.stdout.write(
                    f"\r[{bar}] {pct:3d}% | Frame {f:5d}/{t} | {fps:.1f} fps | ETA: {eta:.0f}s"
                )
                sys.stdout.flush()
                last_pct = pct
            return True

        result = pipeline.run_pipeline(
            input_video, output_video, job_id, progress_callback=check_progress
        )

        elapsed = time.time() - start
        print(f"\n\n{'=' * 60}")
        print(
            f"SUCCESS! Processed {result['frames_processed']} frames in {elapsed:.1f}s"
        )
        print(f"Players identified: {result['players_assigned']}")
        print(f"Output saved to: {output_video}")

        if os.path.exists(output_video):
            size_mb = os.path.getsize(output_video) / (1024 * 1024)
            print(f"Output size: {size_mb:.1f} MB")

    except Exception as e:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
