import pipeline
import os
import time

def benchmark():
    input_vid = r"c:\Users\raksh\Downloads\newww\newww\newww\backend\input\23066f49-e4fe-4b21-b2df-db1493cdf6af_Kabaddi_video.mp4"
    output_vid = r"c:\Users\raksh\Downloads\newww\newww\newww\backend\output\benchmark_output.mp4"
    job_id = "bench_001"
    
    if not os.path.exists(input_vid):
        print(f"Error: Input video not found at {input_vid}")
        return

    print(f"--- STARTING SPEED BENCHMARK ---")
    start = time.time()
    
    # Process only 300 frames for a quick benchmark
    # We can't easily tell the pipeline to stop at 300 frames without modifying it or using a callback
    # I'll use a progress callback that returns False after 300 frames
    
    def progress_limit(current, total):
        if current >= 300:
            return False
        if current % 10 == 0:
            print(f"  Processed {current}/300 frames...")
        return True

    results = pipeline.run_pipeline(input_vid, output_vid, job_id, progress_callback=progress_limit)
    
    end = time.time()
    elapsed = end - start
    fps = results['frames_processed'] / elapsed
    
    print(f"\n--- BENCHMARK RESULTS ---")
    print(f"Frames Processed: {results['frames_processed']}")
    print(f"Total Time:      {elapsed:.2f}s")
    print(f"Average FPS:     {fps:.2f}")
    print(f"Output saved to: {output_vid}")

if __name__ == "__main__":
    benchmark()
