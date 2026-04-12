import sys
import os

# Ensure backend modules are found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline
import database
import time

def main():
    database.init_db()
    
    input_video = "input/23066f49-e4fe-4b21-b2df-db1493cdf6af_Kabaddi_video.mp4"
    if not os.path.exists(input_video):
        print(f"Error: {input_video} not found.")
        return
        
    output_video = "output/test_output.mp4"
    job_id = "test_job_123"
    
    print(f"Starting test run on {input_video}...")
    start = time.time()
    
    try:
        def check_progress(f, t):
            if f % 10 == 0:
                print(f"Frame {f}/{t}")
            if f > 75:
                print("Finished 75 frames! Breaking to check outputs.")
                return False
            return True
                
        pipeline.run_pipeline(
            input_video, 
            output_video, 
            job_id,
            progress_callback=check_progress
        )
        print(f"Success! Finished in {time.time() - start:.2f} seconds.")
        print(f"Output saved to {output_video}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
