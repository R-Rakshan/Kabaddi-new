print("[INFO] Starting Kabaddi CV System...")
print("[INFO] Initializing Core dependencies...")
import os
import shutil
import threading

print("[INFO] Loading FastAPI and Network modules...")
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

print("[INFO] Connecting to Database...")
import database

print("[INFO] Initializing AI Models (this may take 1-2 minutes)...")
import pipeline

print("[INFO] All modules loaded. Setting up App...")
# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="Kabaddi CV API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

database.init_db()


# ── Background task ───────────────────────────────────────────────────────────

def _run_pipeline_thread(job_id: str, input_path: str, output_path: str):
    """Runs the full CV pipeline in a background thread."""
    try:
        database.update_job(job_id, status="processing")
        result = pipeline.run_pipeline(input_path, output_path, job_id)
        print(f"[INFO] Job {job_id} finished: {result}")
    except Exception as exc:
        import traceback
        print(f"[ERROR] Job {job_id} failed:\n{traceback.format_exc()}")
        database.update_job(job_id, status="error")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"message": "Kabaddi CV API is running. Use /api/health to check status."}

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/jobs/upload", status_code=202)
async def upload_video(file: UploadFile = File(...)):
    """Upload a video and immediately start processing in a background thread."""
    allowed = (".mp4", ".avi", ".mov", ".mkv")
    if not file.filename.lower().endswith(allowed):
        raise HTTPException(400, f"Only video files accepted: {', '.join(allowed)}")

    job_id      = database.create_job(file.filename)
    input_path  = os.path.join(INPUT_DIR,  f"{job_id}_{file.filename}")
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}_output.mp4")

    # Save uploaded file to disk
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Update job with saved paths
    database.update_job(job_id, input_path=input_path, output_path=output_path, status="queued")

    # Kick off pipeline in a daemon thread
    t = threading.Thread(
        target=_run_pipeline_thread,
        args=(job_id, input_path, output_path),
        daemon=True,
    )
    t.start()

    return {"job_id": job_id, "status": "queued", "filename": file.filename}


@app.get("/api/jobs")
def list_jobs():
    return database.list_jobs()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/jobs/{job_id}/players")
def get_players(job_id: str):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {"job_id": job_id, "players": database.get_players(job_id)}


@app.get("/api/jobs/{job_id}/video")
def download_video(job_id: str):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("status") != "done":
        raise HTTPException(400, "Video is not ready yet")
    out_path = job.get("output_path")
    if not out_path or not os.path.exists(out_path):
        raise HTTPException(404, "Output file not found on disk")
    return FileResponse(out_path, media_type="video/mp4",
                        filename=f"kabaddi_{job_id}.mp4")


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    for field in ("input_path", "output_path"):
        fp = job.get(field)
        if fp and os.path.exists(fp):
            try:
                os.remove(fp)
            except OSError:
                pass
    database.delete_job(job_id)
    from fastapi import Response
    return Response(status_code=204)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("[DEBUG] Loading modules and starting Uvicorn on 8001...")
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=False)
