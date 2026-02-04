from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import uuid
from fastapi.staticfiles import StaticFiles
from app.pipeline.run import run_pipeline_one
from app.queue import q
from app.jobs import run_job
from rq import Queue
from redis import Redis
from supabase import create_client
import os

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"]
)  

horse = supabase.table("horses").insert({}).execute()
horse_id = horse.data[0]["id"]

app = FastAPI()
app.mount("/data", StaticFiles(directory="/data"), name="data")
MODELS_DIR = Path("/models")
DATA_DIR = Path("/data")


redis_conn = Redis.from_url("redis://redis:6379/0")  # or from env
q = Queue("conformation", connection=redis_conn)

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:12]
    job_dir = DATA_DIR / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "input.jpg"
    input_path.write_bytes(await file.read())

    job = q.enqueue(
        run_job,
        str(input_path),
        str(job_dir),
        str(MODELS_DIR / "dlc_project" / "config_inference.yaml"),
        str(MODELS_DIR / "model_coeffs.json"),
        job_id=job_id,
        horse_id=horse_id
    )

    return {"job_id": job.id, "status": "queued"}

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job_dir = DATA_DIR / "jobs" / job_id
    if not job_dir.exists():
        return {"job_id": job_id, "status": "not_found"}

    status_file = job_dir / "status.txt"
    status = status_file.read_text().strip() if status_file.exists() else "unknown"

    result_file = job_dir / "result.json"
    if result_file.exists():
        return {"job_id": job_id, "status": status, "result": json.loads(result_file.read_text())}

    return {"job_id": job_id, "status": status}