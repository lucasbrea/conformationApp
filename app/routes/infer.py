from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import uuid

from app.queue import q
from app.jobs import run_job
from app.supabase_client import supabase

router = APIRouter()

MODELS_DIR = Path("/models")
DATA_DIR = Path("/data")
horse = supabase.table("horses").insert({
    "sale": "TEMP",
    "hip": int(uuid.uuid4().int % 1_000_000)
}).execute()
horse_id = horse.data[0]["id"]

@router.post("/infer")
async def infer(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:12]
    job_dir = DATA_DIR / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "input.jpg"
    input_path.write_bytes(await file.read())

    # create a horse (placeholder for now)
    horse = supabase.table("horses").insert({"sale": "UNKNOWN", "hip": -1}).execute()
    horse_id = horse.data[0]["id"]

    job = q.enqueue(
        run_job,
        str(input_path),
        str(job_dir),
        str(MODELS_DIR / "dlc_project" / "config_inference.yaml"),
        str(MODELS_DIR / "model_coeffs.json"),
        horse_id=horse_id
    )

    return {"job_id": job.id, "status": "queued", "horse_id": horse_id}