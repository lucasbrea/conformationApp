from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import uuid

from app.queue import q
from app.jobs import run_job
from app.supabase_client import supabase

router = APIRouter()

MODELS_DIR = Path("/models")
DATA_DIR = Path("/data")

@router.post("/infer")
async def infer(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:12]
    job_dir = DATA_DIR / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "input.jpg"
    input_path.write_bytes(await file.read())

    # create a horse (placeholder, must be unique if you have constraints)
    horse = supabase.table("horses").insert({
        "sale": "TEMP",
        "hip": int(uuid.uuid4().int % 1_000_000)
    }).execute()
    horse_id = horse.data[0]["id"]

    # create a run (queued)
    run = supabase.table("runs").insert({
        "horse_id": horse_id,
        "status": "queued",
        "model_name": "dlc_conformation",
        "model_version": "v1"
    }).execute()
    run_id = run.data[0]["id"]

    q.enqueue(
        run_job,
        str(input_path),
        str(job_dir),
        str(MODELS_DIR / "dlc_project" / "config_inference.yaml"),
        str(MODELS_DIR / "model_coeffs.json"),
        horse_id=horse_id,
        run_id=run_id
    )

    return {"run_id": run_id, "status": "queued"}