from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import uuid

from app.pipeline.run import run_pipeline_one

app = FastAPI()

MODELS_DIR = Path("/models")
DATA_DIR = Path("/data")

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:12]
    job_dir = DATA_DIR / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "input.jpg"
    input_path.write_bytes(await file.read())

    result = run_pipeline_one(
        input_image=input_path,
        job_dir=job_dir,
        dlc_config=MODELS_DIR / "dlc_project" / "config_inference.yaml",
        coeffs_json=MODELS_DIR / "model_coeffs.json",
    )

    return result