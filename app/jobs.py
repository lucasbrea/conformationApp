from pathlib import Path
from app.pipeline.run import run_pipeline_one

MODELS_DIR = Path("/models")

def process_job(job_id: str, input_path: str, job_dir: str):
    job_dir = Path(job_dir)
    input_path = Path(input_path)

    result = run_pipeline_one(
        input_image=input_path,
        job_dir=job_dir,
        dlc_config=MODELS_DIR / "dlc_project" / "config_inference.yaml",
        coeffs_json=MODELS_DIR / "model_coeffs.json",
    )

    # Write a status/result file the API can read
    (job_dir / "result.json").write_text(__import__("json").dumps(result, indent=2))
    (job_dir / "status.txt").write_text("done")

    return result