from pathlib import Path
from app.pipeline.run import run_pipeline_one

def run_job(input_image: str, job_dir: str, dlc_config: str, coeffs_json: str):
    result = run_pipeline_one(
        input_image=Path(input_image),
        job_dir=Path(job_dir),
        dlc_config=Path(dlc_config),
        coeffs_json=Path(coeffs_json),
    )

    # Return only what the outside world should see
    return {
        "job_id": result["job_id"],
        "score": result["score"],          # 0–100 linear index
        "labeled_video": result.get("labeled_video"),
    }