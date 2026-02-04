# jobs.py

from supabase import create_client
from app.pipeline.run import run_pipeline_one
import os

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"]
)

def run_job(input_image, job_dir, dlc_config, coeffs_json, horse_id):

    # 1) insert run
    run = supabase.table("runs").insert({
        "horse_id": horse_id,
        "status": "running",
        "model_name": "dlc_conformation",
        "model_version": "v1"
    }).execute()

    run_id = run.data[0]["id"]

    try:
        # 2) run your EXISTING pipeline
        result = run_pipeline_one(
            input_image=input_image,
            job_dir=job_dir,
            dlc_config=dlc_config,
            coeffs_json=coeffs_json,
        )

        # 3) upload artifacts + insert predictions
        # 4) mark run succeeded

    except Exception as e:
        # 5) mark run failed
        raise