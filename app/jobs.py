from supabase import create_client
from app.pipeline.run import run_pipeline_one
import os
from pathlib import Path
import json

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"]
)

# MUST match Supabase Storage bucket name EXACTLY (case-sensitive)
BUCKET = "Conformation_Artifacts"  # change if your bucket is named differently

def run_job(input_image, job_dir, dlc_config, coeffs_json, horse_id, run_id):
    """
    RQ worker entrypoint.
    Expects run_id to already exist (created in API as status='queued').
    """

    # Mark run as running (and set started_at)
    supabase.table("runs").update({
        "status": "running",
        "started_at": "now()"
    }).eq("id", run_id).execute()

    try:
        # Run pipeline
        result = run_pipeline_one(
            input_image=Path(input_image),
            job_dir=Path(job_dir),
            dlc_config=Path(dlc_config),
            coeffs_json=Path(coeffs_json),
        )

        # Upload labeled video artifact
        video_path = Path(result["labeled_video"])
        storage_path = f"results/{run_id}/{video_path.name}"

        job_dir_p = Path(job_dir)

        features = None
        likelihood = None

        # if your pipeline writes these files (it looked like it did before)
        job_dir_p = Path(job_dir)
        matches = list(job_dir_p.rglob("features.json"))
        features = json.loads(matches[0].read_text()) if matches else None  

        # if likelihood is in result, just grab it
        likelihood = result.get("Likelihood") or result.get("likelihood")

        supabase.storage.from_(BUCKET).upload(
            storage_path,
            video_path.read_bytes(),
            {"content-type": "video/mp4"}
        )

        # Record artifact metadata
        supabase.table("artifacts").insert({
            "run_id": run_id,
            "kind": "labeled_video",
            "bucket": BUCKET,
            "path": storage_path,
            "size_bytes": video_path.stat().st_size
        }).execute()

        # Record prediction
        supabase.table("predictions").insert({
            "run_id": run_id,
            "metric": "cr_score",
            "value": float(result["score"]),
            "breakdown": {
                "warnings": result.get("warnings"),
                "likelihood": likelihood,
                "features": features,
            }
        }).execute()

        # Mark run succeeded
        supabase.table("runs").update({
            "status": "succeeded",
            "finished_at": "now()"
        }).eq("id", run_id).execute()

        return {"run_id": run_id, "score": result["score"]}

    except Exception as e:
        # Mark run failed
        supabase.table("runs").update({
            "status": "failed",
            "error_message": str(e),
            "finished_at": "now()"
        }).eq("id", run_id).execute()
        raise