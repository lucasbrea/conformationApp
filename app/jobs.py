from supabase import create_client
from app.pipeline.run import run_pipeline_one
import os
from pathlib import Path

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"]
)

BUCKET = "Conformation_Artifacts"  # MUST match exactly in Supabase

def run_job(input_image, job_dir, dlc_config, coeffs_json, horse_id):

    run = supabase.table("runs").insert({
        "horse_id": horse_id,
        "status": "running",
        "model_name": "dlc_conformation",
        "model_version": "v1",
        "started_at": "now()"
    }).execute()

    run_id = run.data[0]["id"]

    try:
        result = run_pipeline_one(
            input_image=Path(input_image),
            job_dir=Path(job_dir),
            dlc_config=Path(dlc_config),
            coeffs_json=Path(coeffs_json),
        )

        # upload labeled video
        video_path = Path(result["labeled_video"])
        storage_path = f"results/{run_id}/{video_path.name}"

        supabase.storage.from_(BUCKET).upload(
            storage_path,
            video_path.read_bytes(),
            {"content-type": "video/mp4"}
        )

        supabase.table("artifacts").insert({
            "run_id": run_id,
            "kind": "labeled_video",
            "bucket": BUCKET,
            "path": storage_path,
            "size_bytes": video_path.stat().st_size
        }).execute()

        supabase.table("predictions").insert({
            "run_id": run_id,
            "metric": "cr_score",
            "value": float(result["score"]),
            "breakdown": result.get("warnings")  # or {} / None; replace later with real json
        }).execute()

        supabase.table("runs").update({
            "status": "succeeded",
            "finished_at": "now()"
        }).eq("id", run_id).execute()

        return {"run_id": run_id, "score": result["score"]}

    except Exception as e:
        supabase.table("runs").update({
            "status": "failed",
            "error_message": str(e),
            "finished_at": "now()"
        }).eq("id", run_id).execute()
        raise