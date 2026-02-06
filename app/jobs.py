from supabase import create_client
from app.pipeline.run import run_pipeline_one
import os
from pathlib import Path
import json

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"]
)

BUCKET = "Conformation_Artifacts"
BASELINE_PATH = Path("/models/xb_baseline.json")  # must exist in container

def _load_features(job_dir: str) -> dict | None:
    job_dir_p = Path(job_dir)
    matches = list(job_dir_p.rglob("features.json"))
    if not matches:
        return None
    try:
        return json.loads(matches[0].read_text())
    except Exception:
        return None

def _build_feature_contribs(features: dict | None, coeffs_json_path: str) -> dict | None:
    """
    Returns dict:
      feature -> {x, beta, contrib, mean_contrib, delta}
    """
    if not features:
        return None

    # model coeffs
    try:
        model = json.loads(Path(coeffs_json_path).read_text())
        coef_dict = model.get("coeffs", {})  # feature -> beta
    except Exception:
        return None

    # baseline mean contributions
    try:
        baseline = json.loads(BASELINE_PATH.read_text())
        mean_contrib = baseline.get("mean_contrib", baseline.get("mean", {}))  # support either key
    except Exception:
        mean_contrib = {}

    out = {}
    for f, x in features.items():
        if f not in coef_dict:
            continue
        try:
            x_f = float(x)
            b_f = float(coef_dict[f])
        except Exception:
            continue

        contrib = x_f * b_f
        m = float(mean_contrib.get(f, 0.0))
        out[f] = {
            "x": x_f,
            "beta": b_f,
            "contrib": contrib,
            "mean_contrib": m,
            "delta": contrib - m
        }

    return out

def run_job(input_image, job_dir, dlc_config, coeffs_json, horse_id, run_id):
    """
    RQ worker entrypoint.
    Expects run_id to already exist (created in API as status='queued').
    """

    # Mark run as running
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

        # Likelihood gate
        likelihood = result.get("Likelihood") or result.get("likelihood") or {}
        lk_mean = likelihood.get("lk_mean")

        if lk_mean is None:
            supabase.table("runs").update({
                "status": "rejected",
                "quality_score": None,
                "quality_reason": "missing_likelihood",
                "finished_at": "now()"
            }).eq("id", run_id).execute()
            return {"run_id": run_id, "status": "rejected", "reason": "missing_likelihood"}

        if float(lk_mean) < 0.50:
            supabase.table("runs").update({
                "status": "rejected",
                "quality_score": float(lk_mean),
                "quality_reason": "low_likelihood",
                "finished_at": "now()"
            }).eq("id", run_id).execute()
            return {"run_id": run_id, "status": "rejected", "lk_mean": float(lk_mean)}

        # Load features from disk (pipeline writes it under artifacts_dir)
        features = _load_features(job_dir)

        # Build simple per-feature contribution deltas
        feature_contribs = _build_feature_contribs(features, coeffs_json)

        # Upload labeled video
        video_path = Path(result["labeled_video"])
        storage_path = f"results/{run_id}/{video_path.name}"

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

        # Record prediction + extras
        supabase.table("predictions").insert({
            "run_id": run_id,
            "metric": "cr_score",
            "value": float(result["score"]),
            "breakdown": {
                "warnings": result.get("warnings"),
                "likelihood": likelihood,
                "features": features,  # optional; can remove if too big
                "feature_contribs": feature_contribs
            }
        }).execute()

        # Mark run succeeded
        supabase.table("runs").update({
            "status": "succeeded",
            "finished_at": "now()"
        }).eq("id", run_id).execute()

        return {"run_id": run_id, "score": float(result["score"])}

    except Exception as e:
        supabase.table("runs").update({
            "status": "failed",
            "error_message": str(e),
            "finished_at": "now()"
        }).eq("id", run_id).execute()
        raise