from fastapi import APIRouter, HTTPException
from app.supabase_client import supabase  # you create this module

router = APIRouter()

@router.get("/runs/{run_id}")
def get_run(run_id: str):
    # runs
    run_res = (
        supabase.table("runs")
        .select("*")
        .eq("id", run_id)
        .single()
        .execute()
    )
    if not run_res.data:
        raise HTTPException(status_code=404, detail="Run not found")

    # predictions
    preds_res = (
        supabase.table("predictions")
        .select("metric,value,breakdown,created_at")
        .eq("run_id", run_id)
        .execute()
    )

    # artifacts
    arts_res = (
        supabase.table("artifacts")
        .select("kind,bucket,path,size_bytes,sha256,created_at")
        .eq("run_id", run_id)
        .execute()
    )

    # convenience: pick one main score if present
    score = None
    for p in (preds_res.data or []):
        if p["metric"] in ("cr_score", "score"):
            score = p["value"]
            break

    return {
        "run": run_res.data,
        "score": score,
        "predictions": preds_res.data or [],
        "artifacts": arts_res.data or [],
    }