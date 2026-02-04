from fastapi import APIRouter, HTTPException
from app.supabase_client import supabase

router = APIRouter()

@router.get("/runs/{run_id}")
def get_run(run_id: str):
    run_res = (
        supabase.table("runs")
        .select("*")
        .eq("id", run_id)
        .single()
        .execute()
    )
    if not run_res.data:
        raise HTTPException(status_code=404, detail="Run not found")

    preds_res = (
        supabase.table("predictions")
        .select("metric,value,breakdown,created_at")
        .eq("run_id", run_id)
        .execute()
    )

    arts_res = (
        supabase.table("artifacts")
        .select("kind,bucket,path,size_bytes,sha256,created_at")
        .eq("run_id", run_id)
        .execute()
    )

    SIGNED_URL_TTL = 3600

    signed_artifacts = []
    for a in (arts_res.data or []):
        signed = supabase.storage.from_(a["bucket"]).create_signed_url(
            a["path"],
            SIGNED_URL_TTL
        )
        signed_artifacts.append({
            **a,
            "signed_url": signed.get("signedURL") if signed else None
        })

    score = None
    for p in (preds_res.data or []):
        if p["metric"] in ("cr_score", "score"):
            score = p["value"]
            break

    return {
        "run": run_res.data,
        "score": score,
        "predictions": preds_res.data or [],
        "artifacts": signed_artifacts,
    }