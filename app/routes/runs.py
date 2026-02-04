from fastapi import APIRouter
from app.supabase_client import supabase

router = APIRouter()

SIGNED_URL_TTL = 3600  # seconds

@router.get("/runs")
def list_runs(limit: int = 50):
    # 1) fetch latest runs
    runs_res = (
        supabase.table("runs")
        .select("id,status,created_at,horse_id,model_name,model_version")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    runs = runs_res.data or []
    if not runs:
        return {"runs": []}

    run_ids = [r["id"] for r in runs]

    # 2) fetch scores (predictions)
    preds_res = (
        supabase.table("predictions")
        .select("run_id,metric,value")
        .in_("run_id", run_ids)
        .execute()
    )
    preds = preds_res.data or []

    # pick one score per run (cr_score preferred)
    score_by_run = {}
    for p in preds:
        rid = p["run_id"]
        m = p["metric"]
        if m == "cr_score":
            score_by_run[rid] = p["value"]
        elif m == "score" and rid not in score_by_run:
            score_by_run[rid] = p["value"]

    # 3) fetch artifacts (for preview)
    arts_res = (
        supabase.table("artifacts")
        .select("run_id,kind,bucket,path,created_at")
        .in_("run_id", run_ids)
        .order("created_at", desc=True)
        .execute()
    )
    arts = arts_res.data or []

    # choose first artifact per run as preview
    preview_by_run = {}
    for a in arts:
        rid = a["run_id"]
        if rid in preview_by_run:
            continue
        signed = supabase.storage.from_(a["bucket"]).create_signed_url(a["path"], SIGNED_URL_TTL)
        preview_by_run[rid] = signed.get("signedURL") if signed else None

    # 4) shape response
    out = []
    for r in runs:
        rid = r["id"]
        out.append({
            "run_id": rid,
            "status": r["status"],
            "created_at": r["created_at"],
            "horse_id": r["horse_id"],
            "model_name": r["model_name"],
            "model_version": r["model_version"],
            "score": score_by_run.get(rid),
            "preview_url": preview_by_run.get(rid),
        })

    return {"runs": out}