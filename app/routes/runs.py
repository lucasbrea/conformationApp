from fastapi import APIRouter, HTTPException
from app.supabase_client import supabase

router = APIRouter()

SIGNED_URL_TTL = 3600

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

@router.get("/runs")
def list_runs(limit: int = 50):
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
    status_by_run = {r["id"]: r["status"] for r in runs}

    preds_res = (
        supabase.table("predictions")
        .select("run_id,metric,value")
        .in_("run_id", run_ids)
        .execute()
    )
    preds = preds_res.data or []

    score_by_run = {}
    for p in preds:
        rid = p["run_id"]
        m = p["metric"]
        if m == "cr_score":
            score_by_run[rid] = p["value"]
        elif m == "score" and rid not in score_by_run:
            score_by_run[rid] = p["value"]

    arts_res = (
        supabase.table("artifacts")
        .select("run_id,bucket,path,created_at")
        .in_("run_id", run_ids)
        .order("created_at", desc=True)
        .execute()
    )
    arts = arts_res.data or []

    preview_by_run = {}
    for a in arts:
        rid = a["run_id"]
        if status_by_run.get(rid) != "succeeded":
            continue
        if rid in preview_by_run:
            continue

        bucket = a.get("bucket")
        path = a.get("path")
        if not bucket or not path:
            preview_by_run[rid] = None
            continue

        try:
            signed = supabase.storage.from_(bucket).create_signed_url(path, SIGNED_URL_TTL)
            preview_by_run[rid] = signed.get("signedURL") if signed else None
        except Exception:
            preview_by_run[rid] = None

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

    signed_artifacts = []
    for a in (arts_res.data or []):
        try:
            signed = supabase.storage.from_(a["bucket"]).create_signed_url(a["path"], SIGNED_URL_TTL)
            signed_url = signed.get("signedURL") if signed else None
        except Exception:
            signed_url = None

        signed_artifacts.append({**a, "signed_url": signed_url})

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