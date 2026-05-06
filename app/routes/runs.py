from fastapi import APIRouter, HTTPException
from cachetools import TTLCache
from app.supabase_client import supabase

router = APIRouter()

SIGNED_URL_TTL = 3600
# Cache slightly under the URL TTL so we never hand out an expired URL.
# maxsize is generous; tune to your traffic.
_signed_url_cache: TTLCache = TTLCache(maxsize=10000, ttl=SIGNED_URL_TTL - 600)


def _sign_paths(bucket: str, paths: list[str]) -> dict[str, str | None]:
    """
    Return {path: signed_url} for the given bucket.
    Uses an in-process cache and only calls Supabase for the cache misses,
    in a single batch request.
    """
    result: dict[str, str | None] = {}
    misses: list[str] = []

    for p in paths:
        cached = _signed_url_cache.get((bucket, p))
        if cached is not None:
            result[p] = cached
        else:
            misses.append(p)

    if not misses:
        return result

    try:
        signed_list = supabase.storage.from_(bucket).create_signed_urls(
            misses, SIGNED_URL_TTL
        )
    except Exception:
        # Fall back to None for everything we couldn't sign.
        for p in misses:
            result[p] = None
        return result

    # supabase-py returns a list of dicts aligned with the input paths.
    # Key casing has varied across versions, so check both.
    for entry in signed_list or []:
        path = entry.get("path")
        url = entry.get("signedURL") or entry.get("signedUrl")
        if path is None:
            continue
        if url:
            _signed_url_cache[(bucket, path)] = url
        result[path] = url

    # Anything the API silently dropped → None
    for p in misses:
        result.setdefault(p, None)

    return result


def _sign_artifacts(artifacts: list[dict]) -> dict[tuple[str, str], str | None]:
    """
    Sign a mixed list of artifacts (each with bucket+path) in as few calls
    as possible by grouping by bucket. Returns {(bucket, path): url}.
    """
    by_bucket: dict[str, list[str]] = {}
    for a in artifacts:
        b, p = a.get("bucket"), a.get("path")
        if b and p:
            by_bucket.setdefault(b, []).append(p)

    out: dict[tuple[str, str], str | None] = {}
    for bucket, paths in by_bucket.items():
        # de-dupe within the bucket to avoid sending the same path twice
        unique_paths = list(dict.fromkeys(paths))
        signed = _sign_paths(bucket, unique_paths)
        for p, url in signed.items():
            out[(bucket, p)] = url
    return out


@router.get("/runs")
def list_runs(limit: int = 50, offset: int = 0):
    runs_res = (
        supabase.table("runs")
        .select(
            "id,status,created_at,horse_id,model_name,model_version",
            count="exact",
        )
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    runs = runs_res.data or []
    if not runs:
        return {"runs": [], "total": runs_res.count, "limit": limit, "offset": offset}

    run_ids = [r["id"] for r in runs]
    status_by_run = {r["id"]: r["status"] for r in runs}

    preds_res = (
        supabase.table("predictions")
        .select("run_id,metric,value")
        .in_("run_id", run_ids)
        .execute()
    )
    preds = preds_res.data or []

    score_by_run: dict[str, float] = {}
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

    # Pick exactly one preview artifact per succeeded run (newest first,
    # which is how arts is ordered already).
    preview_choice: dict[str, dict] = {}
    for a in arts:
        rid = a["run_id"]
        if status_by_run.get(rid) != "succeeded":
            continue
        if rid in preview_choice:
            continue
        if a.get("bucket") and a.get("path"):
            preview_choice[rid] = a

    # Single batched signing pass for all chosen previews.
    signed_map = _sign_artifacts(list(preview_choice.values()))

    preview_by_run: dict[str, str | None] = {
        rid: signed_map.get((a["bucket"], a["path"]))
        for rid, a in preview_choice.items()
    }

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

    return {
        "runs": out,
        "total": runs_res.count,
        "limit": limit,
        "offset": offset,
    }


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
    arts = arts_res.data or []

    # Batch-sign every artifact for this run in one go (per bucket).
    signed_map = _sign_artifacts(arts)
    signed_artifacts = [
        {**a, "signed_url": signed_map.get((a.get("bucket"), a.get("path")))}
        for a in arts
    ]

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


@router.delete("/runs/{run_id}")
def delete_run(run_id: str):
    # 1) fetch artifacts first (so we can delete files)
    arts_res = (
        supabase.table("artifacts")
        .select("bucket,path")
        .eq("run_id", run_id)
        .execute()
    )
    arts = arts_res.data or []

    # 2) delete files from Storage (best effort), grouped per bucket.
    buckets: dict[str, list[str]] = {}
    for a in arts:
        b = a.get("bucket")
        p = a.get("path")
        if b and p:
            buckets.setdefault(b, []).append(p)

    for bucket, paths in buckets.items():
        try:
            supabase.storage.from_(bucket).remove(paths)
        except Exception:
            # don't block deletion if a file is already gone
            pass
        # invalidate any cached signed URLs for these paths
        for p in paths:
            _signed_url_cache.pop((bucket, p), None)

    # 3) delete the run row (predictions/artifacts cascade if FKs are set up)
    del_res = supabase.table("runs").delete().eq("id", run_id).execute()
    if not del_res.data:
        raise HTTPException(status_code=404, detail="Run not found")

    return {"ok": True, "deleted_run_id": run_id}