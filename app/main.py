from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from redis import Redis
from rq import Queue
from app.routes.infer import router as infer_router
from app.routes.runs import router as runs_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://m37conformation.lovable.app",
        "https://move37conformation.com",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    try:
        r = Redis.from_url("redis://redis:6379/0")
        q = Queue("conformation", connection=r)
        workers = q.workers
        if not workers:
            return Response(status_code=503)
        return {"status": "ok", "workers": len(workers)}
    except Exception:
        return Response(status_code=503)

app.include_router(infer_router)
app.include_router(runs_router)
app.mount("/data", StaticFiles(directory="/data"), name="data")
app.mount("/ui", StaticFiles(directory="app/static", html=True), name="static")