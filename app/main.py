from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes.infer import router as infer_router
from app.routes.runs import router as runs_router
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.mount("/data", StaticFiles(directory="/data"), name="data")

app.include_router(infer_router)
app.include_router(runs_router)

app.mount("/ui", StaticFiles(directory="app/static", html=True), name="static")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://equine-pose-insight.lovable.app",  # Lovable frontend
        "http://localhost:3000",            # optional local dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

