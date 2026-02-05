from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

app.include_router(infer_router)
app.include_router(runs_router)

app.mount("/data", StaticFiles(directory="/data"), name="data")
app.mount("/ui", StaticFiles(directory="app/static", html=True), name="static")