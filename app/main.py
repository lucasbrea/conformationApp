from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes.infer import router as infer_router
from app.routes.runs import router as runs_router
from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

app = FastAPI()
app.mount("/data", StaticFiles(directory="/data"), name="data")

app.include_router(infer_router)
app.include_router(runs_router)




