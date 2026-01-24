from fastapi import FastAPI
from starlette.staticfiles import StaticFiles

from api.routes import router

app = FastAPI(
    title="LoL Analytics API",
    description="PySpark + FastAPI backend",
    version="1.0"
)

app.include_router(router)

app.mount(
    "/",
    StaticFiles(directory="client", html=True),
    name="client"
)
