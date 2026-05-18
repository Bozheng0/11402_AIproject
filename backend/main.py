"""
FastAPI backend (adapter)
啟動：uvicorn main:app --port 8080 --reload
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from config import settings
from routers import predict
from services import ai_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🔌 AI server: {settings.ai_service_url}")
    yield
    await ai_client.shutdown()


app = FastAPI(title="斷捨離 Adapter", version="0.4.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)


@app.get("/health")
async def health():
    ai_alive = await ai_client.health()
    return {
        "backend": "healthy",
        "ai_service": "healthy" if ai_alive else "unreachable",
        "ai_service_url": settings.ai_service_url,
    }


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE_DIR, "templates", "index.html"))

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
