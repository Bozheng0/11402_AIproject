"""
FastAPI 主程式。
啟動：  uvicorn main:app --reload --port 8000
文件：  http://localhost:8000/docs

伺服範圍：
  GET  /                  → 前端 index.html
  GET  /static/*          → 前端 CSS / JS
  POST /predict           → 主要的物品分析（前端 itemForm 打的）
  POST /api/analyze       → 開發者用的純文字 API
  POST /api/chat          → 後續對話（純 LLM）
  GET  /health            → 後端 + AI service 狀態
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers import analyze, chat, predict
from services import ai_client


# ─────────────────────────────────────────────────────────
# 前端檔案位置
# ─────────────────────────────────────────────────────────
# 預期專案結構（合進團隊 repo 後）：
#   11402_AIproject/
#     ├── backend/         ← 你
#     │   └── main.py      ← 這個檔
#     ├── templates/       ← 前端組員
#     │   └── index.html
#     └── static/          ← 前端組員
#         ├── style.css
#         └── script.js
#
# 用環境變數 FRONTEND_DIR 覆寫（本機測試時可指向別處）
BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", str(BACKEND_DIR.parent)))
TEMPLATES_DIR = FRONTEND_DIR / "templates"
STATIC_DIR = FRONTEND_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"📁 Frontend dir: {FRONTEND_DIR}")
    print(f"   templates exists: {TEMPLATES_DIR.exists()}")
    print(f"   static exists:    {STATIC_DIR.exists()}")
    print(f"🤖 AI service: {settings.ai_service_url}")
    print(f"💬 LLM narrative: {'enabled' if settings.openai_api_key else 'disabled (no key)'}")
    yield
    await ai_client.shutdown()


app = FastAPI(
    title="斷捨離 AI 後端",
    description="前端 ↔ BERT 模型 ↔ OpenAI 的中間層",
    version="0.3.0",
    lifespan=lifespan,
)

# CORS：前端 server-side render 在同 origin 其實不需要，
# 但前端組員若先用獨立 server 開發（例如 5500、5173）就需要
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_origin,
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API endpoints
app.include_router(predict.router)        # POST /predict
app.include_router(analyze.router)        # POST /api/analyze
app.include_router(chat.router)           # POST /api/chat


# ─────────────────────────────────────────────────────────
# 靜態檔 + 首頁
# ─────────────────────────────────────────────────────────
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    """首頁 → 前端 index.html"""
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            500,
            f"找不到前端 index.html。預期路徑：{index_file}。"
            "請確認 backend/ 跟 templates/ 是同一個 repo 的兄弟資料夾，"
            "或設定環境變數 FRONTEND_DIR 指向前端所在目錄。",
        )
    return FileResponse(str(index_file))


@app.get("/health")
async def health():
    ai_alive = await ai_client.health()
    return {
        "backend": "healthy",
        "ai_service": "healthy" if ai_alive else "unreachable",
        "ai_service_url": settings.ai_service_url,
        "llm_narrative": "enabled" if settings.openai_api_key else "disabled",
        "frontend_files_found": TEMPLATES_DIR.exists() and STATIC_DIR.exists(),
    }
