from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.inference_engine import FinalEvaluator


# 全域變數，啟動時載入一次（BERT 大模型，載入很慢，不要每次 request 載）
evaluator: FinalEvaluator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global evaluator
    print("⏳ 載入 BERT 模型中（首次啟動約 30-60 秒）...")
    evaluator = FinalEvaluator()
    print("✅ 模型已就緒")
    yield
    # 關閉時不用做什麼，PyTorch 會自動釋放


app = FastAPI(
    title="斷捨離 AI Service",
    description="BERT 三模型推論服務，給後端 (FastAPI) 呼叫",
    version="1.0.0",
    lifespan=lifespan,
)


class EvaluateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


@app.get("/health")
async def health():
    """後端會 ping 這個確認模型有沒有載完"""
    return {"status": "ok", "model_ready": evaluator is not None}


@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    """
    回傳格式跟 FinalEvaluator.evaluate() 完全一致：
    {
      "text": "...",
      "final_score": 72.5,
      "decision": "保留 (Keep)",
      "details": {
        "emotion":   {"label": "love", "conf": 0.87},
        "intensity": 0.92,
        "maslow":    {"label": "legacy", "conf": 0.74},
        "reiss":     {"label": "family", "conf": 0.68}
      }
    }
    """
    if evaluator is None:
        raise HTTPException(503, "模型還在載入中")
    try:
        return evaluator.evaluate(req.text)
    except Exception as e:
        raise HTTPException(500, f"推論失敗：{type(e).__name__}: {e}")
