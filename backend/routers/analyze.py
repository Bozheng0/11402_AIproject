"""
物品分析 endpoint。
流程：前端 → 你的後端 → AI Service (BERT) → 你的後端 → OpenAI (narrative) → 前端
"""
import asyncio
from fastapi import APIRouter, HTTPException

from schemas.item import AnalyzeRequest, AnalyzeResponse
from services import ai_client, llm_service
from config import settings

router = APIRouter(prefix="/api", tags=["analyze"])


def _decision_to_simple(decision_zh: str) -> str:
    """組員的格式是『保留 (Keep)』/『建議斷捨離 (Discard)』，前端用簡單值方便判斷"""
    return "keep" if "Keep" in decision_zh or "保留" in decision_zh else "discard"


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """
    POST /api/analyze
    Body: {"text": "...", "include_narrative": true}
    """
    # 1. 先打 BERT 拿原始分析
    try:
        raw = await ai_client.evaluate(req.text)
    except ai_client.AIServiceError as e:
        raise HTTPException(502, f"AI 模型服務錯誤：{e}")

    # 2. 並行：如果要 narrative 就同時跑 LLM（這裡其實只有一個 task，但留給以後好擴充）
    narrative = None
    if req.include_narrative and settings.enable_llm_narrative and settings.openai_api_key:
        try:
            narrative = await llm_service.generate_narrative(raw)
        except Exception as e:
            # narrative 失敗不應該擋住主要結果 —— 退化掉就好
            narrative = None

    return AnalyzeResponse(
        raw=raw,
        decision_simple=_decision_to_simple(raw.decision),
        narrative=narrative,
    )
