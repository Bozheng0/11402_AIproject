"""
對話 endpoint —— 純 LLM，跟 BERT 模型無關。
使用者分析完物品後可以繼續跟 AI 討論。
"""
from fastapi import APIRouter, HTTPException

from schemas.item import ChatRequest, ChatResponse
from services import llm_service
from config import settings

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    POST /api/chat
    Body: {"session_id": "...", "messages": [...], "last_analysis": {...} | null}
    """
    if not settings.openai_api_key:
        raise HTTPException(503, "對話功能未啟用（後端沒有設定 OPENAI_API_KEY）")
    if not req.messages:
        raise HTTPException(400, "messages 不能為空")
    if req.messages[-1].role != "user":
        raise HTTPException(400, "最後一句必須是 user")

    try:
        reply = await llm_service.chat_reply(req.messages, req.last_analysis)
    except Exception as e:
        raise HTTPException(502, f"LLM 服務錯誤：{type(e).__name__}")

    return ChatResponse(session_id=req.session_id, reply=reply)
