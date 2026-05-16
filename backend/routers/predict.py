"""
/predict — 前端 itemForm submit 進來的主要端點。

完整流程：
    前端 7 個欄位
        ↓
    組合成 BERT 可吃的文字
        ↓
    打 AI Service (BERT) → 拿情緒 / 強度 / 動機
        ↓
    ┌─────────────┬───────────────┬──────────────────┐
    ↓             ↓               ↓
  情感分數        使用分數          二手分數
  (BERT 推導)    (規則表)         (規則表)
    └─────────────┴───────────────┴──────────────────┘
        ↓
    總分加權 + 建議推導
        ↓
    LLM 寫 reason（沒 key 就退化模板）
        ↓
    回傳前端
"""
from fastapi import APIRouter, HTTPException

from schemas.item import PredictRequest, PredictResponse
from services import ai_client, llm_service, scoring

router = APIRouter(tags=["predict"])


def _compose_bert_text(req: PredictRequest) -> str:
    """
    把結構化欄位拼成 BERT 可以理解的一段文字。
    優先用情感描述（這是 BERT 最能發揮的地方），
    再補客觀描述當 context。
    """
    parts = []
    if req.emotional_description.strip():
        parts.append(req.emotional_description.strip())
    if req.objective_description.strip():
        parts.append(req.objective_description.strip())

    if not parts:
        # fallback：兩段描述都空，用基本資訊湊一句
        parts.append(f"這是一個{req.category}類別的物品，叫做{req.item_name}")

    return "\n".join(parts)[:1500]  # BERT max_length 截斷


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    POST /predict — 前端 itemForm 的目標端點。
    """
    # 1. 組文字 → 打 BERT
    bert_text = _compose_bert_text(req)
    try:
        raw = await ai_client.evaluate(bert_text)
    except ai_client.AIServiceError as e:
        raise HTTPException(502, f"AI 模型服務錯誤：{e}")

    # 2. 三維分數
    emotional = scoring.compute_emotional_value(raw)
    use = scoring.compute_use_value(req.usage_period, req.usage_frequency)
    secondhand = scoring.compute_secondhand_value(
        req.category, req.usage_period, req.usage_frequency, req.brand,
    )

    # 3. 總分 + 建議
    total = scoring.compute_total_score(use, emotional, secondhand)
    recommendation = scoring.derive_recommendation(use, emotional, secondhand)

    # 4. reason 文字（不會擋住主流程；LLM 掛掉會退化成模板）
    reason = await llm_service.generate_reason(
        req=req, raw=raw,
        use=use, emotional=emotional, secondhand=secondhand,
        recommendation=recommendation,
    )

    return PredictResponse(
        item_name=req.item_name,
        recommendation=recommendation,
        total_score=total,
        use_value=use,
        emotional_value=emotional,
        secondhand_value=secondhand,
        reason=reason,
    )
