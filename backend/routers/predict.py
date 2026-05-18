"""
/predict — Adapter 主流程。
前端 → 翻譯 → 同學 ai_server /predict → 同學 ai_server /explain → 翻譯回前端格式
"""
from fastapi import APIRouter, HTTPException

from schemas.item import (
    PredictRequest, PredictResponse,
    AIServerPredictRequest, AIServerExplainRequest,
)
from services import ai_client, scoring, llm_service
from services.category_map import to_mercari_category, to_item_condition_id

router = APIRouter()


_PERIOD_ZH = {
    "within_1_year": "1 年內", "1_to_3_years": "1 至 3 年",
    "3_to_5_years": "3 至 5 年", "5_to_8_years": "5 至 8 年",
    "over_8_years": "8 年以上",
}
_FREQ_ZH = {
    "new": "全新未拆", "daily": "每天",
    "weekly": "每週", "monthly": "每月", "yearly": "每年",
}


def _compose_text_input(req: PredictRequest) -> str:
    """
    把 usage_period / usage_frequency / 兩段描述拼成一段給 sentiment + usevalue 模型。
    這是『前端送了但 ai_server schema 沒對應欄位』的解決方案。
    """
    parts = []
    period = _PERIOD_ZH.get(req.usage_period, req.usage_period)
    freq = _FREQ_ZH.get(req.usage_frequency, req.usage_frequency)
    parts.append(f"這件物品已經使用了{period}，使用頻率是{freq}。")

    if req.emotional_description.strip():
        parts.append(req.emotional_description.strip())
    if req.objective_description.strip():
        parts.append(req.objective_description.strip())

    return "\n".join(parts)[:1500]


def _to_ai_server_request(req: PredictRequest) -> AIServerPredictRequest:
    """前端格式 → 同學 ai_server /predict 格式"""
    return AIServerPredictRequest.model_validate({
        "secondhand_input": {
            "name": req.item_name,
            "item_condition_id": to_item_condition_id(req.usage_frequency),
            "category_name": to_mercari_category(req.category),
            "brand_name": req.brand or "",
            "shipping": 1,                                  # 寫死：買家付運費（同學確認）
            "item_description": req.objective_description or "",
        },
        "text_input": _compose_text_input(req),
    })


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    POST /predict — 前端 itemForm 打的端點。
    """
    ai_req = _to_ai_server_request(req)

    # 1. 主推論
    try:
        ai_res = await ai_client.predict(ai_req)
    except ai_client.AIServiceError as e:
        raise HTTPException(502, f"AI 服務錯誤：{e}")

    # 2. 三個 0-100 bar 分數
    secondhand_bar = scoring.usd_to_bar_score(ai_res.secondhand)
    emotional_bar = round(ai_res.sentiment)
    use_bar = scoring.usevalue_class_to_bar(ai_res.usevalue)

    total = scoring.compute_total_score(use_bar, emotional_bar, secondhand_bar)
    recommendation = scoring.decision_to_zh(ai_res.final_decision)

    # 3. reason — 主走 Gemini，失敗才退回模板
    try:
        reason = await ai_client.explain(AIServerExplainRequest(
            input_data=ai_req,
            predict_result=ai_res,
        ))
    except ai_client.AIServiceError:
        reason = llm_service.fallback_reason(req, use_bar, emotional_bar, secondhand_bar, recommendation)

    return PredictResponse(
        item_name=req.item_name,
        recommendation=recommendation,
        total_score=total,
        use_value=use_bar,
        emotional_value=emotional_bar,
        secondhand_value=secondhand_bar,
        secondhand_price_usd=round(ai_res.secondhand, 2),
        reason=reason,
    )
