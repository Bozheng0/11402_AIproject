"""
LLM 服務：把 BERT 分數 + 規則分數翻成人話。
沒有 OpenAI key 時，退化成模板字串，前端不會看到空白。
"""
from typing import Optional
from openai import AsyncOpenAI

from config import settings
from schemas.item import (
    RawEvaluation, ChatMessage, AnalyzeResponse, PredictRequest,
)


_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise RuntimeError("沒有設定 OPENAI_API_KEY")
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


def llm_available() -> bool:
    return bool(settings.openai_api_key) and settings.enable_llm_narrative


# ═════════════════════════════════════════════════════════════
# /predict 的 reason 文字（主要用途）
# ═════════════════════════════════════════════════════════════

REASON_SYSTEM = """你是溫柔且務實的斷捨離顧問。
我會給你一件物品的資訊與系統算出的分數，請寫一段給使用者看的建議。

語氣與格式要求：
- 繁體中文，3 到 5 句話
- 溫和、不強迫，尊重物品在使用者生活中的意義
- 不要把數字念出來（不要說「您的情感分數是 80」），要把分數翻譯成感受與意義
- 結尾給一個可執行的小建議（拍張照、放到展示架、二手平台關鍵字、捐贈管道等）
- 純文字，不要 markdown
"""


def _build_reason_user_prompt(
    req: PredictRequest,
    raw: RawEvaluation,
    use: int, emotional: int, secondhand: int,
    recommendation: str,
) -> str:
    d = raw.details
    return (
        f"## 物品資訊\n"
        f"- 名稱：{req.item_name}\n"
        f"- 品牌：{req.brand or '（未填）'}\n"
        f"- 類別：{req.category}\n"
        f"- 使用了：{req.usage_period}\n"
        f"- 使用頻率：{req.usage_frequency}\n"
        f"- 客觀描述：{req.objective_description or '（未填）'}\n"
        f"- 情感描述：{req.emotional_description or '（未填）'}\n\n"
        f"## AI 分析結果\n"
        f"- 系統建議：{recommendation}\n"
        f"- 使用價值：{use}/100\n"
        f"- 情感價值：{emotional}/100\n"
        f"- 二手價值：{secondhand}/100\n\n"
        f"## BERT 模型偵測到的心理訊號\n"
        f"- 主要情緒：{d.emotion.label}（信心 {d.emotion.conf:.0%}）\n"
        f"- 情感強度：{d.intensity:.2f}/1.5\n"
        f"- 深層需求 (Maslow)：{d.maslow.label}\n"
        f"- 心理動機 (Reiss)：{d.reiss.label}\n"
    )


async def generate_reason(
    req: PredictRequest,
    raw: RawEvaluation,
    use: int, emotional: int, secondhand: int,
    recommendation: str,
) -> str:
    """主要產生 reason 的入口。沒 LLM 就退化成模板"""
    if not llm_available():
        return _fallback_reason(req, use, emotional, secondhand, recommendation)

    try:
        response = await _get_client().chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": REASON_SYSTEM},
                {"role": "user", "content": _build_reason_user_prompt(
                    req, raw, use, emotional, secondhand, recommendation,
                )},
            ],
            temperature=0.6,
            max_tokens=300,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        # LLM 掛掉不應該擋住主流程 → 退化
        return _fallback_reason(req, use, emotional, secondhand, recommendation)


def _fallback_reason(
    req: PredictRequest,
    use: int, emotional: int, secondhand: int,
    recommendation: str,
) -> str:
    """沒 OpenAI key 或 LLM 失敗時的退化模板"""
    name = req.item_name

    if emotional >= 70:
        return f"從你的描述中可以感覺到「{name}」對你有很深的意義。即使使用頻率不高，這份情感連結本身就是值得珍藏的理由。可以考慮為它找一個專屬的位置，而不只是放在收納櫃裡。"
    if use >= 60:
        return f"「{name}」還持續在你的生活中發揮作用，這就是繼續留著它的最好理由。建議繼續使用，等到真的不再用時再來重新評估。"
    if secondhand >= 60:
        return f"「{name}」目前在二手市場上仍有不錯的價值。如果你已經沒在用、情感連結也不深，趁狀況還好的時候找個合適的買家，讓它在另一個人的生活裡繼續被需要。"
    if recommendation == "建議捐贈":
        return f"「{name}」對你而言可能已經完成階段性任務了。如果功能仍正常，可以考慮捐贈給需要的人或單位（如二手商店、社福機構），讓它繼續被使用，也讓你的空間更輕盈。"
    return f"從目前的資訊看起來，「{name}」在實用、情感、二手三個面向都不算突出。如果你已經為它努力過了，或許可以放手 — 但記得，斷捨離不是丟東西，是讓空間留給更重要的事物。"


# ═════════════════════════════════════════════════════════════
# /api/analyze 的 narrative（純文字 API 用）
# ═════════════════════════════════════════════════════════════

NARRATIVE_SYSTEM = """你是溫柔的斷捨離顧問。
使用者剛剛輸入一段對物品的描述，AI 模型給出了情緒、強度與心理動機分析。
請根據這些分析，用 3-4 句話、繁體中文，溫和地告訴使用者「為什麼建議這樣處理」。
不要重複數字，把它翻譯成意義。結尾給一個具體小建議。
"""


async def generate_narrative(raw: RawEvaluation) -> str:
    if not llm_available():
        return ""
    d = raw.details
    user_msg = (
        f"使用者描述：「{raw.text}」\n"
        f"主要情緒：{d.emotion.label}（信心 {d.emotion.conf:.0%}）\n"
        f"情感強度：{d.intensity:.2f}/1.5\n"
        f"深層需求：{d.maslow.label}\n"
        f"心理動機：{d.reiss.label}\n"
        f"綜合評分：{raw.final_score}/100，建議：{raw.decision}"
    )

    response = await _get_client().chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": NARRATIVE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.6,
        max_tokens=250,
    )
    return (response.choices[0].message.content or "").strip()


# ═════════════════════════════════════════════════════════════
# /api/chat 多輪對話
# ═════════════════════════════════════════════════════════════

CHAT_SYSTEM = """你是斷捨離輔助助理。回答要溫和、簡短（最多 3 句話）、繁體中文。"""


async def chat_reply(
    messages: list[ChatMessage],
    last_analysis: Optional[AnalyzeResponse] = None,
) -> str:
    sys = CHAT_SYSTEM
    if last_analysis:
        sys += (
            f"\n上一輪 AI 評分：{last_analysis.raw.final_score}/100，"
            f"建議：{last_analysis.raw.decision}。\n"
            f"使用者描述：「{last_analysis.raw.text}」"
        )

    openai_messages = [{"role": "system", "content": sys}]
    for m in messages:
        openai_messages.append({"role": m.role, "content": m.content})

    response = await _get_client().chat.completions.create(
        model=settings.openai_model,
        messages=openai_messages,
        temperature=0.7,
        max_tokens=200,
    )
    return (response.choices[0].message.content or "").strip()
