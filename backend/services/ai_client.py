"""
AI Service Client。
唯一一個跟組員的 BERT 服務溝通的地方。
"""
import httpx
from config import settings
from schemas.item import RawEvaluation


class AIServiceError(Exception):
    """AI service 出問題時拋這個，上層 endpoint 會轉成 502 給前端"""


# 共用 client；async with 比每次新開連線快
_client = httpx.AsyncClient(
    base_url=settings.ai_service_url,
    timeout=settings.ai_timeout_sec,
)


async def evaluate(text: str) -> RawEvaluation:
    """
    打組員的 POST /evaluate，回傳 BERT 模型的原始分析結果。

    對應到組員 inference_engine.py 的 FinalEvaluator.evaluate(text)。
    """
    try:
        resp = await _client.post("/evaluate", json={"text": text})
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise AIServiceError(f"AI service 回 {e.response.status_code}: {e.response.text}") from e
    except httpx.HTTPError as e:
        raise AIServiceError(f"AI service 連線失敗：{e}") from e

    data = resp.json()
    # 用 Pydantic 驗證對方回傳格式 —— 對方改格式我們會立刻爆，不會回髒資料給前端
    return RawEvaluation(**data)


async def health() -> bool:
    """檢查 AI service 還活著嗎"""
    try:
        resp = await _client.get("/health", timeout=3.0)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


async def shutdown() -> None:
    """app 關閉時呼叫"""
    await _client.aclose()
