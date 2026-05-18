"""
與同學 ai_server (port 8000) 溝通的唯一一個檔。
只負責 HTTP 呼叫，不做翻譯（翻譯交給 routers/predict.py）。
"""
import httpx
from config import settings
from schemas.item import (
    AIServerPredictRequest, AIServerPredictResponse,
    AIServerExplainRequest, AIServerExplainResponse,
)


class AIServiceError(Exception):
    pass


_client = httpx.AsyncClient(
    base_url=settings.ai_service_url,
    timeout=settings.ai_timeout_sec,
)


async def predict(req: AIServerPredictRequest) -> AIServerPredictResponse:
    try:
        r = await _client.post("/predict", json=req.model_dump())
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise AIServiceError(f"/predict 回 {e.response.status_code}: {e.response.text[:200]}") from e
    except httpx.HTTPError as e:
        raise AIServiceError(f"/predict 連線失敗：{e}") from e
    return AIServerPredictResponse(**r.json())


async def explain(req: AIServerExplainRequest) -> str:
    """同學的 /explain 用 Gemini 寫人話，失敗時 caller 自己 fallback"""
    try:
        r = await _client.post("/explain", json=req.model_dump())
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise AIServiceError(f"/explain 失敗：{e}") from e
    return AIServerExplainResponse(**r.json()).reason


async def health() -> bool:
    try:
        r = await _client.get("/health", timeout=3.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


async def shutdown() -> None:
    await _client.aclose()
