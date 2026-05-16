"""
Pydantic Schema：前後端的資料合約。
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ═════════════════════════════════════════════════════════════
# /predict — 跟前端組員的 UI 對齊（主要 endpoint）
# ═════════════════════════════════════════════════════════════

CategoryT = Literal[
    "furniture_bedding", "electronics_3c", "clothing_accessories",
    "beauty_personal_care", "books_office", "kitchen_living",
    "sports_hobbies", "memorabilia", "other",
]
PeriodT = Literal[
    "within_1_year", "1_to_3_years", "3_to_5_years",
    "5_to_8_years", "over_8_years",
]
FrequencyT = Literal["new", "daily", "weekly", "monthly", "yearly"]


class PredictRequest(BaseModel):
    """前端 itemForm submit 過來的資料"""
    item_name: str = Field(..., min_length=1, max_length=100)
    brand: Optional[str] = Field(None, max_length=100)
    category: CategoryT
    usage_period: PeriodT
    usage_frequency: FrequencyT
    objective_description: str = Field(default="", max_length=2000)
    emotional_description: str = Field(default="", max_length=2000)


class PredictResponse(BaseModel):
    """回給前端，欄位名字跟 script.js 的 renderResult(data) 對齊"""
    item_name: str
    recommendation: str            # "建議保留" / "建議出售" / "建議捐贈" / "可以丟棄"
    total_score: int               # 0-100
    use_value: int                 # 0-100
    emotional_value: int           # 0-100
    secondhand_value: int          # 0-100
    reason: str                    # 一段文字說明


# ═════════════════════════════════════════════════════════════
# /api/analyze — 給開發者用的純文字版（保留）
# ═════════════════════════════════════════════════════════════

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    include_narrative: bool = True


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    session_id: str
    messages: list[ChatMessage]
    last_analysis: Optional["AnalyzeResponse"] = None


# ═════════════════════════════════════════════════════════════
# AI Service 原始回應（對齊組員的 FinalEvaluator.evaluate 格式）
# ═════════════════════════════════════════════════════════════

class EmotionDetail(BaseModel):
    label: str
    conf: float


class AnalysisDetails(BaseModel):
    emotion: EmotionDetail
    intensity: float
    maslow: EmotionDetail
    reiss: EmotionDetail


class RawEvaluation(BaseModel):
    text: str
    final_score: float = Field(..., ge=0, le=100)
    decision: str
    details: AnalysisDetails


class AnalyzeResponse(BaseModel):
    raw: RawEvaluation
    decision_simple: Literal["keep", "discard"]
    narrative: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str


ChatRequest.model_rebuild()
