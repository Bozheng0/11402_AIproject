"""
Pydantic Schema：前後端的資料合約。
v0.4 — 對接同學 ai_server (port 8000)。
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ═════════ /predict — 前端契約（不變動） ═════════

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
    item_name: str = Field(..., min_length=1, max_length=100)
    brand: Optional[str] = Field(None, max_length=100)
    category: CategoryT
    usage_period: PeriodT
    usage_frequency: FrequencyT
    objective_description: str = Field(default="", max_length=2000)
    emotional_description: str = Field(default="", max_length=2000)


class PredictResponse(BaseModel):
    """新增 secondhand_price_usd 給前端顯示美元金額"""
    item_name: str
    recommendation: str           # 建議保留/出售/捐贈/可以丟棄
    total_score: int              # 0-100（三項加權平均，做為展示用）
    use_value: int                # 0-100 bar 長度
    emotional_value: int          # 0-100
    secondhand_value: int         # 0-100 bar 長度（從 USD 用同學 fuzzy 分段換算）
    secondhand_price_usd: float   # 原始美元金額（前端顯示「預估 $XX」）
    reason: str                   # Gemini 寫的解釋


# ═════════ AI Server (:8000) 的契約 — 由同學 API.md 定義 ═════════

class _SecondhandInput(BaseModel):
    name: str
    item_condition_id: int
    category_name: str
    brand_name: str = ""
    shipping: int = 1
    item_description: str = ""


class AIServerPredictRequest(BaseModel):
    secondhand_input: _SecondhandInput
    text_input: str


class AIServerPredictResponse(BaseModel):
    secondhand: float          # USD 金額
    sentiment: float           # 0-100
    usevalue: int              # 0-4 class
    final_decision: str        # KEEP / SELL / DONATE / DISCARD


class AIServerExplainRequest(BaseModel):
    input_data: AIServerPredictRequest
    predict_result: AIServerPredictResponse


class AIServerExplainResponse(BaseModel):
    reason: str
