"""
分數計算規則表。
所有「為什麼這個分數？」的答案都在這個檔，老師問起來能直接秒答。

設計原則：
  - 規則表寫死，可被 unit test 驗證
  - 函式純粹（無 side effect、同輸入永遠同輸出）
  - 數字都能解釋（不是隨便調的）
"""
from typing import Optional
from schemas.item import RawEvaluation


# ─────────────────────────────────────────────────────────
# 使用價值 (use_value)：根據使用頻率 + 使用時間
# ─────────────────────────────────────────────────────────
# 頻率是主要決定因素（佔比較大），時間是修正項
# 想法：每天用 = 強烈訊號還在發揮作用；
#       全新未拆 = 訊號模糊（可能是預備品，也可能是衝動購物失敗）

_FREQUENCY_SCORE = {
    "daily":   80,   # 每天用 → 高使用價值
    "weekly":  60,
    "monthly": 35,
    "yearly":  10,   # 一年才一次 → 低
    "new":     25,   # 全新未拆 → 給中低分，後續看其他維度
}

_PERIOD_BONUS = {
    "within_1_year":  15,   # 剛買沒多久且還在用 → 加分
    "1_to_3_years":   10,
    "3_to_5_years":    5,
    "5_to_8_years":    0,
    "over_8_years":   -5,   # 用很久且還在用 → 中性偏負（已折舊）
}


def compute_use_value(usage_period: str, usage_frequency: str) -> int:
    """0-100"""
    base = _FREQUENCY_SCORE.get(usage_frequency, 25)
    bonus = _PERIOD_BONUS.get(usage_period, 0)
    return max(0, min(100, base + bonus))


# ─────────────────────────────────────────────────────────
# 二手價值 (secondhand_value)：類別基礎分 × 折舊 × 品牌
# ─────────────────────────────────────────────────────────

_CATEGORY_BASE = {
    "furniture_bedding":    35,   # 大型家具運費高，流動性差
    "electronics_3c":       55,   # 3C 有市場但折舊極快
    "clothing_accessories": 35,   # 二手衣市場活絡
    "beauty_personal_care": 10,   # 開封過幾乎沒人要
    "books_office":         25,   # 書有 BookOff、二手書店
    "kitchen_living":       20,
    "sports_hobbies":       45,   # 樂器、運動器材保值
    "memorabilia":           5,   # 沒人會買別人的回憶
    "other":                20,
}

_PERIOD_DEPRECIATION = {
    "within_1_year":  +15,
    "1_to_3_years":     0,
    "3_to_5_years":   -10,
    "5_to_8_years":   -20,
    "over_8_years":   -30,
}

_FREQUENCY_CONDITION_MOD = {
    "new":      +10,   # 全新未拆 → 二手價最好
    "daily":    -10,   # 天天用 → 磨損嚴重
    "weekly":    -5,
    "monthly":    0,
    "yearly":     5,   # 幾乎沒用 → 接近新品
}


def compute_secondhand_value(
    category: str,
    usage_period: str,
    usage_frequency: str,
    brand: Optional[str] = None,
) -> int:
    """0-100"""
    base = _CATEGORY_BASE.get(category, 20)
    depreciation = _PERIOD_DEPRECIATION.get(usage_period, 0)
    condition = _FREQUENCY_CONDITION_MOD.get(usage_frequency, 0)
    brand_bonus = 15 if (brand and brand.strip()) else 0

    score = base + depreciation + condition + brand_bonus
    return max(0, min(100, score))


# ─────────────────────────────────────────────────────────
# 情感價值 (emotional_value)：從 BERT 的 details 推導
# ─────────────────────────────────────────────────────────
# 邏輯：
#   - intensity 是情感「有多強」（0-1.5）
#   - emotion.conf 是模型對情緒分類的信心
#   - maslow=legacy/social 或 reiss=family/honor/idealism 都是強情感連結的指標

_HIGH_EMOTION_MASLOW = {"legacy", "social"}
_HIGH_EMOTION_REISS = {"family", "honor", "idealism", "romance"}


def compute_emotional_value(raw: RawEvaluation) -> int:
    """0-100"""
    d = raw.details

    # 1. 強度 × 信心度當主體（最高約 87 分：intensity=1.5, conf=1.0 → 87 + bonus）
    intensity_pct = (d.intensity / 1.5) * 100
    base = intensity_pct * d.emotion.conf

    # 2. 心理動機加成
    bonus = 0.0
    if d.maslow.label in _HIGH_EMOTION_MASLOW:
        bonus += 15 * d.maslow.conf
    if d.reiss.label in _HIGH_EMOTION_REISS:
        bonus += 10 * d.reiss.conf

    return max(0, min(100, round(base + bonus)))


# ─────────────────────────────────────────────────────────
# 總分與建議
# ─────────────────────────────────────────────────────────

def compute_total_score(use: int, emotional: int, secondhand: int) -> int:
    """
    加權平均：情感權重高一點（因為斷捨離主要是心理戰）
    """
    total = 0.4 * emotional + 0.3 * use + 0.3 * secondhand
    return round(total)


def derive_recommendation(use: int, emotional: int, secondhand: int) -> str:
    """
    回 4 種建議文字，前端 badge 直接顯示。
    優先序：情感 > 實用 > 變現 > 處理
    """
    if emotional >= 70:
        return "建議保留"
    if use >= 60:                          # 高頻使用 → 純實用就值得留
        return "建議保留"
    if secondhand >= 60:
        return "建議出售"
    if use < 30 and emotional < 30 and secondhand >= 25:
        return "建議捐贈"
    if max(use, emotional, secondhand) < 25:
        return "可以丟棄"
    # fallback：分數普通 → 給一個比較中性的建議
    return "建議捐贈" if secondhand < 40 else "建議出售"
