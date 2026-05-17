"""
分數轉換邏輯。
配合同學 ai_server 的 fuzzy 分段，把 USD 跟其他原始值轉成前端要的 0-100。
"""


def usd_to_bar_score(usd: float) -> int:
    """
    把美元金額轉成 0-100 進度條長度。
    分段邊界跟同學 fuzzy_inference.py 的 low/medium/high 對齊：
      $0     → 0
      $50    → 33   (low 的邊界)
      $150   → 66   (medium 的邊界)
      $300+  → 100  (high 的飽和點)
    """
    if usd <= 0:
        return 0
    if usd <= 50:
        return round(usd / 50 * 33)
    if usd <= 150:
        return round(33 + (usd - 50) / 100 * 33)
    if usd <= 300:
        return round(66 + (usd - 150) / 150 * 34)
    return 100


def usevalue_class_to_bar(pred_class: int) -> int:
    """
    同學的 usevalue 模型回的是 0-4 整數分類。
    乘以 25 變 0-100：0→0, 1→25, 2→50, 3→75, 4→100
    """
    return max(0, min(100, pred_class * 25))


_DECISION_ZH = {
    "KEEP":    "建議保留",
    "SELL":    "建議出售",
    "DONATE":  "建議捐贈",
    "DISCARD": "可以丟棄",
}


def decision_to_zh(en_decision: str) -> str:
    """KEEP → 建議保留"""
    return _DECISION_ZH.get(en_decision.upper().strip(), en_decision)


def compute_total_score(use: int, emotional: int, secondhand: int) -> int:
    """
    展示用的「總分」徽章：情感權重略高 (0.4) + 使用 + 二手 各 0.3。
    這只是視覺輔助 — 真正的建議來自 ai_server 的 fuzzy final_decision。
    """
    return round(0.4 * emotional + 0.3 * use + 0.3 * secondhand)
