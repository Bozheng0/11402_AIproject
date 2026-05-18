"""
Reason 模板退化（當同學 /explain 掛掉或沒設 Gemini key 時用）。
"""
from schemas.item import PredictRequest


def fallback_reason(
    req: PredictRequest,
    use: int, emotional: int, secondhand: int,
    recommendation: str,
) -> str:
    n = req.item_name
    if "保留" in recommendation:
        if emotional >= 60:
            return f"從你的描述能感覺到「{n}」對你有很深的意義。即使使用頻率不一定高，這份情感連結本身就值得珍藏。可以為它找一個專屬的位置。"
        return f"「{n}」仍持續在你的生活中發揮作用，這就是繼續留著它的好理由。建議繼續使用。"
    if "出售" in recommendation:
        return f"「{n}」在二手市場上仍有不錯的價值。趁狀況還好時找個合適的買家，讓它在別人的生活裡繼續被需要。"
    if "捐贈" in recommendation:
        return f"「{n}」可能已經完成它在你生活中的階段性任務了。可以考慮捐贈給需要的人，讓它繼續被使用。"
    return f"「{n}」在三個面向都不算突出，如果你已經為它努力過了，或許可以放手 —— 斷捨離不是丟東西，是讓空間留給更重要的事物。"
