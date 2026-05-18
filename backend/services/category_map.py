"""
前端 category enum → Mercari 路徑字串 對應表。

⚠️ 草稿 — 待同學 review。Mercari 的類別樹很細，這裡選的是接近的常見父分類。
之後同學說 "這個對應到那個比較準" 就改這個檔即可。
"""

FRONTEND_TO_MERCARI: dict[str, str] = {
    "furniture_bedding":    "Home/Home Décor/Other",
    "electronics_3c":       "Electronics/Cell Phones & Accessories/Other",
    "clothing_accessories": "Women/Tops & Blouses/T-Shirts",
    "beauty_personal_care": "Beauty/Skin Care/Other",
    "books_office":         "Other/Books/Other",
    "kitchen_living":       "Home/Kitchen & Dining/Other",
    "sports_hobbies":       "Sports & Outdoors/Other/Other",
    "memorabilia":          "Vintage & Collectibles/Other/Other",
    "other":                "Other/Other/Other",
}


def to_mercari_category(frontend_category: str) -> str:
    return FRONTEND_TO_MERCARI.get(frontend_category, "Other/Other/Other")


# 從 usage_frequency 推 item_condition_id (Mercari 1-5)
# 1=New, 2=Like new, 3=Good, 4=Fair, 5=Poor
_FREQ_TO_CONDITION = {
    "new":     1,   # 全新未拆 → New
    "yearly":  2,   # 一年才一次 → 接近全新
    "monthly": 3,   # 偶爾 → Good
    "weekly":  3,   # 每週 → Good
    "daily":   4,   # 天天用 → Fair
}


def to_item_condition_id(usage_frequency: str) -> int:
    return _FREQ_TO_CONDITION.get(usage_frequency, 3)
