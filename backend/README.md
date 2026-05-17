# 斷捨離 Adapter (Backend)

> 第三組專題 · 前端跟同學 ai_server 之間的翻譯層

## 架構

```
[瀏覽器]
   │ POST /predict (前端契約)
   ▼
[Adapter :8080] ← 本資料夾
   │ POST /predict  (ai_server 契約)
   │ POST /explain
   ▼
[同學 ai_server :8000]
```

## 啟動

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --port 8080 --reload
```

開 <http://localhost:8080/health> 確認跟 ai_server 連得上。

## API

### `POST /predict`
**收**（前端 script.js 送的格式）：
```json
{"item_name":"...","brand":"...","category":"electronics_3c",
 "usage_period":"over_8_years","usage_frequency":"yearly",
 "objective_description":"...","emotional_description":"..."}
```

**回**：
```json
{
  "item_name": "...",
  "recommendation": "建議保留/出售/捐贈/可以丟棄",
  "total_score": 0-100,
  "use_value": 0-100,
  "emotional_value": 0-100,
  "secondhand_value": 0-100,
  "secondhand_price_usd": 45.23,   ← 新增！前端要顯示「$45 USD」
  "reason": "Gemini 寫的解釋文字"
}
```

## 翻譯邏輯（給組員看）

| 欄位 | 來源 |
|---|---|
| `category` → `category_name` | services/category_map.py（給同學 review） |
| `usage_frequency` → `item_condition_id` | new→1, yearly→2, monthly/weekly→3, daily→4 |
| `shipping` | 寫死 1（買家付）— 同學確認 |
| `usage_period` + `usage_frequency` | 拼進 text_input 前綴 |
| `secondhand` (USD) → `secondhand_value` (0-100) | services/scoring.py，分段對齊同學 fuzzy zones |
| `usevalue` (0-4) → `use_value` (0-100) | × 25 |
| `final_decision` (EN) → `recommendation` (中文) | KEEP→建議保留 等 |

## 檔案

```
backend/
├── main.py
├── config.py
├── requirements.txt
├── .env.example
├── routers/
│   └── predict.py        ← 主流程
├── services/
│   ├── ai_client.py      ← 打 ai_server /predict + /explain
│   ├── scoring.py        ← USD→0-100、decision→中文
│   ├── category_map.py   ← 9 enum → Mercari path（草稿）
│   └── llm_service.py    ← reason 退化模板
└── schemas/item.py       ← 前端契約 + ai_server 契約
```
