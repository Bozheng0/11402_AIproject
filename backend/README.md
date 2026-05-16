# 斷捨離 AI 後端

> 第三組專題 · 後端 + Web 層（FastAPI）
> 串起前端 UI、組員的 BERT 模型、OpenAI 三方

## 架構

```
[瀏覽器] ──HTTP──▶ [FastAPI 後端 :8000] ──HTTP──▶ [AI Service :8001]
                        │ ↑ 同時負責：                     ↑ 載入 BERT × 3
                        │   1. 伺服 index.html / static
                        │   2. /predict 接前端表單
                        │
                        │   3. 規則表算 use_value / secondhand_value
                        │   4. BERT 結果算 emotional_value
                        │   5. 加總、推薦、產生 reason
                        │
                        └──HTTPS──▶ [OpenAI] 寫 reason 文字
```

## 整合到團隊 repo 後的目錄結構

```
11402_AIproject/                 ← 團隊 repo 根目錄
├── backend/                     ← ★ 你（這個資料夾）
│   ├── main.py
│   ├── config.py
│   ├── requirements.txt
│   ├── .env.example
│   ├── routers/
│   │   ├── predict.py           ← /predict（前端打的）
│   │   ├── analyze.py           ← /api/analyze（純文字）
│   │   └── chat.py              ← /api/chat
│   ├── services/
│   │   ├── ai_client.py         ← httpx 打 :8001
│   │   ├── scoring.py           ← ★ 規則表（use / secondhand）
│   │   └── llm_service.py       ← OpenAI 寫 reason
│   └── schemas/item.py
│
├── templates/                   ← 前端組員
│   └── index.html
├── static/                      ← 前端組員
│   ├── style.css
│   └── script.js
│
├── core/                        ← AI 組員（BERT 模型）
│   ├── inference_engine.py
│   └── model_arch.py
├── models/                      ← BERT .pt 權重（不進 git）
└── ai_service.py                ← ★ for_teammate__ai_service.py 改名後放這
```

## 三個分數怎麼算（老師會問）

| 分數 | 來源 | 為什麼 |
|---|---|---|
| **情感價值** | BERT 推導：`intensity × emotion 信心度 + 動機加成` | BERT 本來就是設計來判斷情緒、強度、心理動機的，這裡才是它真正的舞台 |
| **使用價值** | 規則表：`頻率分 + 時間修正` | 同樣輸入永遠同樣輸出；老師問哪個數字哪來都能秒答 |
| **二手價值** | 規則表：`類別基礎分 + 折舊 + 品牌加成` | 同上，且符合常識（3C 折舊快、紀念品沒市場、有品牌加分） |
| **總分** | `0.4 × 情感 + 0.3 × 使用 + 0.3 × 二手` | 斷捨離主要是心理戰，情感權重略高 |
| **建議** | 由三個分數查表得：保留 / 出售 / 捐贈 / 丟棄 | 優先序：情感 > 實用 > 變現 > 處理 |

所有數字都在 [`services/scoring.py`](services/scoring.py) 一個檔，調整很容易。

## 一、第一次設定

### Step 0：把 backend/ 放進團隊 repo

```bash
# 在 11402_AIproject/ 根目錄
cp -r /Users/jero/Documents/Claude/Projects/AI斷捨離/backend ./
cp backend/for_teammate__ai_service.py ./ai_service.py
git add backend/ ai_service.py
git commit -m "feat: add backend + ai service wrapper"
```

### Step 1：AI 組員啟動 BERT 服務（在 11402_AIproject/ 根目錄）

```bash
pip install fastapi uvicorn pydantic        # AI 那邊已有 torch 等
uvicorn ai_service:app --host 0.0.0.0 --port 8001
```
看到 `✅ 模型已就緒` 表示三個 BERT 都載入完畢（30-60 秒）。

### Step 2：你啟動後端 + Web（在 backend/ 資料夾）

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # 編輯填入 OPENAI_API_KEY（沒 key 也能跑）
uvicorn main:app --reload --port 8000
```

啟動成功會印出：
```
📁 Frontend dir: /Users/.../11402_AIproject
   templates exists: True
   static exists:    True
🤖 AI service: http://localhost:8001
💬 LLM narrative: enabled
```

### Step 3：開瀏覽器

- **前端 UI**：<http://localhost:8000/>
- **API 互動文件**：<http://localhost:8000/docs>
- **健康檢查**：<http://localhost:8000/health>

`/health` 期望看到：
```json
{
  "backend": "healthy",
  "ai_service": "healthy",
  "frontend_files_found": true
}
```

## 二、API 規格

### `POST /predict` — 主要端點（前端 itemForm 打的）

**Request:**
```json
{
  "item_name": "高中時的相機",
  "brand": "Canon",
  "category": "electronics_3c",
  "usage_period": "over_8_years",
  "usage_frequency": "yearly",
  "objective_description": "外觀完整，按鍵有點黏，需要換電池",
  "emotional_description": "畢業旅行用的，捨不得丟"
}
```

**Response:**
```json
{
  "item_name": "高中時的相機",
  "recommendation": "建議保留",
  "total_score": 68,
  "use_value": 5,
  "emotional_value": 88,
  "secondhand_value": 50,
  "reason": "從你的描述能感覺到這台相機承載著畢業旅行的回憶..."
}
```

前端 [`script.js`](../static/script.js) 已經寫好 `renderResult(data)` 讀這個格式。

### `POST /api/analyze` — 純文字版（給開發測試）

```json
// Request
{"text": "我有一個舊背包，大學買的", "include_narrative": true}

// Response
{
  "raw": {"final_score": 65, "decision": "保留 (Keep)", "details": {...}},
  "decision_simple": "keep",
  "narrative": "..."
}
```

### `POST /api/chat` — 後續對話（純 LLM）

```json
{
  "session_id": "uuid",
  "messages": [{"role": "user", "content": "我還是猶豫..."}],
  "last_analysis": null
}
```

## 三、沒有 OpenAI key 也能跑

把 `.env` 設成：
```
ENABLE_LLM_NARRATIVE=false
# OPENAI_API_KEY 留空也行
```

`reason` 會用 [llm_service.py 裡的 `_fallback_reason`](services/llm_service.py) 模板：
> 「從你的描述中可以感覺到「X」對你有很深的意義...」

四種 recommendation 各有一段模板，前端不會看到空白。

## 四、改規則 / 調整分數

所有可調整的數字在 [`services/scoring.py`](services/scoring.py)：

- `_FREQUENCY_SCORE` — 每個使用頻率的基礎分
- `_PERIOD_BONUS` — 使用時間修正
- `_CATEGORY_BASE` — 9 個類別的二手基礎分
- `_PERIOD_DEPRECIATION` — 二手折舊
- `_FREQUENCY_CONDITION_MOD` — 條件修正（全新 vs. 天天用）
- 權重在 `compute_total_score`：`0.4 × 情感 + 0.3 × 使用 + 0.3 × 二手`

改完直接重跑，不用 retrain。

## 五、Trouble Shooting

| 症狀 | 可能原因 | 解法 |
|---|---|---|
| `/health` 顯示 `ai_service: unreachable` | AI service 沒跑 / 在別的 port | 確認 `ai_service.py` 在跑、`.env` 的 `AI_SERVICE_URL` 對 |
| 首頁顯示「找不到 index.html」 | backend/ 沒放在 repo 根目錄 | 設 `FRONTEND_DIR` 環境變數指到 templates/ 跟 static/ 的父目錄 |
| 前端按「開始評估」沒反應 | 開瀏覽器 Console 看 fetch 是不是 CORS 錯誤 | 多半是 frontend 用了 file:// 開啟；改用 <http://localhost:8000/> |
| `502 AI service 連線失敗` | 第一次啟動 BERT 還在載入 | 等 30-60 秒 |
| `reason` 一直是模板字 | 沒填 OPENAI_API_KEY 或 `ENABLE_LLM_NARRATIVE=false` | 補上 key |

## 六、給組員的整合 checklist

- [ ] 把 `backend/` 整個資料夾複製到 `11402_AIproject/` 根目錄
- [ ] 把 `backend/for_teammate__ai_service.py` 複製到 `11402_AIproject/ai_service.py`
- [ ] AI 組員把 `.pt` 權重檔放到 `11402_AIproject/models/`
- [ ] 約好 dev 環境都用：AI service `:8001`、Web `:8000`
- [ ] `backend/.env` 不要 commit（已在 .gitignore）
