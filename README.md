# Re:space - AI 斷捨離輔助決策系統

> **讓每個物品，都找到更適合的去處。**

Re:space 是一個基於 AI 與模糊邏輯 (Fuzzy Logic) 的斷捨離決策輔助系統。透過評估物品的「使用價值」、「情感連結」以及「二手市場潛力」，為使用者提供溫柔且理性的物品整理建議。

<p align="center">
  <img src="./static/Respace.PNG" alt="Demo Preview" width="800">
</p>

---

## 🚀 核心功能

- **四步驟物品建檔**：直觀的 Wizard 介面，從類別、基本資料到使用感觸與情感描述。
- **多維度 AI 評估**：
    - **二手行情預測**：基於 Mercari 數據集的價格預估模型。
    - **情感價值分析**：分析使用者描述，量化與物品之間的情感紐帶。
    - **使用價值分類**：根據使用頻率與年限，判斷物品目前的實用程度。
- **模糊推論決策**：不只是給分數，透過模糊邏輯系統產出最終建議（保留、出售、捐贈、丟棄）。
- **個人化 AI 解釋**：整合 Google Gemini API，根據分析數據生成一段暖心且具說服力的分析報告。

---

## 🏗️ 系統架構

專案採用三層架構：

1.  **Frontend (Vanilla Web)**: 提供響應式使用者介面，引導使用者完成評估流程。
2.  **Backend (FastAPI Adapter)**: 擔任中介與翻譯角色，處理前端與 AI Server 的欄位轉換、Gemini 解釋生成與靜態檔案服務。
3.  **AI Server (Core Engine)**: 專門負責運行各項 AI 模型（價格預測、情緒分析、分類）以及模糊推論系統。

---

## 📂 專案結構

```text
11402_AIproject/
├── ai_server/          # AI 推論核心服務 
│   ├── secondhand/     # 二手價格預測模型相關
│   ├── sentiment/      # 情感分析模型相關
│   ├── usevalue/       # 使用價值分類模型相關
│   ├── app.py          # AI Server 入口點
│   └── fuzzy_inference.py # 模糊推論系統
├── backend/            # 後端轉接層 
│   ├── main.py         # 後端入口點（服務靜態檔案與 API）
│   ├── routers/        # API 路由邏輯
│   └── services/       # 翻譯層與評分邏輯
├── static/             # 前端靜態資源 (CSS, JS, Images)
├── templates/          # 前端 HTML 模板 (index.html)
└── docker-compose.yml  # Docker 容器編排設定
```

---

## 🐳 快速部署
**條件：** 安裝 [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### 1. 準備必要檔案

請建立專案資料夾，並準備以下檔案：

#### A. `.env` 
在相同目錄建立 `.env`：
```bash
GROQ_API_KEY=您的_GROQ_API_KEY
```

#### B. 模型檔案 (Models)
確保您已將以下模型資料夾複製到與 `docker-compose.yml` 相同的相對路徑下：
- `ai_server/secondhand/mercari_ext/models_exported/`
- `ai_server/sentiment/models/`
- `ai_server/usevalue/models/`

### 2. 啟動服務

在該目錄下執行：
```bash
docker compose up 
```
Mac 使用者注意事項

如果是在 Mac，尤其是 Apple Silicon Mac，可能會遇到 TensorFlow 版本相容性問題，導致二手價格預測模型出現 Illegal instruction 錯誤。

請先在專案目錄下啟動服務：

docker compose up

接著另外開一個新的 Terminal，進入同一個專案目錄，執行以下指令重新安裝 TensorFlow：

docker compose exec ai_server bash -lc '
/app/ai_server/secondhand/mercari_ext/venv/bin/pip uninstall -y tensorflow tensorflow-cpu tensorflow-gpu tensorflow-estimator tensorflow-cpu-estimator tensorboard
/app/ai_server/secondhand/mercari_ext/venv/bin/pip install --no-cache-dir tensorflow==1.5.0
'

安裝完成後，執行以下指令確認 TensorFlow 是否可以正常載入：

docker compose exec ai_server bash -lc '
/app/ai_server/secondhand/mercari_ext/venv/bin/python -c "import tensorflow as tf; print(tf.__version__)"
echo RETURN_CODE:$?
'

如果輸出類似以下內容，代表 TensorFlow 可以正常使用：

1.5.0
RETURN_CODE:0

如果出現以下內容，代表目前環境仍然不相容：

Illegal instruction
RETURN_CODE:132

確認 TensorFlow 可以正常載入後，重啟服務：

docker compose restart

接著即可重新開啟：

http://localhost:8000/docs

測試 API 是否正常運作。
### 3. 訪問系統
- **前端主介面**: `http://localhost:8080/`
