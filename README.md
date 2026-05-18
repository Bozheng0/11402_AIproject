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
└── templates/          # 前端 HTML 模板 (index.html)
```

---

## 🛠️ 快速啟動

### 1. 環境準備

確保你的環境已安裝 Python 3.9+。

### 2. 設定 AI Server

1. 進入 `ai_server` 目錄。
2. 建立 `.env` 檔案並填入 `GEMINI_API_KEY`。
3. 安裝依賴並啟動：
   ```bash
   cd ai_server
   pip install -r requirements.txt
   python app.py
   ```

### 3. 設定 Backend Adapter (主要存取點)

1. 進入 `backend` 目錄。
2. 建立 `.env` 檔案（可參考 `.env.example`）。
3. 安裝依賴並啟動：
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --port 8080 --reload
   ```

### 4. 開始使用

打開瀏覽器訪問 `http://localhost:8080` 即可看到 Re:space 首頁。



