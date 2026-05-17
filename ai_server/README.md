# AI 斷捨離輔助決策系統 - 後端伺服器 (AI Server)

## 系統架構

- **FastAPI**: 提供 RESTful API 接口。
- **Unified Predictor**: 統一調度器，負責調用三個子模型。
- **子模型**:
  - `secondhand`: 預測物品的二手市場行情。
  - `sentiment`: 分析使用者對物品的情感連結。
  - `usevalue`: 評估物品的實際使用價值。
- **Fuzzy Inference**: 綜合三個分數，透過模糊邏輯產出最終決策。
- **Gemini API**: 根據預測結果與輸入資訊，生成解釋與建議。

---

## 啟動方式

### 1. 環境準備

#### 設定環境變數
- 在 `ai_server` 目錄下建立 `.env` 檔案，並填入您的 Gemini API Key。

### 2. 子模型環境設定

- `UnifiedPredictor` 會透過子進程調用各模型的環境。

### 3. 模型權重放置

- 確保各模型的權重檔案已放置於指定位置

### 4. 啟動伺服器

回到 `ai_server` 根目錄，並確保主虛擬環境已啟動：
```bash
python app.py
```
伺服器預設將運行在 `http://localhost:8000`。

---

## API 說明

#### 1. 測試
- **GET** `/health`
- **說明**: 確認 API 是否正常運行。

#### 2. 斷捨離預測
- **POST** `/predict`
- **Payload**:
  ```json
  {
    "secondhand_input": {
      "name": "物品名稱",
      "item_condition_id": 1,
      "category_name": "分類/路徑",
      "brand_name": "品牌",
      "shipping": 1,
      "item_description": "物品細節描述"
    },
    "text_input": "使用者對物品的心情描述"
  }
  ```
- **Response**: 返回各項分數及 `final_decision`。

#### 3. 生成 AI 解釋
- **POST** `/explain`
- **Payload**: 包含原始輸入與預測結果。
- **Response**: 返回由 Gemini 生成的建議文字。
