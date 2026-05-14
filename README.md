# AI 斷捨離心理決策系統

本專案開發一套結合 NLP 情感分析與深度心理學模型的輔助決策系統，透過分析使用者對物品的文字描述，從情緒、強度及心理動機三個維度提供斷捨離建議。

## 技術架構與資料集
系統採用多任務學習 (Multi-task Learning) 框架，整合多個專業領域的訓練資料：

* **情緒識別 (Emotion)**：使用 **GoEmotions** 資料集 (28種情緒)，理解使用者對物品的情感類別。
* **情感強度 (Intensity)**：使用 **SemEval-2018** 資料集，將情感連結量化為數值 (0.0 - 1.5)。
* **心理動機 (Motivation)**：結合 **StoryCommons** 資料集，解析馬斯洛需求與瑞斯 (Reiss) 16 種基本動機。
* **領域微調 (Domain Adaptation)**：使用自建資料集進行最後的語境校準。

## 核心決策評估演算法
系統總分為 0 至 100 分，計算公式如下：
$$Score = Score_{base} + \Delta Score_{emotion} + \Delta Score_{motivation}$$

### 參數詳細說明：

* **基礎分 ($Score_{base}$)**：`50.0`
    * 系統預設的中立基準點。

* **情緒修正 ($\Delta Score_{emotion}$)**：
    * **正向增益**：`+(30 * Intensity * Confidence_emo)`
    * **負向抑制**：`-(35 * Intensity * Confidence_emo)`
    * *註：Intensity 為情感強度，Confidence 為模型分類信心度。*

* **動機修正 ($\Delta Score_{motivation}$)**：
    * **傳承/社交補償 (Legacy/Social)**：`+(35 * Confidence_maslow)`
    * **成就/榮譽加成 (Status/Honor)**：`+(20 * Confidence_reiss)`
    * **囤積行為抑制 (Stability)**：`-(5 * Confidence_maslow)`

* **特殊補償機制（悲傷處理）**：
    * 若判定為**紀念性質 (Legacy)**：`+(25 * Intensity)`
    * 若判定為**純屬痛苦回憶**：`-(20 * Intensity)`

## 資料夾結構
* `app.py`: 系統進入點。
* `core/`: 存放模型架構 (`model_arch.py`) 與推論引擎 (`inference_engine.py`)。
* `scripts/`: 包含基礎訓練與領域適應微調腳本。
* `models/`: 存放 `.pt` 權重檔（需手動放入）。
* `data/`: 存放 CSV 訓練集與 JSONL 微調資料。

## 快速啟動
1. `pip install -r requirements.txt`
2. 將 `emotion_model.pt`、`intensity_model.pt`、`motivation_model.pt` 放入 `models/`
3. `python3 app.py`

## 技術規格
* **Model**: BERT-base-multilingual-cased
* **Environment**: WSL2 (Ubuntu 22.04), RTX 5060

