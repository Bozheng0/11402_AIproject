# Fuzzy Inference System

本文件說明 `fuzzy_inference.py` 中的模糊推論邏輯。該系統旨在整合多個子模型的預測結果，對物品的處置方式做出最終決策。

## 1. 系統概述

模糊推論系統接收三個主要維度的評分作為輸入，並輸出一個介於 0 到 100 之間的決策分數，最後將該分數映射到四種處置建議之一：
- **丟棄 (DISCARD)**
- **捐贈 (DONATE)**
- **拍賣 (SELL)**
- **保留 (KEEP)**

## 2. 輸入變數

系統定義了三個輸入變數：

### 2.1 使用價值 (Usage Value, `usage`)
*   **範圍**: 0 - 100
*   **來源**: 由 `usevalue` 模型預測的 `soft_score` (1-5) 映射而來（`soft_score * 20`）。
*   **模糊集合 (Membership Functions)**:
    *   `low`: `trimf(0, 0, 40)` - 代表使用頻率低或幾乎不使用。
    *   `medium`: `trimf(30, 50, 70)` - 代表偶爾使用。
    *   `high`: `trimf(60, 100, 100)` - 代表經常使用或必需品。

### 2.2 情感價值 (Sentimental Value, `sentimental`)
*   **範圍**: 0 - 100
*   **來源**: 由 `sentiment` 模型的 `final_score` 直接提供。
*   **模糊集合**:
    *   `low`: `trimf(0, 0, 40)` - 代表對該物品沒有特殊情感連結。
    *   `medium`: `trimf(30, 50, 70)` - 代表具有一般紀念價值。
    *   `high`: `trimf(60, 100, 100)` - 代表具有極高情感意義或不可替代性。

### 2.3 二手價值 (Secondhand Value, `secondhand`)
*   **範圍**: 0 - 300 (單位：USD)
*   **來源**: 由 `secondhand` 模型預測的市場價格。
*   **模糊集合**:
    *   `low`: `trimf(0, 0, 50)` - 低價值物品。
    *   `medium`: `trimf(30, 90, 150)` - 中等價值物品。
    *   `high`: `trimf(120, 300, 300)` - 高價值或奢侈品。

---

## 3. 輸出變數

### 3.1 決策分數 (`output`)
*   **範圍**: 0 - 100
*   **模糊集合**:
    *   `discard` (丟棄): `trimf(0, 0, 25)`
    *   `donate` (捐贈): `trimf(20, 35, 50)`
    *   `sell` (拍賣): `trimf(45, 60, 75)`
    *   `keep` (保留): `trimf(70, 100, 100)`

---

## 4. 模糊規則

系統目前定義了 10 條規則來引導決策過程：

1.  **IF** `usage` is **high** **AND** `sentimental` is **high** **THEN** `output` is **keep**.
2.  **IF** `usage` is **high** **AND** `secondhand` is **low** **THEN** `output` is **keep**. (即便沒價值，但常用則留)
3.  **IF** `usage` is **low** **AND** `secondhand` is **high** **AND** `sentimental` is **low** **THEN** `output` is **sell**. (不常用、沒感情但值錢)
4.  **IF** `usage` is **medium** **AND** `secondhand` is **high** **THEN** `output` is **sell**.
5.  **IF** `usage` is **low** **AND** `secondhand` is **low** **AND** `sentimental` is **medium** **THEN** `output` is **donate**.
6.  **IF** `usage` is **medium** **AND** `secondhand` is **low** **THEN** `output` is **donate**. (不值錢但還可以用的東西)
7.  **IF** `usage` is **low** **AND** `secondhand` is **low** **AND** `sentimental` is **low** **THEN** `output` is **discard**.
8.  **IF** `usage` is **low** **AND** `sentimental` is **low** **THEN** `output` is **discard**.
9.  **IF** `sentimental` is **high** **AND** `secondhand` is **high** **THEN** `output` is **keep**. (高感性且高價值，通常不建議賣)
10. **IF** `usage` is **medium** **AND** `sentimental` is **high** **THEN** `output` is **keep**.

---

## 5. 推論與映射

### 5.1 推論過程
系統使用重心法 (Centroid) 進行去模糊化 (Defuzzification)，計算出最終的 `result_score`。

### 5.2 最終決策映射
根據得到的分數，系統進行以下分類：

| 分數區間 | 處置建議 (Decision) |
| :--- | :--- |
| 0 <= 分數 < 25 | **DISCARD** (丟棄) |
| 25 <= 分數 < 50 | **DONATE** (捐贈) |
| 50 <= 分數 < 75 | **SELL** (拍賣) |
| 75 <= 分數 <= 100 | **KEEP** (保留) |

---

## 6. 整合流程

`run_full_inference` 函數負責協調整個流程：
1.  **模型調度**: 同時運行 `UnifiedPredictor` 下的三個模型。
2.  **數據預處理**:
    *   將 `usevalue` 的 1-5 分映射至 0-100。
    *   將市場價格限制在 0-300 的範圍內供模糊系統處理。
3.  **模糊運算**: 將處理後的數據輸入模糊引擎。
4.  **結果封裝**: 返回包含各子模型結果、模糊輸入值以及最終決策的完整 JSON 物件。
