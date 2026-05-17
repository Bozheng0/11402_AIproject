## AI Fuzzy Inference API 

---

### 1. AI Predicttion API

```http
POST /predict
```

---

#### 1.1 Request Body

```json
{
  "secondhand_input": {
    "name": "Nike Shoes",
    "item_condition_id": 1,
    "category_name": "Shoes",
    "brand_name": "Nike",
    "shipping": 1,
    "item_description": "Brand new"
  },
  "text_input": "I like it but rarely use it"
}
```

#### Request 欄位說明

##### secondhand_input

| 欄位 | 型別 | 說明 |
|---|---|---|
| name | string | 商品名稱 |
| item_condition_id | integer | 商品狀態 |
| category_name | string | 商品分類 |
| brand_name | string | 品牌名稱 |
| shipping | integer | 是否含運 |
| item_description | string | 商品描述 |


#### text_input

| 欄位 | 型別 | 說明 |
|---|---|---|
| text_input | string | 使用者對物品的情感描述 |

---

#### 1.2 Response Body

```json
{
  "secondhand": 29.33,
  "sentiment": 67.69,
  "usevalue": 4,
  "final_decision": "KEEP"
}
```

#### Response 欄位說明

| 欄位 | 型別 | 說明 |
|---|---|---|
| secondhand | float | 二手價格預測 |
| sentiment | float | 情感價值分數 |
| usevalue | integer | 使用價值分類 |
| final_decision | string | 模糊推論最終決策 |

---

### 2. AI Explanation API

```http
POST /explain
```

將推論結果轉化為自然語言解釋。

---

#### 2.1 Request Body

```json
{
  "input_data": {
    "secondhand_input": {
      "name": "Nike Shoes",
      "item_condition_id": 3,
      "category_name": "shoes",
      "brand_name": "Nike",
      "shipping": 1,
      "item_description": "Lightly used running shoes"
    },
    "text_input": "I don't use it much but it still feels valuable to me"
  },
  "predict_result": {
    "secondhand": 85.2,
    "sentiment": 72.5,
    "usevalue": 1,
    "final_decision": "Keep"
  }
}
```

---

#### 2.2 Response Body

```json
{
    "reason": "Hello there! Let's take a closer look at your Nike Shoes and why the system recommends keeping them.\n\nYou mentioned that even though you don't use them much, they still feel valuable to you, and our analysis picked up on that strong emotional connection you have with them.\n\nHere's the breakdown of what the model considered:\n\n*   **Sentimental Value (72.5):** This is a really significant score, clearly aligning with your feeling that the shoes are valuable to you. Your personal connection and the positive feelings associated with them are a major factor in our recommendation. We understand that items can hold value far beyond just their practical use, and your emotional attachment is very important.\n*   **Secondhand Value (85.2):** The shoes are described as \"lightly used,\" and Nike is a popular brand, so they have excellent potential if you were to consider selling them in the future. This high resale value means that even if you don't wear them often, they represent a tangible asset.\n*   **Usage Value (1):** As you noted, you don't use them much, which is reflected in this low score. From a purely practical, day-to-day usage perspective, they aren't seeing much action right now.\n\nConsidering all these aspects, the system's final decision is to **Keep** them.\n\nEven though their current usage is low, the strong emotional value you place on them, combined with their excellent potential resale value, makes them worth holding onto for now. It seems that your personal connection, which directly aligns with the high sentimental value, outweighs the current lack of practical use. The system understands that sometimes an item's emotional weight is more significant than its current functionality.\n\nThink of it this way: you have a valuable item that holds personal meaning for you. While selling them is an option due to their high secondhand value, the system respects your current emotional attachment. You can always revisit this decision later if your feelings change or if you find yourself needing the space more, but for now, it's perfectly valid to keep something that brings you a sense of value and connection.\n\nWe're here to help you make decisions that feel right to *you*!"
}
```


