from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import os
from google import genai
from dotenv import load_dotenv
from fuzzy_inference import FuzzyInferenceEngine
from unified_predict import UnifiedPredictor

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None

app = FastAPI(
    title="AI Fuzzy Inference API",
    description="API for predicting item value and making disposal decisions using fuzzy logic.",
    version="1.0.0"
)

predictor = UnifiedPredictor()
fuzzy_engine = FuzzyInferenceEngine()

class SecondhandInput(BaseModel):
    name: str
    item_condition_id: int
    category_name: str
    brand_name: Optional[str] = ""
    shipping: int
    item_description: Optional[str] = ""

class PredictRequest(BaseModel):
    secondhand_input: SecondhandInput
    text_input: str

class PredictResult(BaseModel):
    secondhand: float
    sentiment: float
    usevalue: int
    final_decision: str

class ExplainRequest(BaseModel):
    input_data: PredictRequest
    predict_result: PredictResult

@app.get("/health")
async def root():
    return {
        "status": "online",
        "message": "AI Fuzzy Inference API is running"
    }

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        sh_input = request.secondhand_input.model_dump()
        text_input = request.text_input
        
        sh_res = predictor.predict_secondhand(sh_input)
        sent_res = predictor.predict_sentiment(text_input)
        use_res = predictor.predict_usevalue(text_input)

        errors = {}
        if "error" in sh_res: errors["secondhand"] = sh_res["error"]
        if "error" in sent_res: errors["sentiment"] = sent_res["error"]
        if "error" in use_res: errors["usevalue"] = use_res["error"]
        
        if errors:
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": "One or more models failed",
                    "details": {
                        "secondhand": sh_res,
                        "sentiment": sent_res,
                        "usevalue": use_res
                    }
                }
            )

        usage_score = use_res.get('soft_score', 0) * 20
        sentimental_score = sent_res.get('final_score', 0)
        secondhand_score = sh_res.get('price', 0)
        final_result = fuzzy_engine.infer(usage_score, sentimental_score, secondhand_score)
        
        return {
            "secondhand": round(secondhand_score, 2),
            "sentiment": round(sentimental_score, 2),
            "usevalue": int(use_res.get('pred_class', 0)),
            "final_decision": final_result['decision']
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Unhandled Exception: {error_trace}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
async def explain(request: ExplainRequest):
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
    
    try:
        input_data = request.input_data
        predict_res = request.predict_result
        
        prompt = f"""
你是一個「斷捨離 AI 助理」，請用**繁體中文**寫一段完整分析文章。

你的任務是解釋為什麼系統會建議「保留 / 出售 / 捐贈 / 丟棄」這個物品。

請注意：
- 一定要用「繁體中文」
- 一定要用「文章形式」，不要條列
- 不要使用 bullet points
- 不要使用英文回答
- 語氣要溫和、有同理心、像 AI 顧問

請依照以下資訊撰寫：

物品名稱：
{input_data.secondhand_input.name}

物品描述：
{input_data.secondhand_input.item_description}

使用者補充描述：
{input_data.text_input}

AI 模型分析結果：
- 二手價值：{predict_res.secondhand}
- 情感價值：{predict_res.sentiment}
- 使用價值：{predict_res.usevalue}
- 最終決策：{predict_res.final_decision}

---

請輸出一段完整中文分析文章，內容需包含：
1. 對這個決策的總體解釋
2. 使用價值的分析
3. 情感價值的分析
4. 二手價值的考量
5. 最後溫和收尾建議
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        return {
            "reason": response.text.strip()
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Explain Error: {error_trace}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
