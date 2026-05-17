import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import json
from unified_predict import UnifiedPredictor

class FuzzyInferenceEngine:
    def __init__(self):
        self._setup_system()

    def _setup_system(self):
        
        # 輸入變數
        self.usage = ctrl.Antecedent(np.arange(0, 101, 1), 'usage')
        self.sentimental = ctrl.Antecedent(np.arange(0, 101, 1), 'sentimental')
        self.secondhand = ctrl.Antecedent(np.arange(0, 301, 1), 'secondhand')

        # 輸出變數
        self.output = ctrl.Consequent(np.arange(0, 101, 1), 'output')

        
        # 使用價值
        self.usage['low'] = fuzz.trimf(self.usage.universe, [0, 0, 40])
        self.usage['medium'] = fuzz.trimf(self.usage.universe, [30, 50, 70])
        self.usage['high'] = fuzz.trimf(self.usage.universe, [60, 100, 100])

        # 情感價值
        self.sentimental['low'] = fuzz.trimf(self.sentimental.universe, [0, 0, 40])
        self.sentimental['medium'] = fuzz.trimf(self.sentimental.universe, [30, 50, 70])
        self.sentimental['high'] = fuzz.trimf(self.sentimental.universe, [60, 100, 100])

        # 二手價值
        self.secondhand['low'] = fuzz.trimf(self.secondhand.universe, [0, 0, 50])
        self.secondhand['medium'] = fuzz.trimf(self.secondhand.universe, [30, 90, 150])
        self.secondhand['high'] = fuzz.trimf(self.secondhand.universe, [120, 300, 300])

        # 輸出決策
        self.output['discard'] = fuzz.trimf(self.output.universe, [0, 0, 25])
        self.output['donate'] = fuzz.trimf(self.output.universe, [20, 35, 50])
        self.output['sell'] = fuzz.trimf(self.output.universe, [45, 60, 75])
        self.output['keep'] = fuzz.trimf(self.output.universe, [70, 100, 100])

        # 模糊規則
        rules = [
            ctrl.Rule(self.usage['high'] & self.sentimental['high'], self.output['keep']), # Rule 1
            ctrl.Rule(self.usage['high'] & self.secondhand['low'], self.output['keep']),   # Rule 2
            ctrl.Rule(self.usage['low'] & self.secondhand['high'] & self.sentimental['low'], self.output['sell']), # Rule 3
            ctrl.Rule(self.usage['medium'] & self.secondhand['high'], self.output['sell']), # Rule 4
            ctrl.Rule(self.usage['low'] & self.secondhand['low'] & self.sentimental['medium'], self.output['donate']), # Rule 5
            ctrl.Rule(self.usage['medium'] & self.secondhand['low'], self.output['donate']), # Rule 6
            ctrl.Rule(self.usage['low'] & self.secondhand['low'] & self.sentimental['low'], self.output['discard']), # Rule 7
            ctrl.Rule(self.usage['low'] & self.sentimental['low'], self.output['discard']), # Rule 8
            ctrl.Rule(self.sentimental['high'] & self.secondhand['high'], self.output['keep']), # Rule 9
            ctrl.Rule(self.usage['medium'] & self.sentimental['high'], self.output['keep']) # Rule 10
        ]

        # 控制系統
        self.system_ctrl = ctrl.ControlSystem(rules)
        self.system = ctrl.ControlSystemSimulation(self.system_ctrl)

    def infer(self, usage_score, sentimental_score, secondhand_score):
        secondhand_score = min(max(secondhand_score, 0), 300)
        
        self.system.input['usage'] = usage_score
        self.system.input['sentimental'] = sentimental_score
        self.system.input['secondhand'] = secondhand_score

        try:
            self.system.compute()
            result_score = self.system.output['output']
        except Exception as e:
            # 如果推論失敗（例如輸入不在 universe 內），返回預設值
            print(f"Fuzzy Inference Error: {e}")
            return {"score": 0, "decision": "ERROR", "error": str(e)}

        if result_score < 25:
            decision = 'DISCARD'
        elif result_score < 50:
            decision = 'DONATE'
        elif result_score < 75:
            decision = 'SELL'
        else:
            decision = 'KEEP'

        return {
            'score': round(result_score, 2),
            'decision': decision
        }

def run_full_inference(sh_input, text_input):

    predictor = UnifiedPredictor()
    engine = FuzzyInferenceEngine()

    print("Running individual models...")

    sh_res = predictor.predict_secondhand(sh_input)
    sent_res = predictor.predict_sentiment(text_input)
    use_res = predictor.predict_usevalue(text_input)

    if "error" in sh_res or "error" in sent_res or "error" in use_res:
        return {
            "error": "One or more models failed",
            "details": {
                "secondhand": sh_res,
                "sentiment": sent_res,
                "usevalue": use_res
            }
        }

    usage_score = use_res['soft_score'] * 20
    sentimental_score = sent_res['final_score']
    secondhand_score = sh_res['price']

    print(f"Fuzzy Inputs: Usage={usage_score:.2f}, Sentimental={sentimental_score:.2f}, Secondhand={secondhand_score:.2f}")

    final_result = engine.infer(usage_score, sentimental_score, secondhand_score)
    
    return {
        "individual_results": {
            "secondhand": sh_res,
            "sentiment": sent_res,
            "usevalue": use_res
        },
        "fuzzy_inputs": {
            "usage_score": usage_score,
            "sentimental_score": sentimental_score,
            "secondhand_score": secondhand_score
        },
        "final_decision": final_result
    }

if __name__ == "__main__":
    # sample input
    sh_input = {
        "name": "Nike Air Max 97",
        "item_condition_id": 1,
        "category_name": "Men/Shoes/Athletic",
        "brand_name": "Nike",
        "shipping": 1,
        "item_description": "Brand new in box, never worn. Very rare colorway."
    }
    
    text_input = "These shoes are one of my favorite collections. Although I rarely wear them, just looking at them makes me feel that they are very valuable."
    
    print("=== Full Item Decision System ===")
    result = run_full_inference(sh_input, text_input)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        print(json.dumps(result["details"], indent=2, ensure_ascii=False))
    else:
        print("\n=== Final Decision ===")
        print(f"Decision: {result['final_decision']['decision']}")
        print(f"Score: {result['final_decision']['score']}")
        print("\nDetailed breakdown:")
        print(f"- Market Price: ${result['individual_results']['secondhand']['price']}")
        print(f"- Sentiment Score: {result['individual_results']['sentiment']['final_score']}")
        print(f"- Usage Score (Mapped): {result['fuzzy_inputs']['usage_score']:.2f}")
