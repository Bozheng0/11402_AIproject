import os
import subprocess
import json
import tempfile
import csv
import sys

class UnifiedPredictor:
    def __init__(self, root_dir=None):
        if root_dir is None:
            root_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = root_dir
        
        self.configs = {
            "secondhand": {
                "venv": os.path.join(root_dir, "secondhand", "mercari_ext", "venv", "Scripts", "python.exe"),
                "script": os.path.join(root_dir, "secondhand", "mercari_ext", "predict_service.py"),
                "cwd": os.path.join(root_dir, "secondhand", "mercari_ext")
            },
            "sentiment": {
                "venv": os.path.join(root_dir, "sentiment", "venv", "Scripts", "python.exe"),
                "cwd": os.path.join(root_dir, "sentiment")
            },
            "usevalue": {
                "venv": os.path.join(root_dir, "usevalue", "venv", "Scripts", "python.exe"),
                "cwd": os.path.join(root_dir, "usevalue")
            }
        }

    def predict_secondhand(self, item_data):
        """
        item_data: dict with keys: name, item_condition_id, category_name, brand_name, shipping, item_description
        """
        config = self.configs["secondhand"]
        if not os.path.exists(config["venv"]):
            return {"error": f"Virtual environment not found at {config['venv']}"}

        # Create a temporary TSV file for input
        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False, mode='w', encoding='utf-8', newline='') as f:
            # Ensure all required columns are present
            required_cols = ["name", "item_condition_id", "category_name", "brand_name", "shipping", "item_description"]
            data = {col: item_data.get(col, "") for col in required_cols}
            
            writer = csv.DictWriter(f, fieldnames=required_cols, delimiter='\t')
            writer.writeheader()
            writer.writerow(data)
            temp_input = f.name
            
        temp_output = temp_input + ".out.csv"
        
        try:
            cmd = [config["venv"], config["script"], "--input", temp_input, "--output", temp_output, "--rounds", "1"]
            result = subprocess.run(cmd, cwd=config["cwd"], capture_output=True, text=True, encoding='utf-8')
            
            if os.path.exists(temp_output):
                with open(temp_output, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        return {"price": float(rows[0]["price"])}
            
            return {"error": "Prediction failed", "stdout": result.stdout, "stderr": result.stderr}
        finally:
            if os.path.exists(temp_input):
                try: os.remove(temp_input)
                except: pass
            if os.path.exists(temp_output):
                try: os.remove(temp_output)
                except: pass

    def predict_sentiment(self, text):
        config = self.configs["sentiment"]
        if not os.path.exists(config["venv"]):
            return {"error": f"Virtual environment not found at {config['venv']}"}

        # Bridge code to run inside the sentiment environment
        code = f"""
import json
import sys
import os
# Add current directory to path so 'core' can be found
sys.path.append(os.getcwd())
from core.inference_engine import FinalEvaluator
try:
    ev = FinalEvaluator()
    res = ev.evaluate({repr(text)})
    print("JSON_START" + json.dumps(res) + "JSON_END")
except Exception as e:
    print("ERROR_START" + str(e) + "ERROR_END")
"""
        result = subprocess.run([config["venv"], "-c", code], cwd=config["cwd"], capture_output=True, text=True, encoding='utf-8')
        
        stdout = result.stdout
        if "JSON_START" in stdout and "JSON_END" in stdout:
            json_str = stdout.split("JSON_START")[1].split("JSON_END")[0]
            return json.loads(json_str)
        elif "ERROR_START" in stdout and "ERROR_END" in stdout:
            err_str = stdout.split("ERROR_START")[1].split("ERROR_END")[0]
            return {"error": err_str}
        
        return {"error": "Unexpected output format", "stdout": stdout, "stderr": result.stderr}

    def predict_usevalue(self, text):
        config = self.configs["usevalue"]
        if not os.path.exists(config["venv"]):
            return {"error": f"Virtual environment not found at {config['venv']}"}

        # Bridge code to run inside the usevalue environment
        code = f"""
import json
import sys
import os
import numpy as np
sys.path.append(os.getcwd())
from infer import predict
try:
    res = predict({repr(text)})
    # Convert numpy types to native types for JSON serialization
    if 'probs' in res:
        res['probs'] = res['probs'].tolist()
    if 'soft_score' in res:
        res['soft_score'] = float(res['soft_score'])
    if 'pred_class' in res:
        res['pred_class'] = int(res['pred_class'])
    print("JSON_START" + json.dumps(res) + "JSON_END")
except Exception as e:
    print("ERROR_START" + str(e) + "ERROR_END")
"""
        result = subprocess.run([config["venv"], "-c", code], cwd=config["cwd"], capture_output=True, text=True, encoding='utf-8')
        
        stdout = result.stdout
        if "JSON_START" in stdout and "JSON_END" in stdout:
            json_str = stdout.split("JSON_START")[1].split("JSON_END")[0]
            return json.loads(json_str)
        elif "ERROR_START" in stdout and "ERROR_END" in stdout:
            err_str = stdout.split("ERROR_START")[1].split("ERROR_END")[0]
            return {"error": err_str}
                
        return {"error": "Unexpected output format", "stdout": stdout, "stderr": result.stderr}

    def predict_all(self, secondhand_input, text_input):
        """
        Executes all three models and returns a combined result.
        """
        return {
            "secondhand": self.predict_secondhand(secondhand_input),
            "sentiment": self.predict_sentiment(text_input),
            "usevalue": self.predict_usevalue(text_input)
        }
        
    def predict(self, secondhand_input, text_input):

        secondhand = self.predict_secondhand(secondhand_input)
        sentiment = self.predict_sentiment(text_input)
        usevalue = self.predict_usevalue(text_input)

        return {
            "price": round(secondhand["price"], 2),
            "final_score": sentiment["final_score"],
            "pred_class": usevalue["pred_class"]
        }    

def main():
    predictor = UnifiedPredictor()
    
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
    
    print("=== Unified Model Inference ===")
    print(f"Text Input: {text_input}")
    print("Processing...")
    
    results = predictor.predict(sh_input, text_input)
    # results = predictor.predict_all(sh_input, text_input)
    
    print("\n=== Results ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
