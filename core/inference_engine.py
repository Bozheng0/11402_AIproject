import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from .model_arch import DanshariClassifier, IntensityRegressor, MultiTaskMotivationClassifier

class FinalEvaluator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "bert-base-multilingual-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.go_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
        self.m_labels = ['social', 'doing', 'legacy', 'stability', 'neutral']
        self.reiss_labels = ['status', 'approval', 'independence', 'order', 'social_contact', 'honor', 'idealism', 'vengeance', 'romance', 'family', 'food', 'physical_exercise', 'saving', 'curiosity', 'tranquility', 'neutral']

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, "..", "models")

        emo_path = os.path.join(model_dir, "emotion_model.pt")
        int_path = os.path.join(model_dir, "intensity_model.pt")
        stc_path = os.path.join(model_dir, "motivation_model.pt")

        try:
            self.model_emo = DanshariClassifier(self.model_name).to(self.device)
            self.model_emo.load_state_dict(torch.load(emo_path, map_location=self.device))
            
            self.model_int = IntensityRegressor(self.model_name).to(self.device)
            self.model_int.load_state_dict(torch.load(int_path, map_location=self.device))
            
            self.model_stc = MultiTaskMotivationClassifier(self.model_name).to(self.device)
            self.model_stc.load_state_dict(torch.load(stc_path, map_location=self.device))
            
            self.model_emo.eval()
            self.model_int.eval()
            self.model_stc.eval()
            print("✅ 成功從 models/ 目錄載入所有權重")
        except Exception as e:
            print(f"❌ 載入模型失敗: {e}")

    def evaluate(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
        
        with torch.no_grad():

            emo_logits = self.model_emo(inputs['input_ids'], inputs['attention_mask'])
            emo_probs = F.softmax(emo_logits, dim=1)
            emo_conf, emo_idx = torch.max(emo_probs, dim=1)
            emo_label = self.go_labels[emo_idx.item()]

            raw_int = self.model_int(inputs['input_ids'], inputs['attention_mask']).item()
            intensity = max(0.0, min(1.5, raw_int))

            m_logits, r_logits, _ = self.model_stc(inputs['input_ids'], inputs['attention_mask'])
            m_probs = F.softmax(m_logits, dim=1)
            r_probs = F.softmax(r_logits, dim=1)
            
            m_conf, m_idx = torch.max(m_probs, dim=1)
            r_conf, r_idx = torch.max(r_probs, dim=1)
            
            m_pred = self.m_labels[m_idx.item()]
            r_pred = self.reiss_labels[r_idx.item()]

        score = 50.0 
        
        positive_list = ['love', 'joy', 'gratitude', 'admiration', 'pride', 'optimism', 'caring']
        negative_list = ['anger', 'disgust', 'annoyance', 'remorse', 'disappointment']
        
        if emo_label in positive_list:
            score += (30 * intensity * emo_conf.item())
        elif emo_label in negative_list:
            score -= (35 * intensity * emo_conf.item())
        elif emo_label in ['sadness', 'grief']:
            if m_pred in ['social', 'legacy'] or r_pred in ['family', 'honor']:
                score += (25 * intensity)
            else:
                score -= (20 * intensity)

        if m_pred in ['social', 'legacy']:
            score += (35 * m_conf.item())
        elif m_pred == 'stability' and emo_label == 'neutral':
            score += (5 * m_conf.item())
            
        if r_pred in ['family', 'honor', 'idealism']:
            score += (20 * r_conf.item())

        final_score = round(max(0, min(100, score)), 2)
        decision = "保留 (Keep)" if final_score >= 60 else "建議斷捨離 (Discard)"
        
        return {
            "text": text,
            "final_score": final_score,
            "decision": decision,
            "details": {
                "emotion": {"label": emo_label, "conf": emo_conf.item()},
                "intensity": intensity,
                "maslow": {"label": m_pred, "conf": m_conf.item()},
                "reiss": {"label": r_pred, "conf": r_conf.item()}
            }
        }
