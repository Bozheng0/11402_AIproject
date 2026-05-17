import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import os
import sys

# 確保能找到核心架構定義
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_server.sentiment.core.model_arch import DanshariClassifier, IntensityRegressor, MultiTaskMotivationClassifier

class CustomTuningDataset(Dataset):
    def __init__(self, file_path, tokenizer, go_labels, m_labels, r_labels):
        self.data = []
        self.tokenizer = tokenizer
        self.go_map = {label: i for i, label in enumerate(go_labels)}
        self.m_map = {label: i for i, label in enumerate(m_labels)}
        self.r_map = {label: i for i, label in enumerate(r_labels)}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # 自動修正資料集中的不相容標籤
                if item['emotion'] == 'guilt': item['emotion'] = 'remorse'
                if item['emotion'] == 'nostalgia': item['emotion'] = 'love'
                self.data.append(item)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(item['text'], truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'emo_label': torch.tensor(self.go_map[item['emotion']]),
            'int_label': torch.tensor(item['intensity'], dtype=torch.float),
            'm_label': torch.tensor(self.m_map[item['maslow']]),
            'r_label': torch.tensor(self.r_map[item['reiss']]) if 'reiss' in item else torch.tensor(self.r_map['neutral'])
        }

def finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    go_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
    m_labels = ['social', 'doing', 'legacy', 'stability', 'neutral']
    r_labels = ['status', 'approval', 'independence', 'order', 'social_contact', 'honor', 'idealism', 'vengeance', 'romance', 'family', 'food', 'physical_exercise', 'saving', 'curiosity', 'tranquility', 'neutral']

    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    
    # 載入現有模型
    model_emo = DanshariClassifier(model_name).to(device)
    model_emo.load_state_dict(torch.load(os.path.join(model_dir, "emotion_model.pt"), map_location=device))
    
    model_int = IntensityRegressor(model_name).to(device)
    model_int.load_state_dict(torch.load(os.path.join(model_dir, "intensity_model.pt"), map_location=device))
    
    model_stc = MultiTaskMotivationClassifier(model_name).to(device)
    model_stc.load_state_dict(torch.load(os.path.join(model_dir, "motivation_model.pt"), map_location=device))

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "additional_train_data.jsonl")
    dataset = CustomTuningDataset(data_path, tokenizer, go_labels, m_labels, r_labels)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = optim.AdamW(list(model_emo.parameters()) + list(model_int.parameters()) + list(model_stc.parameters()), lr=2e-6)

    print("🚀 開始針對斷捨離語境進行深度微調...")
    for epoch in range(10):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            input_ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            
            loss_emo = nn.CrossEntropyLoss()(model_emo(input_ids, mask), batch['emo_label'].to(device))
            loss_int = nn.MSELoss()(model_int(input_ids, mask).squeeze(), batch['int_label'].to(device))
            out_m, out_r, _ = model_stc(input_ids, mask)
            loss_m = nn.CrossEntropyLoss()(out_m, batch['m_label'].to(device))
            loss_r = nn.CrossEntropyLoss()(out_r, batch['r_label'].to(device))
            
            loss = loss_emo + loss_int + loss_m + loss_r
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

    torch.save(model_emo.state_dict(), os.path.join(model_dir, "emotion_model.pt"))
    torch.save(model_int.state_dict(), os.path.join(model_dir, "intensity_model.pt"))
    torch.save(model_stc.state_dict(), os.path.join(model_dir, "motivation_model.pt"))
    print("✅ 微調完成，模型權重已更新。")

if __name__ == "__main__":
    finetune()