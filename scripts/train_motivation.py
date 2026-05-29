import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import sys
import ast

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from core.model_arch import MultiTaskMotivationClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-multilingual-cased"
SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "motivation_model.pt")

MASLOW_LABELS = ['social', 'doing', 'legacy', 'stability', 'neutral']
REISS_LABELS = ['status', 'approval', 'independence', 'order', 'social_contact', 'honor', 'idealism', 'vengeance', 'romance', 'family', 'food', 'physical_exercise', 'saving', 'curiosity', 'tranquility', 'neutral']

def get_idx(val, labels):
    try:
        actual_list = ast.literal_eval(val)
        label = str(actual_list[0]).split(':')[0].lower().strip() if actual_list else 'neutral'
        mapping = {'physiological': 'stability', 'love': 'social', 'esteem': 'social', 'spiritual growth': 'legacy', 'indep': 'independence', 'contact': 'social_contact', 'health': 'physical_exercise', 'rest': 'tranquility'}
        final = mapping.get(label, label)
        return labels.index(final) if final in labels else labels.index('neutral')
    except: return labels.index('neutral')

def train():
    global PROJECT_ROOT, SAVE_PATH 
    
    print("🚀 載入心理動機資料...")

    motiv_csv = os.path.join(PROJECT_ROOT, "data", "motiv.csv")
    emo_csv = os.path.join(PROJECT_ROOT, "data", "emo.csv")
    
    if not os.path.exists(motiv_csv) or not os.path.exists(emo_csv):
        print(f"❌ 錯誤：找不到資料檔案。請確認檔案位於: {os.path.join(PROJECT_ROOT, 'data/')}")
        return

    df = pd.merge(pd.read_csv(motiv_csv), pd.read_csv(emo_csv), on=['storyid', 'linenum']).rename(columns={'sentence_x': 'sentence'})
    df['m_idx'] = df['maslow'].apply(lambda x: get_idx(x, MASLOW_LABELS))
    df['r_idx'] = df['reiss'].apply(lambda x: get_idx(x, REISS_LABELS))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    class MTDataset(Dataset):
        def __init__(self, df, tokenizer): 
            self.df = df
            self.tokenizer = tokenizer
        def __len__(self): return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            enc = self.tokenizer(str(row['sentence']), truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            return {
                'input_ids': enc['input_ids'].flatten(), 
                'attention_mask': enc['attention_mask'].flatten(), 
                'm_label': row['m_idx'], 
                'r_label': row['r_idx']
            }

    loader = DataLoader(MTDataset(df, tokenizer), batch_size=16, shuffle=True)
    model = MultiTaskMotivationClassifier(MODEL_NAME).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    
    print(f"🔥 開始多任務訓練，權重將儲存至: {SAVE_PATH}")
    for epoch in range(10):
        model.train()
        loop = tqdm(loader)
        for batch in loop:
            optimizer.zero_grad()
            m_out, r_out, _ = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
            loss = nn.CrossEntropyLoss()(m_out, batch['m_label'].to(DEVICE)) + nn.CrossEntropyLoss()(r_out, batch['r_label'].to(DEVICE))
            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch+1}/10]")
            loop.set_postfix(loss=loss.item())

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ 心理動機模型訓練完成！已儲存至 {SAVE_PATH}")

if __name__ == "__main__":
    train()
