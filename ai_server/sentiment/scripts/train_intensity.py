import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_server.sentiment.core.model_arch import IntensityRegressor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-multilingual-cased"
SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "intensity_model.pt")

def main():
    print("正在準備強度訓練資料...")
    train_url = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/train_text.txt"
    label_url = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/train_labels.txt"
    
    texts = pd.read_csv(train_url, sep='\t', header=None, names=['tweet'], quoting=3)
    labels = pd.read_csv(label_url, sep='\t', header=None, names=['label'])
    df = pd.concat([texts, labels], axis=1).dropna()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def preprocess(examples):
        tokenized = tokenizer(examples["tweet"], truncation=True, padding="max_length", max_length=128)
        tokenized["labels"] = [float(l) / 3.0 for l in examples["label"]]
        return tokenized

    ds = Dataset.from_pandas(df).map(preprocess, batched=True)
    ds.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    model = IntensityRegressor(MODEL_NAME).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()

    print("🚀 開始訓練強度模型...")
    for epoch in range(3):
        model.train()
        loop = tqdm(loader)
        for batch in loop:
            optimizer.zero_grad()
            outputs = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
            loss = criterion(outputs, batch["labels"].to(DEVICE).float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
    
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ 強度模型已儲存至 {SAVE_PATH}")

if __name__ == "__main__":
    main()