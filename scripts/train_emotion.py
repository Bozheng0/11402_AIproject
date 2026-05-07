import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model_arch import DanshariClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-multilingual-cased"
BATCH_SIZE = 16
EPOCHS = 10
SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "emotion_model.pt")
NUM_LABELS = 28

def preprocess_func(examples):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    labels_matrix = np.zeros((len(examples["labels"]), NUM_LABELS))
    for i, label_list in enumerate(examples["labels"]):
        for l in label_list:
            if l < NUM_LABELS: labels_matrix[i, l] = 1.0
    tokenized["labels"] = labels_matrix.tolist()
    return tokenized

def train():
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
    tokenized_ds = dataset.map(preprocess_func, batched=True, remove_columns=dataset["train"].column_names)
    tokenized_ds.set_format("torch")
    train_loader = DataLoader(tokenized_ds["train"], batch_size=BATCH_SIZE, shuffle=True)

    model = DanshariClassifier(MODEL_NAME, num_labels=NUM_LABELS).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()

    print(f"🚀 開始訓練情緒識別模型...")
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader)
        for batch in loop:
            input_ids, mask = batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE).float()
            optimizer.zero_grad()
            loss = criterion(model(input_ids, mask), labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
        torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ 訓練完成！已儲存至 {SAVE_PATH}")

if __name__ == "__main__":
    train()