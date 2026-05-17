import torch
import torch.nn as nn
import numpy as np
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

checkpoint_path = "models/best_macro_f1_bilstm_attention_model.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)

word2idx = checkpoint["word2idx"]
idx2word = checkpoint["idx2word"]
config = checkpoint["config"]

MAX_LEN = config["max_len"]
PAD_IDX = config["pad_idx"]


class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_classes, pad_idx, num_layers=2, dropout=0.5):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = nn.Linear(hidden_dim * 2, 1)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids):
        mask = input_ids != 0

        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)

        attention_scores = self.attention(lstm_out).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~mask, -1e9)

        attention_weights = torch.softmax(attention_scores, dim=1)

        context = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        context = self.dropout(context)

        logits = self.classifier(context)
        return logits


model = BiLSTMAttentionClassifier(
    vocab_size=config["vocab_size"],
    embed_dim=config["embed_dim"],
    hidden_dim=config["hidden_dim"],
    num_classes=config["num_classes"],
    pad_idx=config["pad_idx"],
    num_layers=config["num_layers"],
    dropout=config["dropout"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded successfully")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return text.split()


def encode_text(text, word2idx, max_len):
    tokens = tokenize(text)
    ids = [word2idx.get(t, word2idx["<UNK>"]) for t in tokens]

    ids = ids[:max_len]

    if len(ids) < max_len:
        ids += [word2idx["<PAD>"]] * (max_len - len(ids))

    return ids


def predict(text):
    model.eval()

    text = clean_text(text)
    ids = encode_text(text, word2idx, MAX_LEN)

    input_ids = torch.tensor([ids], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_ids)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    scores = np.array([1, 2, 3, 4, 5])
    soft_score = float(np.sum(probs * scores))
    pred_class = int(np.argmax(probs)) + 1

    return {
        "pred_class": pred_class,
        "soft_score": soft_score,
        "probs": probs
    }


if __name__ == "__main__":
    print("\n=== 推論系統啟動 ===")

    while True:
        text = input("輸入評論 (q離開): ")

        if text.lower() == "q":
            print("結束推論")
            break

        result = predict(text)

        print("\n=== 結果 ===")
        print("預測類別(1~5):", result["pred_class"])
        print("soft score:", round(result["soft_score"], 3))

        print("機率分佈:")
        for i, p in enumerate(result["probs"], 1):
            print(f"{i}分: {p:.4f}")

        print("-" * 40)
