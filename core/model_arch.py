import torch
import torch.nn as nn
from transformers import AutoModel

class DanshariClassifier(nn.Module):
    def __init__(self, model_name, num_labels=28):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(768, num_labels)
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.classifier(out)

class IntensityRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Linear(768, 1)
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.regressor(out)

class MultiTaskMotivationClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.maslow_head = nn.Linear(768, 5)
        self.reiss_head = nn.Linear(768, 16)
        self.effect_head = nn.Linear(768, 9)
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.maslow_head(out), self.reiss_head(out), self.effect_head(out)
