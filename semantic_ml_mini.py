
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from datetime import datetime
import numpy as np

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"
MODEL_PATH = "/content/emile_semantic_ml_mini.pt"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
transformer_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
transformer_model.eval()

sentence_model = SentenceTransformer("all-MiniLM-L6-v2").to(DEVICE)
sentence_model.eval()

class LogSemioticExtractor(nn.Module):
    def __init__(self, hidden_dim=384, numeric_dim=8):
        super().__init__()
        self.encoder = transformer_model
        self.text_fc = nn.Linear(self.encoder.config.hidden_size, hidden_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.1)

        self.numeric_enabled = numeric_dim > 0
        if self.numeric_enabled:
            self.numeric_fc = nn.Linear(numeric_dim, hidden_dim // 2)
            self.fc_combined = nn.Linear(hidden_dim + (hidden_dim // 2), 384) # match ST dimension
        else:
            self.fc_direct = nn.Linear(hidden_dim, 384)

        self.norm = nn.LayerNorm(384)

    def forward(self, input_ids, attention_mask, numeric_values=None):
        with torch.no_grad():
            outputs = self.encoder(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        text_features = self.activation(self.text_fc(pooled_output))
        text_features = self.dropout(text_features)

        if self.numeric_enabled and numeric_values is not None:
            numeric_feats = self.activation(self.numeric_fc(numeric_values))
            if numeric_feats.ndim == 1:
                numeric_feats = numeric_feats.unsqueeze(0)
            combined = torch.cat([text_features, numeric_feats], dim=1)
            out = self.fc_combined(combined)
        else:
            out = self.fc_direct(text_features) if hasattr(self, 'fc_direct') else text_features

        return self.norm(out)

def extract_log_numeric_features(line: str, max_features=8):
    """
    Simple numeric extraction from a log line.
    You can refine to parse (ms), CPU usage, etc.
    """
    numbers = re.findall(r'\b\d+\.?\d*\b', line)
    raw = []
    for num in numbers:
        try:
            raw.append(float(num))
        except ValueError:
            pass

    features = []
    if raw:
        features.extend([
            len(raw),
            np.mean(raw),
            np.std(raw) if len(raw) > 1 else 0.0,
            max(raw),
            min(raw),
            np.median(raw),
            np.percentile(raw, 25) if len(raw)>3 else min(raw),
            np.percentile(raw, 75) if len(raw)>3 else max(raw)
        ])
    # pad or truncate
    features = features[:max_features]
    if len(features) < max_features:
        features.extend([0.0]*(max_features-len(features)))
    return torch.tensor(features, dtype=torch.float32)

def build_log_dataset(log_paths):
    data = []
    for p in log_paths:
        if not os.path.exists(p):
            print(f"Missing: {p}")
            continue
        with open(p, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line_text = line.strip()
            if line_text:
                data.append(line_text)
    return data

def train_on_logs(model, log_data, epochs=3, batch_size=16, lr=1e-5):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CosineEmbeddingLoss()

    random.shuffle(log_data)
    model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        total_loss = 0.0
        count = 0

        for i in range(0, len(log_data), batch_size):
            chunk = log_data[i:i+batch_size]
            if not chunk:
                continue

            batch_loss = 0.0
            subcount = 0

            for line_text in chunk:
                inputs = tokenizer(line_text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
                numeric_vals = extract_log_numeric_features(line_text).to(DEVICE)

                outputs = model(inputs['input_ids'], inputs['attention_mask'], numeric_vals.unsqueeze(0))

                with torch.no_grad():
                    ref_emb = sentence_model.encode([line_text], convert_to_tensor=True).to(DEVICE)

                # shape match
                min_sz = min(outputs.shape[0], ref_emb.shape[0])
                out_batch = outputs[:min_sz]
                ref_batch = ref_emb[:min_sz]

                target = torch.ones(out_batch.shape[0], device=DEVICE)
                loss = loss_fn(out_batch, ref_batch, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()
                subcount += 1

            if subcount>0:
                total_loss += batch_loss / subcount
                count += 1

        if count>0:
            print(f"  Avg Loss: {total_loss / count:.4f}")
        else:
            print("  No valid lines")

    print("Done training logs")

if __name__ == "__main__":
    # Example usage
    log_files = [
       "/content/logs/emile4_sim_20250228_044253.log",
       # etc.
    ]
    log_data = build_log_dataset(log_files)

    # either load or new
    if os.path.exists(MODEL_PATH):
        cpt = torch.load(MODEL_PATH)
        model = LogSemioticExtractor().to(DEVICE)
        model.load_state_dict(cpt['model_state_dict'])
    else:
        model = LogSemioticExtractor(hidden_dim=384, numeric_dim=8).to(DEVICE)

    train_on_logs(model, log_data, epochs=2, batch_size=8, lr=1e-5)

    torch.save({'model_state_dict': model.state_dict()}, MODEL_PATH)
    print(f"Saved to {MODEL_PATH}")
