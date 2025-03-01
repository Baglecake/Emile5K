
import os
import re
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# ================================
# 1. CONFIGURATIONS
# ================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"
MODEL_PATH = "/content/emile_semantic_ml_flash.pt"  # Where to save
LOG_DIR = "/content/logs"  # Directory holding emile logs

# If you have a second reference model
REFERENCE_MODEL = "/content/emile_semantic_ml_mini.pt"

# Create output directories if needed
os.makedirs(LOG_DIR, exist_ok=True)

# ================================
# 2. PRETRAINED MODELS
# ================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
transformer_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
transformer_model.eval()

sentence_model = SentenceTransformer(REFERENCE_MODEL).to(DEVICE)
sentence_model.eval()

# ================================
# 3. CUSTOM MODEL
# ================================

class EmileLogExtractor(nn.Module):
    """
    Merges text embeddings from a BERT-like model with numeric features
    into a final 384-dim embedding (matching e.g. sentence transformers).
    """
    def __init__(self, hidden_dim=384, numeric_dim=8):
        super().__init__()
        self.encoder = transformer_model

        # Text path
        self.fc_text = nn.Linear(self.encoder.config.hidden_size, hidden_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.1)

        # Numeric path
        self.numeric_enabled = numeric_dim > 0
        if self.numeric_enabled:
            self.fc_numeric = nn.Linear(numeric_dim, hidden_dim // 2)
            self.fc_combined = nn.Linear(hidden_dim + hidden_dim // 2, 384)
        else:
            # Just go directly to 384 if no numeric features
            self.fc_direct = nn.Linear(hidden_dim, 384)

        self.norm = nn.LayerNorm(384)

    def forward(self, input_ids, attention_mask, numeric_values=None):
        # Pass text through BERT
        with torch.no_grad():
            outputs = self.encoder(input_ids, attention_mask=attention_mask)
        # Use [CLS] embedding
        text_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (batch, 768)

        # Map down
        text_features = self.activation(self.fc_text(text_embedding))
        text_features = self.dropout(text_features)  # shape: (batch, hidden_dim)

        if self.numeric_enabled and numeric_values is not None:
            numeric_proj = self.activation(self.fc_numeric(numeric_values))
            if numeric_proj.ndim == 1:  # if shape (dim,) => make it (1,dim)
                numeric_proj = numeric_proj.unsqueeze(0)
            combined = torch.cat([text_features, numeric_proj], dim=1)
            out = self.fc_combined(combined)  # (batch, 384)
        else:
            # direct path
            out = self.fc_direct(text_features) if hasattr(self, 'fc_direct') else text_features

        return self.norm(out)  # final shape: (batch, 384)

# ================================
# 4. LOG PARSING FOR NUMERIC FEATURES
# ================================
def extract_targeted_numeric_features(line: str):
    """
    Parse known patterns from your logs.
    Example:
     - "Pass: HighLevelSynthesis - 0.04816 (ms)"
     - "Resource Usage => CPU: 8.7%, Memory: 5.2%, Avail: 80998 MB"
     - "Distinction: 0.647, Coherence: 0.052, Entropy: 0.869"
     - etc.

    We'll keep it simple with some placeholders:
    """
    # Initialize numeric features
    # Suppose we want 8 "slots"
    # slot 0: pass_time_ms
    # slot 1: CPU usage
    # slot 2: memory usage
    # slot 3: distinction
    # slot 4: coherence
    # slot 5: surplus
    # slot 6: random_value
    # slot 7: something else
    feats = [0.0]*8

    # 1) Look for pass time in (ms)
    match_pass = re.search(r'Pass:\s*\S+\s*-\s*([\d.]+)\s*\(ms\)', line)
    if match_pass:
        feats[0] = float(match_pass.group(1))

    # 2) Resource usage
    match_cpu = re.search(r'CPU:\s*([\d.]+)%', line)
    if match_cpu:
        feats[1] = float(match_cpu.group(1))

    match_mem = re.search(r'Memory:\s*([\d.]+)%', line)
    if match_mem:
        feats[2] = float(match_mem.group(1))

    # 3) Distinction, Coherence, Surplus
    match_dist = re.search(r'Distinction:\s*([\d.]+)', line)
    if match_dist:
        feats[3] = float(match_dist.group(1))

    match_coh = re.search(r'Coherence:\s*([\d.]+)', line)
    if match_coh:
        feats[4] = float(match_coh.group(1))

    match_surp = re.search(r'surplus.?=?([\d.]+)', line)
    if match_surp:
        feats[5] = float(match_surp.group(1))

    # 4) A random example: "random_value=1.234"
    match_rand = re.search(r'random_value.?=?([\d.]+)', line)
    if match_rand:
        feats[6] = float(match_rand.group(1))

    # We can store something else in slot 7 if we want, or keep it zero.

    return torch.tensor(feats, dtype=torch.float32)

# ================================
# 5. BUILD DATASET FROM LOGS
# ================================
def gather_log_lines(log_dir):
    """
    Collect lines from all .log files in the directory.
    """
    lines_collected = []
    for fname in os.listdir(log_dir):
        if fname.endswith(".log"):
            path = os.path.join(log_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lines_collected.append(line)
    return lines_collected

# ================================
# 6. TRAINING FUNCTION
# ================================
def train_logs_only(model, log_lines, epochs=3, batch_size=16, lr=1e-5, max_length=512):
    """
    Each line => text + numeric => model => embedding =>
    compare with reference embedding => CosineEmbeddingLoss.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CosineEmbeddingLoss()

    random.shuffle(log_lines)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(log_lines), batch_size):
            chunk = log_lines[i : i+batch_size]
            if not chunk:
                continue

            # We'll accumulate loss across the chunk
            batch_loss = 0.0
            subcount = 0

            for line in chunk:
                # 1) Tokenize text
                inputs = tokenizer(line, return_tensors="pt",
                                   truncation=True, max_length=max_length).to(DEVICE)

                # 2) Extract numeric features
                numeric_feats = extract_targeted_numeric_features(line).to(DEVICE)
                # Expand to shape (batch=1, numeric_dim=8)
                numeric_feats = numeric_feats.unsqueeze(0)

                # 3) Forward pass
                outputs = model(
                    inputs['input_ids'],
                    inputs['attention_mask'],
                    numeric_feats
                )  # shape: (1, 384)

                # 4) Reference embedding from your domain reference model
                with torch.no_grad():
                    ref_emb = sentence_model.encode([line], convert_to_tensor=True).to(DEVICE)
                    # shape: (1, 384)

                # 5) cos embed loss
                # shape check
                if outputs.shape[0] != ref_emb.shape[0]:
                    min_sz = min(outputs.shape[0], ref_emb.shape[0])
                    outputs = outputs[:min_sz]
                    ref_emb = ref_emb[:min_sz]

                target = torch.ones(outputs.shape[0], device=DEVICE)
                loss = loss_fn(outputs, ref_emb, target)

                # 6) Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()
                subcount += 1

            if subcount>0:
                total_loss += (batch_loss/subcount)
                num_batches += 1

        if num_batches>0:
            avg_loss = total_loss / num_batches
            print(f"  Avg Loss: {avg_loss:.4f}")
        else:
            print("  No valid lines processed.")

    print("Training complete on logs-only data.")

# ================================
# 7. MAIN EXECUTION
# ================================
if __name__ == "__main__":
    # 1) Gather log lines
    lines = gather_log_lines(LOG_DIR)
    print(f"Collected {len(lines)} lines from logs at {LOG_DIR}.")

    if not lines:
        print("No log lines found. Exiting.")
        exit(0)

    # 2) Create or load model
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH} ...")
        checkpoint = torch.load(MODEL_PATH)
        # If you want to handle config, do so; else, assume default dims
        model = EmileLogExtractor(hidden_dim=384, numeric_dim=8).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    else:
        print("Creating new logs-only model ...")
        model = EmileLogExtractor(hidden_dim=384, numeric_dim=8).to(DEVICE)

    # 3) Train
    train_logs_only(model, lines, epochs=2, batch_size=16, lr=1e-5)

    # 4) Save
    tosave = {
        'model_state_dict': model.state_dict(),
        'timestamp': time.strftime("%Y%m%d-%H%M%S")
    }
    torch.save(tosave, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
