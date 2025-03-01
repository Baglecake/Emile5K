# -*- coding: utf-8 -*-
"""
Enhanced & Integrated Émile5SemanticML Trainer
Implements improvements for stability, efficiency, and enhanced numeric integration
with direct Émile simulation connection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import spacy
import time
import subprocess
from collections import Counter
import nltk
import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from transformers import AutoTokenizer, AutoModel
import random
import threading
import matplotlib.pyplot as plt
from datetime import datetime
import json

# ================================
# 1. CONFIGURATIONS & DATA LOADING
# ================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"  # Or any other suitable BERT model
MODEL_PATH = "/content/emile_semantic_ml_mini.pt"  # Path to save the trained model
CACHE_DIR = "/content/cache"  # Directory to cache embeddings
LOG_DIR = "/content/logs"  # Directory for new simulation logs
EMILE_SCRIPT = "simulation_runner_logs.py"  # Path to Émile simulation script

# Create necessary directories
for directory in [CACHE_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# --- Load Pretrained Models ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
transformer_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
transformer_model.eval()

try:
    spacy_model = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm model...")
    spacy.cli.download("en_core_web_sm")
    spacy_model = spacy.load("en_core_web_sm")

sentence_model = SentenceTransformer('all-MiniLM-L6-v2').to(DEVICE)
sentence_model.eval()

# --- Define Paths and Load Data ---
data_path = "/content/"  # Path for training data
log_paths = glob.glob(os.path.join(LOG_DIR, "emile*_sim_*.log"))  # Find all Émile log files
if not log_paths:
    log_paths = [os.path.join(LOG_DIR, f"emile5_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")]
training_files = glob.glob(os.path.join(data_path, "*.txt"))

def read_text(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

training_data = {os.path.basename(f): read_text(f) for f in training_files}

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# Initialize Tracking Variables
training_metrics = {
    "initial_losses": [],
    "recursive_losses": [],
    "coherence_scores": [],
    "numeric_integration_scores": [],
    "timestamps": []
}

# ================================
# 2. ENHANCED SEMIOTIC DISTINCTION EXTRACTOR
# ================================

class SemioticExtractor(nn.Module):
    """
    Enhanced semantic model with numeric value integration capabilities.
    Supports both old (text_fc/numeric_fc) and new (fc1/fc1_numeric) naming conventions.
    """
    def __init__(self, hidden_dim=384, numeric_dim=8):  # Use 384 instead of 256
        super().__init__()
        self.encoder = transformer_model

        # Text processing pathway
        self.fc1 = nn.Linear(self.encoder.config.hidden_size, hidden_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.1)

        # Numeric processing pathway
        self.numeric_enabled = numeric_dim > 0
        if self.numeric_enabled:
            self.fc1_numeric = nn.Linear(numeric_dim, hidden_dim // 2)
            self.fc_combined = nn.Linear(hidden_dim + (hidden_dim // 2), 384)  # Change to 384
        else:
            self.fc2 = nn.Linear(hidden_dim, 384)  # Change to 384

        # Final normalization
        self.norm = nn.LayerNorm(384)  # Change to 384

    def forward(self, input_ids, attention_mask, numeric_values=None):
        with torch.no_grad():
            outputs = self.encoder(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Process text features
        text_features = self.activation(self.fc1(pooled_output))
        text_features = self.dropout(text_features)

        # Process and combine with numeric features if provided
        if self.numeric_enabled and numeric_values is not None:
            numeric_features = self.activation(self.fc1_numeric(numeric_values))
            combined_features = torch.cat([text_features, numeric_features], dim=1)
            output = self.fc_combined(combined_features)
        else:
            output = self.fc2(text_features) if hasattr(self, 'fc2') else text_features

        return self.norm(output)

# ================================
# 3. IMPROVED QUANTUM SEMANTIC STABILIZER
# ================================

def quantum_semantic_stabilization(num_qubits=4, shots=1024):
    """Enhanced quantum stabilization with measurement shots parameter"""
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    qc = QuantumCircuit(num_qubits)

    # Apply Hadamard gates for superposition
    for i in range(num_qubits):
        qc.h(i)

    # Add entanglement for more complex stabilization
    for i in range(num_qubits-1):
        qc.cx(i, i+1)

    # Add phase kickback
    for i in range(num_qubits):
        qc.rz(np.pi/4, i)

    qc.measure_all()

    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit, shots=shots).result()
    return result.get_counts()

# ================================
# 4. ENHANCED NLP THEME EXTRACTION WITH NUMERIC RECOGNITION
# ================================

def extract_themes(text, top_n=20, min_word_length=3, extract_numeric=True):
    """
    Extract themes with improved filtering and weighting,
    and optional numeric value extraction.
    """
    # Process with spaCy
    doc = spacy_model(text)

    # Extract lemmatized words
    words = [token.lemma_.lower() for token in doc
             if token.is_alpha and
             not token.is_stop and
             len(token.text) >= min_word_length and
             token.pos_ in ['NOUN', 'VERB', 'ADJ']]  # Focus on meaningful POS

    word_freq = Counter(words)
    top_words = word_freq.most_common(top_n)

    result = {"word_themes": top_words}

    # Optional numeric extraction
    if extract_numeric:
        numeric_values = []
        numeric_patterns = {}

        # Extract numbers via regex
        numbers = re.findall(r'\b\d+\.?\d*\b', text)
        numeric_values.extend([float(num) for num in numbers])

        # Potential numeric-related terms
        numeric_modifiers = [
            "increase", "decrease", "oscillate", "accelerate",
            "decelerate", "threshold", "critical", "harmonic",
            "resonant", "quantize", "continuous", "discrete"
        ]

        numeric_transformations = [
            "multiply", "divide", "amplify", "attenuate",
            "exponential", "logarithmic", "scale", "normalize",
            "bound", "unbound", "fraction", "integral"
        ]

        for term in numeric_modifiers:
            count = len(re.findall(rf'\b{term}\w*\b', text.lower()))
            if count > 0:
                numeric_patterns[term] = count

        for term in numeric_transformations:
            count = len(re.findall(rf'\b{term}\w*\b', text.lower()))
            if count > 0:
                numeric_patterns[term] = count

        result["numeric_values"] = numeric_values
        result["numeric_patterns"] = numeric_patterns

        # Basic stats
        if numeric_values:
            result["numeric_stats"] = {
                "count": len(numeric_values),
                "mean": np.mean(numeric_values),
                "std": np.std(numeric_values),
                "min": min(numeric_values),
                "max": max(numeric_values)
            }

    return result

symbolic_themes = {file: extract_themes(content) for file, content in training_data.items()}

# ================================
# 5. IMPROVED SENTENCE EMBEDDING & CONCEPT MAPPING
# ================================

def precompute_embeddings(text_data, cache_path=None, extract_numeric=True):
    """
    Precompute and optionally cache sentence embeddings.
    Also can extract numeric values from sentences.
    """
    sentences = [sent.strip() for sent in text_data.split(".") if sent.strip()]

    if cache_path and os.path.exists(cache_path):
        try:
            cached_data = torch.load(cache_path)
            print(f"Loaded cached embeddings from {cache_path}")
            return (cached_data['sentences'],
                    cached_data['embeddings'],
                    cached_data.get('numeric_values', None))
        except Exception as e:
            print(f"Could not load cache: {e}")

    batch_size = 32
    all_embeddings = []

    numeric_values = None
    if extract_numeric:
        numeric_values = []
        for sentence in sentences:
            numbers = re.findall(r'\b\d+\.?\d*\b', sentence)
            for num in numbers:
                try:
                    numeric_values.append({
                        'value': float(num),
                        'sentence_idx': sentences.index(sentence)
                    })
                except ValueError:
                    pass

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        if not batch:
            continue
        batch_embeddings = sentence_model.encode(batch, convert_to_tensor=True)
        all_embeddings.append(batch_embeddings)

    if not all_embeddings:
        return [], torch.tensor([]), None

    embeddings = torch.cat(all_embeddings, dim=0)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache_data = {
            'sentences': sentences,
            'embeddings': embeddings
        }
        if numeric_values:
            cache_data['numeric_values'] = numeric_values

        torch.save(cache_data, cache_path)
        print(f"Saved embeddings to cache at {cache_path}")

    return sentences, embeddings, numeric_values

sentence_embeddings = {}
numeric_data = {}
for file, content in training_data.items():
    cache_path = os.path.join(CACHE_DIR, f"{file}.emb")
    sentences, embeddings, numeric_values = precompute_embeddings(content, cache_path)
    sentence_embeddings[file] = (sentences, embeddings)
    if numeric_values:
        numeric_data[file] = numeric_values

def find_related_concepts(query, top_n=5):
    """Find semantically related concepts using the sentence embeddings."""
    query_embedding = sentence_model.encode([query], convert_to_tensor=True).to(DEVICE)
    results = []

    for file, (sentences, embeddings) in sentence_embeddings.items():
        if len(sentences) == 0 or embeddings.shape[0] == 0:
            continue

        if embeddings.device != query_embedding.device:
            embeddings = embeddings.to(query_embedding.device)

        similarities = cosine_similarity(query_embedding.cpu().numpy(),
                                         embeddings.cpu().numpy())[0]

        top_indices = np.argsort(similarities)[-3:][::-1]
        for idx in top_indices:
            if idx < len(sentences):
                results.append((file, sentences[idx], similarities[idx]))

    results = sorted(results, key=lambda x: x[2], reverse=True)
    return results[:top_n]

# ================================
# 6. IMPROVED NUMERIC-SEMANTIC INTEGRATION
# ================================

def extract_numeric_features(text, max_features=8):
    """
    Extract numeric features from text for input to the semantic model.
    """
    features = []

    numbers = re.findall(r'\b\d+\.?\d*\b', text)
    raw_values = []
    for num in numbers:
        try:
            raw_values.append(float(num))
        except ValueError:
            pass

    if raw_values:
        features.extend([
            len(raw_values),
            np.mean(raw_values),
            np.std(raw_values) if len(raw_values) > 1 else 0.0,
            max(raw_values),
            min(raw_values),
            np.median(raw_values),
            np.percentile(raw_values, 25) if len(raw_values) > 3 else min(raw_values),
            np.percentile(raw_values, 75) if len(raw_values) > 3 else max(raw_values)
        ])

    features = features[:max_features]
    if len(features) < max_features:
        features.extend([0.0] * (max_features - len(features)))

    return torch.tensor(features, dtype=torch.float32)

def calculate_numeric_integration_score(text, numeric_values):
    """
    Calculate how well numeric values are integrated with semantic context.
    """
    if not numeric_values:
        return 0.0

    context_indicators = [
        "value", "measure", "rate", "score", "level", "threshold",
        "percent", "amount", "quantity", "frequency", "probability",
        "equals", "is", "was", "reached", "exceeded", "dropped",
        "increased", "decreased", "accelerated", "decelerated",
        "equals", "=", "approximately", "about", "around", "nearly"
    ]
    context_score = 0.0
    for indicator in context_indicators:
        if indicator in text.lower():
            context_score += 0.1
    context_score = min(1.0, context_score)

    unit_indicators = [
        "%", "percent", "kg", "meters", "seconds", "minutes", "hours",
        "days", "watts", "joules", "degrees", "Hz", "hertz", "bytes",
        "bps", "bits", "pixels", "frames", "cycles", "iterations"
    ]
    unit_score = 0.0
    for unit in unit_indicators:
        if unit in text:
            unit_score += 0.2
    unit_score = min(1.0, unit_score)

    transform_indicators = [
        "times", "multiplied", "divided", "plus", "minus", "added",
        "subtracted", "increased by", "decreased by", "factor of",
        "scaled", "normalized", "averaged", "mean", "median", "deviation"
    ]
    transform_score = 0.0
    for transform in transform_indicators:
        if transform in text.lower():
            transform_score += 0.15
    transform_score = min(1.0, transform_score)

    final_score = 0.4 * context_score + 0.3 * unit_score + 0.3 * transform_score
    return final_score

# ================================
# 7. IMPROVED TRAINING AND REFINEMENT
# ================================

def train_semiotic_model(model, training_data, epochs=5, batch_size=16, learning_rate=1e-5, save_interval=2):
    """Training loop that uses CosineEmbeddingLoss and numeric integration tracking."""
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    loss_function = nn.CosineEmbeddingLoss()

    best_loss = float('inf')

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0
        batch_count = 0

        data_items = list(training_data.items())
        random.shuffle(data_items)

        for i in range(0, len(data_items), batch_size):
            batch_data = dict(data_items[i:i + min(batch_size, len(data_items) - i)])
            if not batch_data:
                continue

            batch_text = " ".join(batch_data.values())
            if not batch_text.strip():
                continue

            inputs = tokenizer(batch_text, return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(DEVICE)

            numeric_features = extract_numeric_features(batch_text).to(DEVICE)

            # Expand dims if needed
            batch_size_actual = inputs['input_ids'].shape[0]
            if batch_size_actual > 1:
                numeric_features = numeric_features.unsqueeze(0).expand(batch_size_actual, -1)

            outputs = model(inputs['input_ids'], inputs['attention_mask'], numeric_features)

            with torch.no_grad():
                sentence_transformer_embeddings = sentence_model.encode(batch_text, convert_to_tensor=True).to(DEVICE)

            if outputs.shape[0] != sentence_transformer_embeddings.shape[0]:
                min_size = min(outputs.shape[0], sentence_transformer_embeddings.shape[0])
                outputs = outputs[:min_size]
                sentence_transformer_embeddings = sentence_transformer_embeddings[:min_size]

            target = torch.ones(outputs.shape[0]).to(DEVICE)
            loss = loss_function(outputs, sentence_transformer_embeddings, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            numeric_score = calculate_numeric_integration_score(batch_text, extract_numeric_features(batch_text).tolist())
            training_metrics['numeric_integration_scores'].append(numeric_score)

            training_metrics['initial_losses'].append(loss.item())
            training_metrics['timestamps'].append(time.time())

            if batch_count % 5 == 0:
                print(f"  Batch {batch_count}: Loss = {loss.item():.4f}, Numeric Integration = {numeric_score:.4f}")

        avg_loss = total_loss / max(1, batch_count)
        print(f"  Average Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

        if epoch % save_interval == 0 or avg_loss < best_loss:
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_model(model, MODEL_PATH.replace('.pt', '_best.pt'))
                print(f"  New best model saved (loss: {best_loss:.4f})")
            else:
                save_model(model)
                print(f"  Model saved at epoch {epoch+1}")


def run_emile_simulation():
    """Runs the Émile simulation in a separate process, logs to a new file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"emile4_sim_{timestamp}.log")

    global log_paths
    log_paths = [log_file]

    print(f"🚀 Starting Émile simulation. Logging to: {log_file}")
    log_handle = open(log_file, 'w', encoding='utf-8')

    try:
        process = subprocess.Popen(
            ["python", EMILE_SCRIPT],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        return process, log_file, log_handle
    except Exception as e:
        log_handle.close()
        print(f"Error starting Émile simulation: {e}")
        return None, log_file, None


def refine_semiotic_model_with_simulation(model, update_interval=10, batch_size=16, learning_rate=1e-5, runtime=1800):
    """
    Runs Émile simulation while training on new Symbolic Expression lines in the logs.
    """
    print("\n🔄 Starting integrated Émile simulation and semantic model training")
    emile_process, log_file, log_handle = run_emile_simulation()

    if emile_process is None:
        print("❌ Failed to start Émile simulation. Aborting integrated training.")
        return

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_function = nn.CosineEmbeddingLoss()

    model.train()

    start_time = time.time()
    last_position = 0
    batch_text = []
    batch_numeric = []
    batch_count = 0
    total_loss = 0
    symbolic_expressions_count = 0
    numeric_integration_score_sum = 0.0

    try:
        while time.time() - start_time < runtime and emile_process.poll() is None:
            time.sleep(0.1)
            with open(log_file, "r", encoding="utf-8") as f:
                f.seek(last_position)
                new_lines = f.readlines()
                last_position = f.tell()

            for line in new_lines:
                if "Symbolic Expression:" in line:
                    symbolic_content = line.split("Symbolic Expression:", 1)[1].strip()

                    numeric_values = {}
                    numeric_matches = re.findall(r'(\w+)=(\d+\.?\d*)', line)
                    for key, value in numeric_matches:
                        try:
                            numeric_values[key] = float(value)
                        except ValueError:
                            pass

                    if symbolic_content:
                        batch_text.append(symbolic_content)
                        batch_numeric.append(numeric_values)
                        symbolic_expressions_count += 1
                        print(f"📝 Collected symbolic expression: \"{symbolic_content[:50]}...\"")

                        if numeric_values:
                            numeric_str = ", ".join([f"{k}={v}" for k, v in numeric_values.items()])
                            print(f"   📊 With numeric values: {numeric_str}")

            if len(batch_text) >= update_interval:
                batch_count += 1
                print(f"\n🔍 Processing batch {batch_count} with {len(batch_text)} symbolic expressions")

                text_data = " ".join(batch_text)
                inputs = tokenizer(text_data, return_tensors="pt", padding=True,
                                   truncation=True, max_length=512).to(DEVICE)

                numeric_features = extract_numeric_features(text_data).to(DEVICE)
                batch_size_actual = inputs['input_ids'].shape[0]
                if batch_size_actual > 1:
                    numeric_features = numeric_features.unsqueeze(0).expand(batch_size_actual, -1)

                outputs = model(inputs['input_ids'], inputs['attention_mask'], numeric_features)

                with torch.no_grad():
                    ref_embeds = sentence_model.encode(text_data, convert_to_tensor=True).to(DEVICE)

                if outputs.shape[0] != ref_embeds.shape[0]:
                    min_size = min(outputs.shape[0], ref_embeds.shape[0])
                    outputs = outputs[:min_size]
                    ref_embeds = ref_embeds[:min_size]

                target = torch.ones(outputs.shape[0]).to(DEVICE)
                loss = loss_function(outputs, ref_embeds, target)

                if torch.isnan(loss) or torch.isinf(loss):
                    print("⚠️ Warning: NaN/Inf loss detected. Skipping batch.")
                    batch_text = []
                    batch_numeric = []
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                numeric_score = calculate_numeric_integration_score(text_data, extract_numeric_features(text_data).tolist())
                numeric_integration_score_sum += numeric_score

                current_loss = loss.item()
                total_loss += current_loss
                training_metrics['recursive_losses'].append(current_loss)
                training_metrics['numeric_integration_scores'].append(numeric_score)
                training_metrics['timestamps'].append(time.time())

                print(f"📊 Batch {batch_count} Loss: {current_loss:.4f}, Numeric Integration: {numeric_score:.4f}")

                if batch_count % 5 == 0:
                    coherence = evaluate_semantic_coherence(model, batch_text[:5])
                    training_metrics['coherence_scores'].append(coherence)
                    print(f"🧠 Current semantic coherence: {coherence:.4f}")

                if batch_count % 5 == 0:
                    save_model(model, MODEL_PATH.replace('.pt', f'_sim_{batch_count}.pt'))
                    print(f"💾 Model checkpoint saved at batch {batch_count}")

                batch_text = []
                batch_numeric = []

        if len(batch_text) > 0:
            save_model(model, MODEL_PATH.replace('.pt', '_final.pt'))

        runtime_elapsed = time.time() - start_time
        avg_loss = total_loss / max(1, batch_count)
        avg_numeric_score = numeric_integration_score_sum / max(1, batch_count)

        print(f"\n✅ Integrated training complete after {runtime_elapsed:.1f} seconds")
        print(f"📊 Processed {symbolic_expressions_count} symbolic expressions in {batch_count} batches")
        print(f"📈 Average loss: {avg_loss:.4f}, Average numeric integration: {avg_numeric_score:.4f}")

    except Exception as e:
        print(f"❌ Error during integrated training: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if emile_process and emile_process.poll() is None:
            emile_process.terminate()
            print("🛑 Émile simulation terminated")

        if log_handle:
            log_handle.close()

        save_model(model, MODEL_PATH.replace('.pt', '_integrated_final.pt'))
        print("💾 Final integrated model saved")

        plot_training_metrics()

        return symbolic_expressions_count, batch_count, runtime_elapsed


def evaluate_semantic_coherence(model, query_texts, reference_texts=None):
    """
    Evaluates how close the model embeddings are to a "ground truth" (SentenceTransformer) reference.
    """
    model.eval()

    if reference_texts is None:
        reference_texts = list(training_data.values())

    query_embeddings = []
    for text in query_texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        numeric_features = extract_numeric_features(text).to(DEVICE)

        with torch.no_grad():
            output = model(inputs['input_ids'], inputs['attention_mask'], numeric_features)
        query_embeddings.append(output)

    if query_embeddings:
        query_embeddings = torch.cat(query_embeddings, dim=0)
    else:
        return 0.0

    reference_embeddings = sentence_model.encode(reference_texts, convert_to_tensor=True).to(DEVICE)

    similarities = []
    for q_emb in query_embeddings:
        q_emb = q_emb.unsqueeze(0)
        cos_sims = torch.nn.functional.cosine_similarity(q_emb, reference_embeddings)
        max_sim = torch.max(cos_sims).item()
        similarities.append(max_sim)

    return sum(similarities) / len(similarities) if similarities else 0.0


def plot_training_metrics():
    """
    Plot training metrics over time with numeric integration scores.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # Losses
    if training_metrics['initial_losses']:
        ax1.plot(range(len(training_metrics['initial_losses'])),
                 training_metrics['initial_losses'],
                 label='Initial Training Loss')

    if training_metrics['recursive_losses']:
        ax1.plot(range(len(training_metrics['recursive_losses'])),
                 training_metrics['recursive_losses'],
                 label='Recursive Training Loss')

    ax1.set_title('Loss over Training Iterations')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Coherence
    if training_metrics['coherence_scores']:
        ax2.plot(range(len(training_metrics['coherence_scores'])),
                 training_metrics['coherence_scores'],
                 label='Semantic Coherence',
                 color='green')
        ax2.set_title('Semantic Coherence over Training')
        ax2.set_xlabel('Evaluation Point')
        ax2.set_ylabel('Coherence Score')
        ax2.legend()
        ax2.grid(True)

    # Numeric Integration
    if training_metrics['numeric_integration_scores']:
        ax3.plot(range(len(training_metrics['numeric_integration_scores'])),
                 training_metrics['numeric_integration_scores'],
                 label='Numeric Integration',
                 color='purple')
        window_size = min(10, len(training_metrics['numeric_integration_scores']))
        if window_size > 1:
            rolling_avg = np.convolve(training_metrics['numeric_integration_scores'],
                                      np.ones(window_size)/window_size,
                                      mode='valid')
            ax3.plot(range(window_size-1, len(training_metrics['numeric_integration_scores'])),
                     rolling_avg,
                     label=f'{window_size}-point Moving Average',
                     color='blue',
                     linestyle='--')

        ax3.set_title('Numeric Integration over Training')
        ax3.set_xlabel('Batch')
        ax3.set_ylabel('Integration Score')
        ax3.legend()
        ax3.grid(True)

    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(LOG_DIR, f"training_metrics_{timestamp}.png")
    plt.savefig(filename)
    print(f"📊 Training metrics plot saved to {filename}")
    plt.close(fig)


def refine_semiotic_model(model, log_file, update_interval=10, batch_size=16, learning_rate=1e-5):
    """
    Optimized log-based training with improved monitoring and numeric integration
    (for existing log files).
    """
    print(f"Starting recursive training on {log_file}...")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_function = nn.CosineEmbeddingLoss()

    last_position = 0
    batch_text = []
    batch_numeric = []
    batch_count = 0
    total_loss = 0
    save_counter = 0
    numeric_integration_score_sum = 0.0

    model.train()

    def stream_log(filepath, start_pos=0):
        with open(filepath, 'r', encoding='utf-8') as f:
            f.seek(start_pos)
            while True:
                line = f.readline()
                if not line:
                    break
                yield line, f.tell()

    for line, last_position in stream_log(log_file, last_position):
        if "Symbolic Expression:" in line:
            symbolic_content = line.split("Symbolic Expression:", 1)[1].strip()
            numeric_values = {}
            numeric_matches = re.findall(r'(\w+)=(\d+\.?\d*)', line)
            for key, value in numeric_matches:
                try:
                    numeric_values[key] = float(value)
                except ValueError:
                    pass

            if symbolic_content:
                batch_text.append(symbolic_content)
                batch_numeric.append(numeric_values)

        if len(batch_text) >= update_interval:
            batch_count += 1
            save_counter += 1
            print(f"  Processing batch {batch_count} from log...")

            text_data = " ".join(batch_text)
            inputs = tokenizer(text_data, return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(DEVICE)

            numeric_features = extract_numeric_features(text_data).to(DEVICE)
            batch_size_actual = inputs['input_ids'].shape[0]
            if batch_size_actual > 1:
                numeric_features = numeric_features.unsqueeze(0).expand(batch_size_actual, -1)

            outputs = model(inputs['input_ids'], inputs['attention_mask'], numeric_features)

            with torch.no_grad():
                sentence_transformer_embeddings = sentence_model.encode(text_data, convert_to_tensor=True).to(DEVICE)

            if outputs.shape[0] != sentence_transformer_embeddings.shape[0]:
                min_size = min(outputs.shape[0], sentence_transformer_embeddings.shape[0])
                outputs = outputs[:min_size]
                sentence_transformer_embeddings = sentence_transformer_embeddings[:min_size]

            target = torch.ones(outputs.shape[0]).to(DEVICE)
            loss = loss_function(outputs, sentence_transformer_embeddings, target)

            if torch.isnan(loss) or torch.isinf(loss):
                print("    WARNING: NaN/Inf loss detected. Skipping batch.")
                batch_text = []
                batch_numeric = []
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            numeric_score = calculate_numeric_integration_score(text_data, extract_numeric_features(text_data).tolist())
            numeric_integration_score_sum += numeric_score

            current_loss = loss.item()
            training_metrics['recursive_losses'].append(current_loss)
            training_metrics['numeric_integration_scores'].append(numeric_score)
            training_metrics['timestamps'].append(time.time())

            print(f"    Batch Loss: {current_loss:.4f}, Numeric Integration: {numeric_score:.4f}")
            total_loss += current_loss
            batch_text = []
            batch_numeric = []

            if save_counter >= 5:
                save_model(model)
                print(f"    Model saved after {save_counter} batches")
                save_counter = 0

    if save_counter > 0:
        save_model(model)
        avg_loss = total_loss / max(1, batch_count)
        avg_numeric_score = numeric_integration_score_sum / max(1, batch_count)
        print(f"Final model saved with average loss: {avg_loss:.4f}, average numeric integration: {avg_numeric_score:.4f}")


def extract_numeric_properties_from_logs(log_file):
    """
    Extracts numeric properties from simulation logs for training data enrichment.
    """
    print(f"Extracting numeric properties from {log_file}...")

    numeric_series = {
        'surplus': [],
        'distinction': [],
        'coherence': [],
        'entropy': [],
        'dimensionality': [],
        'timestamps': []
    }

    symbolic_expressions = []
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        current_time = time.time()
        time_offset = 0

        for line in lines:
            timestamp_match = re.search(r'\[([\d\-\s:.]+)\]', line)
            if timestamp_match:
                try:
                    timestamp_str = timestamp_match.group(1)
                    for fmt in ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
                                '%H:%M:%S.%f', '%H:%M:%S']:
                        try:
                            dt = datetime.strptime(timestamp_str, fmt)
                            time_offset = (dt - datetime(1970, 1, 1)).total_seconds()
                            break
                        except ValueError:
                            continue
                except Exception:
                    time_offset += 0.1
            else:
                time_offset += 0.1

            timestamp = current_time - (len(lines) - lines.index(line)) * 0.1

            if "Symbolic Expression:" in line:
                expr = line.split("Symbolic Expression:", 1)[1].strip()
                symbolic_expressions.append({
                    'expression': expr,
                    'timestamp': timestamp
                })
                numeric_matches = re.findall(r'(\w+)=(\d+\.?\d*)', line)
                for key, value in numeric_matches:
                    try:
                        value = float(value)
                        if key in numeric_series:
                            numeric_series[key].append(value)
                            if len(numeric_series[key]) > len(numeric_series['timestamps']):
                                numeric_series['timestamps'].append(timestamp)
                    except ValueError:
                        pass

            for key in numeric_series.keys():
                if key == 'timestamps':
                    continue
                pattern = fr'{key}\s*[=:]\s*(\d+\.?\d*)'
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1))
                        numeric_series[key].append(value)
                        if len(numeric_series[key]) > len(numeric_series['timestamps']):
                            numeric_series['timestamps'].append(timestamp)
                    except ValueError:
                        pass

        total_values = sum(len(values) for k, values in numeric_series.items() if k != 'timestamps')
        print(f"Extracted {total_values} numeric values and {len(symbolic_expressions)} symbolic expressions")

        stats = {}
        for key, values in numeric_series.items():
            if key == 'timestamps' or not values:
                continue
            stats[key] = {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'std': np.std(values)
            }

        return {
            'time_series': numeric_series,
            'symbolic_expressions': symbolic_expressions,
            'stats': stats
        }

    except Exception as e:
        print(f"Error extracting numeric properties: {e}")
        import traceback
        traceback.print_exc()
        return {
            'time_series': numeric_series,
            'symbolic_expressions': symbolic_expressions,
            'stats': {},
            'error': str(e)
        }

def create_training_dataset_with_numeric_values(log_data, output_folder=None):
    """
    Creates an enhanced training dataset with numeric values from log data.
    """
    if output_folder is None:
        output_folder = CACHE_DIR

    os.makedirs(output_folder, exist_ok=True)

    expressions = log_data.get('symbolic_expressions', [])
    time_series = log_data.get('time_series', {})

    training_texts = []
    numeric_features = []

    for expr_data in expressions:
        expr = expr_data['expression']
        timestamp = expr_data['timestamp']

        closest_values = {}
        for key, values in time_series.items():
            if key == 'timestamps' or not values:
                continue
            timestamps = time_series['timestamps']
            if not timestamps:
                continue
            closest_idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - timestamp))
            if closest_idx < len(values):
                closest_values[key] = values[closest_idx]

        features = []
        for key in ['surplus', 'distinction', 'coherence', 'entropy', 'dimensionality']:
            if key in closest_values:
                features.append(closest_values[key])
            else:
                features.append(0.0)

        if features:  # If we have any
            for key in ['surplus', 'distinction', 'coherence']:
                if key in time_series and len(time_series[key]) > 5:
                    recent_values = time_series[key][-5:]
                    features.append(np.var(recent_values))
                else:
                    features.append(0.0)

        while len(features) < 8:
            features.append(0.0)

        training_texts.append(expr)
        numeric_features.append(features)

    if output_folder:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        text_file = os.path.join(output_folder, f'training_texts_{timestamp}.txt')
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(training_texts))

        features_file = os.path.join(output_folder, f'numeric_features_{timestamp}.npy')
        np.save(features_file, np.array(numeric_features))

        print(f"Saved {len(training_texts)} training examples to {output_folder}")

    return training_texts, numeric_features

# --- Improved Helper Functions ---

def save_model(model, path=MODEL_PATH):
    """Saves the trained model plus numeric config info."""
    try:
        numeric_enabled = hasattr(model, 'numeric_enabled') and model.numeric_enabled
        numeric_dim = model.fc1_numeric.in_features if numeric_enabled else 0

        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'hidden_dim': model.fc1.out_features,
                'numeric_dim': numeric_dim,
                'numeric_enabled': numeric_enabled,
                'output_dim': 256
            },
            'training_metrics': {
                'numeric_integration_scores': training_metrics.get('numeric_integration_scores', []),
                'coherence_scores': training_metrics.get('coherence_scores', [])
            },
            'timestamp': time.strftime("%Y%m%d-%H%M%S")
        }, path)
        print(f"Model saved to {path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        alt_path = os.path.join(os.path.dirname(path), f"backup_model_{time.strftime('%Y%m%d_%H%M%S')}.pt")
        try:
            torch.save(model.state_dict(), alt_path)
            print(f"Model saved to alternate location: {alt_path}")
        except Exception as e2:
            print(f"Could not save to alternate location either: {e2}")

def load_model(path=MODEL_PATH):
    """
    Loads a previously trained model with numeric integration support.
    """
    try:
        checkpoint = torch.load(path)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            config = checkpoint.get('model_config', {})
            hidden_dim = config.get('hidden_dim', 384)
            numeric_dim = config.get('numeric_dim', 0)
            numeric_enabled = config.get('numeric_enabled', False)

            print(f"Loading model with config: hidden_dim={hidden_dim}, numeric_dim={numeric_dim}, numeric_enabled={numeric_enabled}")

            model = SemioticExtractor(hidden_dim=hidden_dim, numeric_dim=numeric_dim).to(DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])

            if 'training_metrics' in checkpoint:
                metrics = checkpoint['training_metrics']
                if 'numeric_integration_scores' in metrics:
                    training_metrics['numeric_integration_scores'] = metrics['numeric_integration_scores']
                if 'coherence_scores' in metrics:
                    training_metrics['coherence_scores'] = metrics['coherence_scores']
        else:
            print("Loading legacy model format (state dict only)")
            model = SemioticExtractor().to(DEVICE)
            model.load_state_dict(checkpoint)

        model.eval()
        print(f"Model loaded from {path}")
        return model
    except Exception as e:
        print(f"Error loading model from {path}: {e}")
        backup_pattern = os.path.join(os.path.dirname(path), "backup_model_*.pt")
        backups = sorted(glob.glob(backup_pattern), reverse=True)

        if backups:
            print(f"Attempting to load latest backup: {backups[0]}")
            try:
                model = SemioticExtractor().to(DEVICE)
                model.load_state_dict(torch.load(backups[0]))
                model.eval()
                print(f"Successfully loaded backup from {backups[0]}")
                return model
            except Exception as e2:
                print(f"Could not load backup either: {e2}")

        print("Creating new model instead.")
        return SemioticExtractor().to(DEVICE)

# ================================
# 8. NUMERIC INTEGRATION TESTING
# ================================

def test_numeric_integration(model, test_expressions=None):
    """
    Test how the model handles numeric values in text.
    """
    model.eval()

    if test_expressions is None:
        test_expressions = [
            "Coherence stabilizes within ontological field [coherence=0.85].",
            "Flux dissolves across phase space [entropy=0.72].",
            "Distinction emerges through complexity [distinction=0.67, surplus=3.5].",
            "Increasing surplus=4.2 amplifies dimensional emergence.",
            "Equilibrium aligns with stability [threshold=2.8 normalized].",
            "Oscillating coherence=0.45 within attractor dynamics.",
            "Recursion stabilizes within feedback [dimensionality=4]."
        ]

    results = []

    for expr in test_expressions:
        inputs = tokenizer(expr, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(DEVICE)
        numeric_features = extract_numeric_features(expr).to(DEVICE)

        with torch.no_grad():
            output_with_numeric = model(inputs['input_ids'], inputs['attention_mask'], numeric_features)

        zero_numeric = torch.zeros_like(numeric_features).to(DEVICE)
        with torch.no_grad():
            output_without_numeric = model(inputs['input_ids'], inputs['attention_mask'], zero_numeric)

        reference_embedding = sentence_model.encode([expr], convert_to_tensor=True).to(DEVICE)

        sim_with_numeric_tensor = torch.nn.functional.cosine_similarity(
            output_with_numeric,  # shape (1, 384)
            reference_embedding,  # shape (1, 384)
            dim=1
        )
        sim_with_numeric = sim_with_numeric_tensor[0].item()

        sim_without_numeric_tensor = torch.nn.functional.cosine_similarity(
            output_without_numeric,  # shape (1, 384)
            reference_embedding,     # shape (1, 384)
            dim=1
        )
        sim_without_numeric = sim_without_numeric_tensor[0].item()

        numeric_impact = sim_with_numeric - sim_without_numeric

        integration_score = calculate_numeric_integration_score(expr, extract_numeric_features(expr).tolist())
        numeric_impact = sim_with_numeric - sim_without_numeric

        results.append({
            'expression': expr,
            'similarity_with_numeric': sim_with_numeric,
            'similarity_without_numeric': sim_without_numeric,
            'numeric_impact': numeric_impact,
            'integration_score': integration_score,
            'numeric_features': extract_numeric_features(expr).tolist()
        })

    avg_impact = sum(r['numeric_impact'] for r in results) / len(results)
    avg_integration = sum(r['integration_score'] for r in results) / len(results)

    print(f"\n==== Numeric Integration Test Results ====")
    print(f"Average numeric impact: {avg_impact:.4f}")
    print(f"Average integration score: {avg_integration:.4f}")
    print("\nDetailed results:")

    for i, r in enumerate(results):
        print(f"\n{i+1}. Expression: {r['expression']}")
        print(f"   Integration score: {r['integration_score']:.4f}")
        print(f"   Numeric impact: {r['numeric_impact']:.4f}")

    return {
        'results': results,
        'avg_impact': avg_impact,
        'avg_integration': avg_integration
    }

# ================================
# 9. MAIN EXECUTION
# ================================

if __name__ == "__main__":
    for directory in [CACHE_DIR, LOG_DIR]:
        os.makedirs(directory, exist_ok=True)

    print("Émile5 Semantic ML Trainer")
    print("=" * 40)
    print(f"Device: {DEVICE}")
    print(f"Model name: {MODEL_NAME}")
    print(f"Model path: {MODEL_PATH}")

    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model = load_model(MODEL_PATH)
    else:
        print("Creating new semantic model with numeric integration capabilities")
        model = SemioticExtractor(hidden_dim=384, numeric_dim=8).to(DEVICE)

    import sys
    run_simulation = "--sim" in sys.argv or "-s" in sys.argv
    run_training = "--train" in sys.argv or "-t" in sys.argv
    run_testing = "--test" in sys.argv or "-e" in sys.argv
    epochs = 5

    for i, arg in enumerate(sys.argv):
        if arg == "--epochs" or arg == "-e":
            if i + 1 < len(sys.argv):
                try:
                    epochs = int(sys.argv[i + 1])
                except ValueError:
                    pass

    if run_training:
        print(f"\nTraining semantic model for {epochs} epochs")
        if training_data:
            train_semiotic_model(model, training_data, epochs=epochs)
        else:
            print("❌ No training data found. Please add .txt files to the data directory.")

    if run_simulation:
        print("\nRunning integrated simulation and training")
        refine_semiotic_model_with_simulation(model)

    if run_testing or not (run_training or run_simulation):
        print("\nTesting numeric integration")
        test_numeric_integration(model)

    print("\nTraining complete!")
    print(f"Final model saved to {MODEL_PATH}")

    # Plot final training metrics
    plot_training_metrics()


def visualize_semantic_numeric_relationship(model, log_data, save_path=None):
    """
    Visualizes the relationship between symbolic expressions and numeric values.
    """
    model.eval()

    expressions = log_data.get('symbolic_expressions', [])
    time_series = log_data.get('time_series', {})

    if not expressions or not time_series:
        print("Insufficient data for visualization")
        return

    keys_to_plot = []
    for key in ['coherence', 'distinction', 'entropy', 'surplus']:
        if key in time_series and len(time_series[key]) >= 5:
            keys_to_plot.append(key)

    if not keys_to_plot:
        print("No numeric time series with sufficient data points found")
        return

    expression_texts = [e['expression'] for e in expressions]
    expression_embeddings = []

    for text in expression_texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(DEVICE)
        numeric_features = extract_numeric_features(text).to(DEVICE)

        with torch.no_grad():
            embedding = model(inputs['input_ids'], inputs['attention_mask'], numeric_features)
            expression_embeddings.append(embedding.cpu().numpy())

    expression_embeddings = np.vstack(expression_embeddings)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    embedding_2d = pca.fit_transform(expression_embeddings)

    fig, axes = plt.subplots(len(keys_to_plot), 1, figsize=(12, 4 * len(keys_to_plot)))
    if len(keys_to_plot) == 1:
        axes = [axes]

    for i, key in enumerate(keys_to_plot):
        ax = axes[i]
        values = time_series[key]
        timestamps = time_series['timestamps'][:len(values)]

        ax.plot(range(len(values)), values, 'b-', label=f'{key} values')

        for j, expr in enumerate(expressions):
            idx = find_closest_timestamp_index(expr['timestamp'], timestamps)
            if idx is not None and idx < len(values):
                ax.scatter(idx, values[idx], c='red', s=50, zorder=5)
                if j % max(1, len(expressions) // 5) == 0:
                    short_expr = expr['expression'][:30] + "..." if len(expr['expression'])>30 else expr['expression']
                    ax.annotate(short_expr, (idx, values[idx]),
                                textcoords="offset points",
                                xytext=(0, 10),
                                ha='center',
                                fontsize=8,
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_title(f'{key.capitalize()} Values Over Time with Expression Markers')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel(f'{key.capitalize()} Value')
        ax.grid(True)
        ax.legend()

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    timestamps = [e['timestamp'] for e in expressions]
    min_time, max_time = min(timestamps), max(timestamps)
    normalized_times = [(t - min_time) / (max_time - min_time) if max_time > min_time else 0.5 for t in timestamps]

    scatter = ax2.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                          c=normalized_times, cmap='viridis',
                          s=100, alpha=0.8)

    for i, expr in enumerate(expressions):
        if i % max(1, len(expressions) // 10) == 0:
            short_expr = expr['expression'][:20] + "..." if len(expr['expression'])>20 else expr['expression']
            ax2.annotate(short_expr, (embedding_2d[i, 0], embedding_2d[i, 1]),
                         textcoords="offset points",
                         xytext=(5, 5),
                         ha='left',
                         fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax2.set_title('Semantic Space of Expressions (PCA Projection)')
    ax2.set_xlabel('Principal Component 1')
    ax2.set_ylabel('Principal Component 2')
    ax2.grid(True)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Time (normalized)')
    plt.tight_layout()

    if save_path:
        fig_path = save_path.replace('.png', '_timeseries.png')
        fig2_path = save_path.replace('.png', '_semantic.png')
        fig.savefig(fig_path)
        fig2.savefig(fig2_path)
        print(f"Visualizations saved to {fig_path} and {fig2_path}")
    else:
        plt.show()

    plt.close(fig)
    plt.close(fig2)


def find_closest_timestamp_index(target, timestamps):
    if not timestamps:
        return None
    return min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - target))


def plot_numeric_integration_progress(training_metrics, save_path=None):
    """
    Plots the progress of numeric integration during training.
    """
    if 'numeric_integration_scores' not in training_metrics or not training_metrics['numeric_integration_scores']:
        print("No numeric integration scores available for plotting")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    scores = training_metrics['numeric_integration_scores']
    ax1.plot(range(len(scores)), scores, 'purple-', label='Numeric Integration')

    if len(scores) > 5:
        window_size = min(10, len(scores)//2)
        if window_size > 1:
            avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(range(window_size-1, len(scores)), avg, 'r--',
                     label=f'{window_size}-point Moving Average')

    ax1.set_title('Numeric Integration Progress')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Integration Score')
    ax1.grid(True)
    ax1.legend()

    if 'recursive_losses' in training_metrics and training_metrics['recursive_losses']:
        losses = training_metrics['recursive_losses']
        losses = losses[:len(scores)]

        ax2.scatter(losses, scores, alpha=0.6, c=range(len(scores)), cmap='viridis')
        ax2.set_title('Loss vs. Numeric Integration')
        ax2.set_xlabel('Loss')
        ax2.set_ylabel('Numeric Integration Score')
        ax2.grid(True)

        norm = plt.Normalize(0, len(scores))
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2)
        cbar.set_label('Training Step')

        from scipy import stats
        if len(losses) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(losses, scores)
            x_line = np.array([min(losses), max(losses)])
            y_line = slope * x_line + intercept
            ax2.plot(x_line, y_line, 'r--', label=f'r={r_value:.2f}')
            ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Numeric integration progress plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)

# Add to main script to use these functions
if __name__ == "__main__":
    if 'run_testing' in globals() and run_testing and log_paths:
        print("\nAnalyzing simulation logs for numeric values...")
        log_data = extract_numeric_properties_from_logs(log_paths[0])

        training_texts, numeric_features = create_training_dataset_with_numeric_values(log_data)

        print("\nVisualizing semantic-numeric relationship...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        viz_path = os.path.join(LOG_DIR, f"semantic_numeric_viz_{timestamp}.png")
        visualize_semantic_numeric_relationship(model, log_data, viz_path)

        num_progress_path = os.path.join(LOG_DIR, f"numeric_integration_progress_{timestamp}.png")
        plot_numeric_integration_progress(training_metrics, save_path=num_progress_path)
