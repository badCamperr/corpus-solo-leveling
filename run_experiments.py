import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import random
import numpy as np

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch version:", torch.__version__)
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# Import Block from transformer_blocks.py
from transformer_blocks import Block

# -------------------------------------------------------------
# 1. Custom Character-level Tokenizer
# -------------------------------------------------------------
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        
    def encode(self, s):
        # Skip characters not in vocabulary
        return [self.stoi[c] for c in s if c in self.stoi]
        
    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids])
        
    def get_piece_size(self):
        return self.vocab_size

# -------------------------------------------------------------
# 2. Tokenizer Wrapper to Unify SentencePiece & Character-Level
# -------------------------------------------------------------
class TokenizerWrapper:
    def __init__(self, name, tokenizer_obj, is_spm=True):
        self.name = name
        self.tokenizer = tokenizer_obj
        self.is_spm = is_spm
        
    def encode(self, text):
        if self.is_spm:
            return self.tokenizer.encode(text, out_type=int)
        else:
            return self.tokenizer.encode(text)
            
    def decode(self, ids):
        if self.is_spm:
            return self.tokenizer.decode(ids)
        else:
            return self.tokenizer.decode(ids)
            
    def get_vocab_size(self):
        if self.is_spm:
            return self.tokenizer.get_piece_size()
        else:
            return self.tokenizer.vocab_size

# -------------------------------------------------------------
# 3. Model Definition (TinyGPT)
# -------------------------------------------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, block_size=6, n_heads=2, n_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.block_size = block_size
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim) 
        self.position_embedding = nn.Embedding(block_size, embedding_dim) 
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]) 
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size) 

    def forward(self, idx, targets=None):
        B, T = idx.shape 
        tok_emb = self.token_embedding(idx) 
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb  
        x = self.blocks(x) 
        x = self.ln_f(x)
        logits = self.head(x) 
        loss = None
        if targets is not None:
            B, T, C = logits.shape 
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T)) 
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

# -------------------------------------------------------------
# Helper function to train SentencePiece models safely
# -------------------------------------------------------------
def train_sentencepiece_safe(input_file, model_prefix, vocab_size, model_type):
    try:
        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            byte_fallback=False,
            character_coverage=0.9995
        )
    except Exception as e:
        print(f"Warning: SentencePiece training failed for {model_prefix} with vocab_size={vocab_size}: {e}")
        # Fallback to a slightly larger vocab size because of vocabulary size requirements
        fallback_vocab = 90
        print(f"Retrying with fallback vocab_size={fallback_vocab}...")
        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=fallback_vocab,
            model_type=model_type,
            byte_fallback=False,
            character_coverage=0.9995
        )

# -------------------------------------------------------------
# Training and Evaluation Harness
# -------------------------------------------------------------
def run_experiment(name, tokenizer_wrapper, raw_text, epochs=2000, batch_size=16, lr=1e-3, block_size=6):
    print(f"\n--- Running Experiment: {name} ---")
    
    # 1. Encode Data
    encoded_ids = tokenizer_wrapper.encode(raw_text)
    data = torch.tensor(encoded_ids, dtype=torch.long)
    vocab_size = tokenizer_wrapper.get_vocab_size()
    total_tokens = len(data)
    compression = len(raw_text) / total_tokens
    
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Total Corpus Tokens: {total_tokens}")
    print(f"Compression Ratio (Chars/Token): {compression:.4f}")
    
    # 2. Setup Data Loader
    def get_batch():
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    # 3. Instantiate Model and Optimizer
    model = TinyGPT(vocab_size=vocab_size, block_size=block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # 4. Training loop
    start_time = time.time()
    losses = []
    
    for step in range(epochs):
        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 500 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f}")
        
        if step == epochs - 1 or step % 100 == 0:
            losses.append((step, loss.item()))
            
    training_time = time.time() - start_time
    final_loss = loss.item()
    print(f"Finished Training in {training_time:.2f} seconds. Final loss: {final_loss:.4f}")
    
    # 5. Generation Test
    prompt = "Sung Jin-woo"
    context_ids = tokenizer_wrapper.encode(prompt)
    if len(context_ids) == 0:
        # fallback prompt if empty encoding
        prompt = "Sung"
        context_ids = tokenizer_wrapper.encode(prompt)
    
    # Crop context to fit block_size
    context_ids = context_ids[-block_size:]
    x_prompt = torch.tensor([context_ids], dtype=torch.long, device=device)
    
    # Generate 100 tokens
    model.eval()
    with torch.no_grad():
        out = model.generate(x_prompt, max_new_tokens=100)
    
    generated_ids = out[0].tolist()
    generated_text = tokenizer_wrapper.decode(generated_ids)
    
    print("\nSample Generation:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)
    
    return {
        "name": name,
        "vocab_size": vocab_size,
        "total_tokens": total_tokens,
        "compression_ratio": compression,
        "final_loss": final_loss,
        "training_time_sec": training_time,
        "generated_text": generated_text,
        "losses": losses
    }

# -------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------
def main():
    # Load corpus
    with open("corpus.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Pre-train SentencePiece Tokenizers
    print("Training SentencePiece Tokenizers...")
    train_sentencepiece_safe("corpus.txt", "sp_bpe_80", 80, "bpe")
    train_sentencepiece_safe("corpus.txt", "sp_bpe_300", 300, "bpe")
    train_sentencepiece_safe("corpus.txt", "sp_uni_200", 200, "unigram")
    
    # Load all models into processors
    sp_bpe_80 = spm.SentencePieceProcessor()
    sp_bpe_80.load("sp_bpe_80.model")
    
    sp_bpe_300 = spm.SentencePieceProcessor()
    sp_bpe_300.load("sp_bpe_300.model")
    
    sp_uni_200 = spm.SentencePieceProcessor()
    sp_uni_200.load("sp_uni_200.model")
    
    # Instantiate custom CharTokenizer
    char_tokenizer = CharTokenizer(raw_text)
    
    # Wrap tokenizers
    tokenizers = [
        TokenizerWrapper("Character-Level Custom", char_tokenizer, is_spm=False),
        TokenizerWrapper("SentencePiece BPE (Vocab=80)", sp_bpe_80, is_spm=True),
        TokenizerWrapper("SentencePiece BPE (Vocab=300)", sp_bpe_300, is_spm=True),
        TokenizerWrapper("SentencePiece Unigram (Vocab=200)", sp_uni_200, is_spm=True)
    ]
    
    results = []
    
    # Run experiments
    for tok in tokenizers:
        res = run_experiment(tok.name, tok, raw_text, epochs=2000)
        results.append(res)
        
    # Output comparison Markdown table
    print("\n" + "="*80)
    print("FINAL EXPERIMENT RESULTS")
    print("="*80)
    
    print("| Tokenizer Approach | Vocab Size | Total Tokens | Compression Ratio | Final Loss | Training Time (s) |")
    print("|--------------------|------------|--------------|-------------------|------------|-------------------|")
    for r in results:
        print(f"| {r['name']:<30} | {r['vocab_size']:<10} | {r['total_tokens']:<12} | {r['compression_ratio']:<17.4f} | {r['final_loss']:<10.4f} | {r['training_time_sec']:<17.2f} |")
    
    # Save a detailed results report to file
    with open("experiment_summary.txt", "w", encoding="utf-8") as f:
        f.write("=== SUMMARY OF EXPERIMENTS ===\n\n")
        f.write("| Tokenizer Approach | Vocab Size | Total Tokens | Compression Ratio | Final Loss | Training Time (s) |\n")
        f.write("|--------------------|------------|--------------|-------------------|------------|-------------------|\n")
        for r in results:
            f.write(f"| {r['name']:<30} | {r['vocab_size']:<10} | {r['total_tokens']:<12} | {r['compression_ratio']:<17.4f} | {r['final_loss']:<10.4f} | {r['training_time_sec']:<17.2f} |\n")
        
        f.write("\n\n=== GENERATION SAMPLES ===\n\n")
        for r in results:
            f.write(f"--- {r['name']} ---\n")
            f.write(f"Generated text:\n{r['generated_text']}\n")
            f.write("-" * 40 + "\n\n")

if __name__ == "__main__":
    main()
