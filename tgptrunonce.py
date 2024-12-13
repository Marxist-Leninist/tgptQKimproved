import sys
import subprocess

# Try to import datasets, if fail, install it
try:
    import datasets
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    import datasets

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

###########################################
# Load the Wikitext-2 dataset (raw)
###########################################
print("Loading dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_text = "\n".join(dataset['train']['text'])
valid_text = "\n".join(dataset['validation']['text'])

# Basic whitespace tokenizer
train_tokens = train_text.strip().split()
valid_tokens = valid_text.strip().split()

# Build a minimal vocabulary
vocab = list(set(train_tokens))
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

def encode(tokens):
    return [word2idx[t] for t in tokens if t in word2idx]

train_ids = torch.tensor(encode(train_tokens), dtype=torch.long)
valid_ids = torch.tensor(encode(valid_tokens), dtype=torch.long)

print(f"Train tokens: {len(train_ids)}, Valid tokens: {len(valid_ids)}")

# Hyperparameters
d_model = 128
N_layers = 2
h = 4
d_ff = 256
r = 16
block_size = 32
batch_size = 16
lr = 1e-3
device = torch.device("cpu")

###########################################
# Data Loader Function
###########################################
def get_batch(data_ids, block_size, batch_size):
    n = data_ids.size(0) - block_size
    idx = torch.randint(low=0, high=n, size=(batch_size,))
    x = torch.stack([data_ids[i:i+block_size] for i in idx])
    y = torch.stack([data_ids[i+1:i+block_size+1] for i in idx])
    return x, y

###########################################
# Model Definitions
###########################################
class LowRankMultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, r):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.r = r

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.U = nn.Parameter(torch.randn(self.d_k, r))
        nn.init.orthogonal_(self.U)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, N, _ = x.size()
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, N, self.h, self.d_k).transpose(1, 2)
        K = K.view(B, N, self.h, self.d_k).transpose(1, 2)
        V = V.view(B, N, self.h, self.d_k).transpose(1, 2)

        Q_prime = torch.matmul(Q, self.U)  # [B,h,N,r]
        K_prime = torch.matmul(K, self.U)  # [B,h,N,r]

        attn_scores = torch.matmul(Q_prime, K_prime.transpose(-1, -2)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores + mask
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.W_o(out)
        return out

class LowRankTransformerBlock(nn.Module):
    def __init__(self, d_model, h, d_ff, r):
        super().__init__()
        self.attn = LowRankMultiHeadAttention(d_model, h, r)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, mask=None):
        attn_out = self.attn(x, mask=mask)
        x = x + attn_out
        x = self.ln1(x)
        ff_out = self.ff(x)
        x = x + ff_out
        x = self.ln2(x)
        return x

class LowRankGPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, N_layers, h, d_ff, r, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([LowRankTransformerBlock(d_model, h, d_ff, r) for _ in range(N_layers)])
        self.ln_out = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, input_ids):
        B, N = input_ids.size()
        positions = torch.arange(N, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(positions)

        # Causal mask
        mask = torch.full((1, 1, N, N), float('-inf'), device=input_ids.device)
        mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.ln_out(x)
        logits = self.out_proj(x)  # [B, N, vocab_size]
        return logits

###########################################
# Training Loop (Short demo)
###########################################
model = LowRankGPTModel(vocab_size, d_model, N_layers, h, d_ff, r).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()
for step in range(50):
    x_batch, y_batch = get_batch(train_ids, block_size, batch_size)
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    logits = model(x_batch)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y_batch.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

print("Training complete. If you see loss values, it ran successfully!")
