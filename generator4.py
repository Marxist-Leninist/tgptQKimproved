import torch
import torch.nn as nn
import torch.nn.functional as F

# Make sure these definitions match exactly what you used during training
# (Same vocabulary, same model architecture parameters)
d_model = 128
N_layers = 2
h = 4
d_ff = 256
r = 16
max_len = 512

# Load the vocabulary from the training script
# You must have `vocab`, `word2idx`, `idx2word` defined as before
# For this example, just copy and paste the same code you used in the training script.

###############################
# REPEAT VOCAB LOADING CODE HERE
###############################
import os
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_text = "\n".join(dataset['train']['text'])
train_tokens = train_text.strip().split()
vocab = list(set(train_tokens))
vocab_size = len(vocab)
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

def encode(tokens):
    return [word2idx[t] for t in tokens if t in word2idx]

def decode(indices):
    return [idx2word[i] for i in indices]

###############################
# DEFINE MODEL CLASSES SAME AS TRAINING
###############################
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

        Q_prime = torch.matmul(Q, self.U)
        K_prime = torch.matmul(K, self.U)

        import math
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

###############################
# LOAD THE MODEL
###############################
model = LowRankGPTModel(vocab_size, d_model, N_layers, h, d_ff, r, max_len=max_len)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

###############################
# GENERATION FUNCTION
###############################
def generate_tokens(model, prompt_tokens, max_new_tokens=20):
    model.eval()
    input_ids = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)
        # Get the last token's logits
        next_token_logits = logits[0, -1, :]
        # Sample from the distribution
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        # Append next token
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    return input_ids[0].tolist()

###############################
# INTERACTIVE PROMPT
###############################
prompt = input("Enter a prompt: ")
prompt_tokens = prompt.strip().split()
encoded_prompt = encode(prompt_tokens)
if len(encoded_prompt) == 0:
    print("No tokens found in the prompt that are in the vocabulary!")
else:
    out_tokens = generate_tokens(model, encoded_prompt, max_new_tokens=20)
    out_text = decode(out_tokens)
    print("Generated text:", " ".join(out_text))
