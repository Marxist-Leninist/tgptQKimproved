import torch
import torch.nn as nn
import torch.nn.functional as F

class LowRankMultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, r):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.r = r

        # Linear layers for Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Low-rank projection matrix U for queries and keys
        # We make U a learnable parameter. U should be of shape (d_k, r).
        self.U = nn.Parameter(torch.randn(self.d_k, r))
        nn.init.orthogonal_(self.U)  # ensure U is orthonormal

        # Final projection
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, d_model]
        B, N, _ = x.size()

        # Compute Q, K, V
        # Q, K, V are [B, N, d_model]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape Q, K, V to [B, h, N, d_k]
        Q = Q.view(B, N, self.h, self.d_k).transpose(1, 2)  # [B, h, N, d_k]
        K = K.view(B, N, self.h, self.d_k).transpose(1, 2)  # [B, h, N, d_k]
        V = V.view(B, N, self.h, self.d_k).transpose(1, 2)  # [B, h, N, d_k]

        # Apply low-rank projection U to Q and K: Q' = Q U, K' = K U
        # Q', K' will be [B, h, N, r]
        # U: [d_k, r]
        Q_prime = torch.matmul(Q, self.U)  # [B, h, N, r]
        K_prime = torch.matmul(K, self.U)  # [B, h, N, r]

        # Compute attention weights using Q'K'^T
        # Q'K'^T: [B, h, N, r] x [B, h, r, N] -> [B, h, N, N]
        attn_scores = torch.matmul(Q_prime, K_prime.transpose(-1, -2)) / (self.d_k ** 0.5)

        if mask is not None:
            # mask: [B, 1, N, N] or [1, 1, N, N]
            attn_scores = attn_scores + mask

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, h, N, N]

        # Compute output
        # out: [B, h, N, d_k]
        out = torch.matmul(attn_weights, V)
        # Reshape back to [B, N, d_model]
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
        # x: [B, N, d_model]
        # First, MHA with residual + LN
        attn_out = self.attn(x, mask=mask)
        x = x + attn_out
        x = self.ln1(x)

        # FFN with residual + LN
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
        positions = torch.arange(N, device=input_ids.device).unsqueeze(0)  # [1, N]

        x = self.embed(input_ids) + self.pos_embed(positions)

        # causal mask: shape [1, 1, N, N]
        # -inf for future tokens
        mask = torch.full((1, 1, N, N), float('-inf'), device=input_ids.device)
        mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.ln_out(x)
        logits = self.out_proj(x)  # [B, N, vocab_size]

        return logits


# Example usage:
# vocab_size = 30000
# model = LowRankGPTModel(vocab_size=vocab_size, d_model=512, N_layers=6, h=8, d_ff=2048, r=64)
# input_ids = torch.randint(0, vocab_size, (2, 50))  # batch=2, seq_len=50
# logits = model(input_ids)
# print(logits.shape)  # [2, 50, vocab_size]
