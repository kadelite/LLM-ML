"""
model/attention.py — Building blocks of the Transformer.

Hierarchy (bottom → top):
  Head              — one self-attention head
  MultiHeadAttention — N heads concatenated
  FeedForward        — two-layer MLP applied to each token
  TransformerBlock   — Attention + FeedForward with residuals & LayerNorm
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Single Self-Attention Head
# ─────────────────────────────────────────────────────────────────────────────
class Head(nn.Module):
    """
    One causal (masked) self-attention head.

    Each token creates three vectors:
      Query (q) — "what am I looking for?"
      Key   (k) — "what do I contain?"
      Value (v) — "what do I communicate if attended to?"

    Attention score between positions i and j  =  q_i · k_j / sqrt(head_size)
    The causal mask sets future positions to -inf so softmax gives them 0 weight.
    """

    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Lower-triangular mask — ones at positions the model IS allowed to see.
        # Registered as a buffer so it moves to the correct device with the model.
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embedding dim

        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Scaled dot-product attention scores
        scale  = math.sqrt(k.shape[-1])
        scores = q @ k.transpose(-2, -1) / scale   # (B, T, T)

        # Mask future positions (autoregressive / causal)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Softmax → attention weights → weighted sum of values
        weights = F.softmax(scores, dim=-1)         # (B, T, T)
        weights = self.dropout(weights)
        out     = weights @ v                       # (B, T, head_size)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Head Attention
# ─────────────────────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """
    Run num_heads attention heads in parallel, then concatenate and project.

    Different heads can specialise in different relationships:
      head 1 — syntactic agreement
      head 2 — semantic similarity
      head 3 — positional proximity
      ... etc.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        n_embd: int,
        block_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, n_embd, block_size, dropout)
            for _ in range(num_heads)
        ])
        # Project concatenated heads back to n_embd
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        out = self.dropout(self.proj(out))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────
class FeedForward(nn.Module):
    """
    Position-wise two-layer MLP applied identically to each token.

    The inner dimension is 4× the embedding size — this matches the original
    "Attention Is All You Need" paper and scales well in practice.
    GELU activation is smoother than ReLU and used in most modern LLMs.
    """

    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    """
    One Transformer layer = MultiHeadAttention + FeedForward.

    Uses Pre-LN (LayerNorm applied BEFORE each sub-layer) which trains more
    stably than the original Post-LN described in the 2017 paper.

    Residual connections  x = x + sublayer(norm(x))  allow gradients to flow
    cleanly through many stacked layers without vanishing.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        head_size = n_embd // n_head
        self.attention    = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.feed_forward = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))      # "communicate"
        x = x + self.feed_forward(self.ln2(x))   # "think"
        return x