"""
model/gpt.py — The complete GPT language model.

Architecture summary:
  Token Embedding      — converts token IDs to dense vectors
  Position Embedding   — adds positional information
  N × TransformerBlock — processes context with attention + FFN
  LayerNorm            — final normalisation
  Linear (lm_head)     — projects to vocabulary logits

Default config  ≈ 10 M parameters, trains on a laptop in < 4 hours.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import TransformerBlock


class SimpleGPT(nn.Module):
    """
    Decoder-only GPT-style language model.

    Args:
        vocab_size  : number of unique tokens (characters or BPE subwords)
        n_embd      : embedding / hidden dimension (default 384)
        n_head      : number of attention heads   (default 6)
        n_layer     : number of Transformer blocks (default 6)
        block_size  : maximum context length in tokens (default 256)
        dropout     : dropout probability for regularisation (default 0.2)
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int     = 384,
        n_head: int     = 6,
        n_layer: int    = 6,
        block_size: int = 256,
        dropout: float  = 0.2,
    ):
        super().__init__()
        self.block_size = block_size

        # ── Embeddings ───────────────────────────────────────────────────────
        # Token embedding: maps each token ID to a learned vector
        self.token_embedding    = nn.Embedding(vocab_size, n_embd)
        # Position embedding: adds information about WHERE in the sequence each token is
        self.position_embedding = nn.Embedding(block_size, n_embd)

        # ── Transformer Blocks ───────────────────────────────────────────────
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])

        # ── Output head ──────────────────────────────────────────────────────
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head  = nn.Linear(n_embd, vocab_size)   # logits over vocabulary

        # ── Weight initialisation ────────────────────────────────────────────
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f'Model parameters: {n_params:,}')

    # ── Initialisation ───────────────────────────────────────────────────────
    def _init_weights(self, module: nn.Module) -> None:
        """Small random weights → stable training from the start."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ── Forward pass ─────────────────────────────────────────────────────────
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx     : (B, T) integer token IDs
            targets : (B, T) integer token IDs to compute cross-entropy loss against

        Returns:
            logits : (B, T, vocab_size) — unnormalised next-token scores
            loss   : scalar cross-entropy loss, or None if targets not provided
        """
        B, T = idx.shape
        assert T <= self.block_size, (
            f'Sequence length {T} exceeds block_size {self.block_size}'
        )

        # Token + position embeddings
        tok_emb = self.token_embedding(idx)                              # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb                                           # (B, T, n_embd)

        # Transformer blocks
        x = self.blocks(x)      # (B, T, n_embd)
        x = self.ln_final(x)    # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Loss (only when targets are provided — not during generation)
        loss = None
        if targets is not None:
            # Flatten to (B*T, vocab_size) and (B*T,) for cross_entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    # ── Text generation ───────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Auto-regressively generate max_new_tokens tokens.

        Args:
            idx            : (1, T) seed token IDs
            max_new_tokens : how many tokens to generate
            temperature    : > 1.0 → more random; < 1.0 → more focused
            top_k          : if set, only sample from the top-k most likely tokens

        Returns:
            (1, T + max_new_tokens) tensor of token IDs
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -self.block_size:]

            # Forward pass → logits for the last position
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # (1, vocab_size)

            # Optional top-k nucleus filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample one token
            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (1, 1)
            idx      = torch.cat([idx, idx_next], dim=1)        # (1, T+1)

        return idx