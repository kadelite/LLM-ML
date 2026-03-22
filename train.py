"""
train.py — Train the SimpleGPT model.

Usage:
    python train.py

Before running make sure you have:
  1. python data/download.py   (get the text)
  2. python data/prepare.py    (tokenise it)

The trained checkpoint is saved to model/simple_gpt.pt when done.
"""
import math
import time
import torch

from config import get_device
from model.gpt import SimpleGPT
from data.loader import get_batch

# -----------------------------------------------------------------------------
# Configuration
#
# CPU-friendly defaults (~1 hour, produces readable English):
#   n_embd=192, n_head=4, n_layer=4, block_size=128, batch_size=32
#
# GPU / full 10M model (~1-2 hours on GPU, ~12 hours on CPU):
#   n_embd=384, n_head=6, n_layer=6, block_size=256, batch_size=64
# -----------------------------------------------------------------------------
CONFIG = {
    # Model architecture
    'n_embd'     : 192,    # Embedding dimension  (must be divisible by n_head)
    'n_head'     : 4,      # Attention heads
    'n_layer'    : 4,      # Transformer blocks
    'block_size' : 128,    # Context length (tokens)
    'dropout'    : 0.1,    # Regularisation dropout

    # Training
    'batch_size'    : 32,    # Sequences per step
    'max_steps'     : 5000,  # Training iterations (~1 hr on CPU, good English output)
    'eval_interval' : 500,   # Print loss every N steps
    'eval_iters'    : 100,   # Batches to average for loss estimate

    # Learning rate schedule (cosine decay with linear warmup)
    'learning_rate' : 3e-4,
    'min_lr'        : 3e-5,
    'warmup_steps'  : 200,

    # Gradient clipping (prevents exploding gradients)
    'grad_clip'     : 1.0,
}


# -----------------------------------------------------------------------------
# Learning rate schedule
# -----------------------------------------------------------------------------
def get_lr(step: int) -> float:
    max_lr       = CONFIG['learning_rate']
    min_lr       = CONFIG['min_lr']
    warmup_steps = CONFIG['warmup_steps']
    max_steps    = CONFIG['max_steps']

    # 1. Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # 2. Cosine decay
    decay_ratio = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# -----------------------------------------------------------------------------
# Loss estimation
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(
    model: SimpleGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    device: str,
) -> dict[str, float]:
    """Average loss over eval_iters batches for a stable reading."""
    model.eval()
    results = {}
    for split in ('train', 'val'):
        losses = [
            model(
                *[t for t in get_batch(
                    split, train_data, val_data,
                    CONFIG['block_size'], CONFIG['batch_size'], device
                )]
            )[1].item()
            for _ in range(CONFIG['eval_iters'])
        ]
        results[split] = sum(losses) / len(losses)
    model.train()
    return results


# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------
def train() -> SimpleGPT:
    device = get_device()
    print(f'Training on : {device}\n')

    # Load tokenised data
    train_data = torch.load('data/train.pt', weights_only=True)
    val_data   = torch.load('data/val.pt',   weights_only=True)

    vocab_size = int(max(train_data.max(), val_data.max()).item()) + 1
    print(f'Vocab size  : {vocab_size}')

    # Build model
    model = SimpleGPT(
        vocab_size  = vocab_size,
        n_embd      = CONFIG['n_embd'],
        n_head      = CONFIG['n_head'],
        n_layer     = CONFIG['n_layer'],
        block_size  = CONFIG['block_size'],
        dropout     = CONFIG['dropout'],
    ).to(device)

    # Optimiser — AdamW is the standard choice for Transformers
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=0.1,
    )

    # Training
    print('\nStarting training ...\n')
    print(f'{"Step":>6}  {"Train loss":>10}  {"Val loss":>10}  {"LR":>8}  {"Time":>6}')
    print('-' * 55)

    start = time.time()
    train_losses = []
    val_losses   = []

    for step in range(CONFIG['max_steps'] + 1):

        # -- Evaluate -------------------------------------------------------
        if step % CONFIG['eval_interval'] == 0:
            losses  = estimate_loss(model, train_data, val_data, device)
            elapsed = time.time() - start
            lr_now  = get_lr(step)
            print(
                f'{step:>6}  '
                f'{losses["train"]:>10.4f}  '
                f'{losses["val"]:>10.4f}  '
                f'{lr_now:>8.2e}  '
                f'{elapsed:>5.0f}s'
            )
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])

        if step == CONFIG['max_steps']:
            break   # Evaluation-only on final step

        # -- Forward + backward ---------------------------------------------
        x, y = get_batch(
            'train', train_data, val_data,
            CONFIG['block_size'], CONFIG['batch_size'], device,
        )
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])

        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(step)

        optimizer.step()

    # -- Save checkpoint ----------------------------------------------------
    import os
    os.makedirs('model', exist_ok=True)
    checkpoint_path = 'model/simple_gpt.pt'
    torch.save(
        {
            'model_state_dict' : model.state_dict(),
            'config'           : CONFIG,
            'vocab_size'       : vocab_size,
        },
        checkpoint_path,
    )
    print(f'\nModel saved ->{checkpoint_path}')

    # -- Optional loss plot ------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        steps = list(range(0, CONFIG['max_steps'] + 1, CONFIG['eval_interval']))
        plt.figure(figsize=(8, 4))
        plt.plot(steps, train_losses, label='Train')
        plt.plot(steps, val_losses,   label='Val')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('model/loss_curve.png', dpi=120)
        print('Loss curve  -> model/loss_curve.png')
    except ImportError:
        pass

    return model


if __name__ == '__main__':
    train()