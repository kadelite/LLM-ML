"""
data/loader.py — Random batch generator for training.

The model learns to predict the NEXT token at every position, so:
  x (input)  = tokens[i   : i+block_size]
  y (target) = tokens[i+1 : i+block_size+1]

Example with block_size=5 on the text "Hello":
  x = [H, e, l, l, o]
  y = [e, l, l, o, !]

  The model sees "H"     and must predict "e"
  The model sees "He"    and must predict "l"
  The model sees "Hel"   and must predict "l"
  ... and so on — every context length from 1 to block_size.
"""
import torch


def get_batch(
    split: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random mini-batch from train or val data.

    Args:
        split      : 'train' or 'val'
        train_data : 1-D tensor of encoded training tokens
        val_data   : 1-D tensor of encoded validation tokens
        block_size : number of tokens the model sees at once (context window)
        batch_size : how many independent sequences per batch
        device     : 'cpu', 'cuda', or 'mps'

    Returns:
        x : (batch_size, block_size)  — input token IDs
        y : (batch_size, block_size)  — target token IDs (x shifted right by 1)
    """
    data = train_data if split == 'train' else val_data

    # Random starting positions (one per sequence in the batch)
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i     : i + block_size    ] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)