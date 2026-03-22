"""
data/prepare.py — Tokenise raw text and split into train / val sets.

Usage:
    python data/prepare.py

Reads  : data/input.txt
Writes : data/train.pt, data/val.pt, data/vocab.json

Uses character-level tokenisation (simplest possible approach):
  • Every unique character becomes one token.
  • Vocabulary is ~65 characters for English text.
  • No library dependencies — totally self-contained.
"""
import os
import json
import torch


INPUT_PATH = 'data/input.txt'
VOCAB_PATH  = 'data/vocab.json'
TRAIN_PATH  = 'data/train.pt'
VAL_PATH    = 'data/val.pt'

TRAIN_RATIO = 0.9   # 90 % training, 10 % validation


def build_vocab(text: str) -> tuple[dict, dict]:
    """Return (stoi, itos) mappings for every unique character in text."""
    chars = sorted(set(text))
    stoi  = {ch: i for i, ch in enumerate(chars)}
    itos  = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode(text: str, stoi: dict) -> list[int]:
    return [stoi[c] for c in text]


def decode(ids: list[int], itos: dict) -> str:
    return ''.join(itos[i] for i in ids)


def prepare(input_path: str = INPUT_PATH) -> None:
    # ── 1. Read raw text ────────────────────────────────────────────────────
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f'{input_path} not found.\n'
            'Run  python data/download.py  first.'
        )

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f'Total characters : {len(text):,}')

    # ── 2. Build vocabulary ─────────────────────────────────────────────────
    stoi, itos = build_vocab(text)
    vocab_size  = len(stoi)
    print(f'Vocabulary size  : {vocab_size}')
    print(f'Characters       : {"".join(stoi.keys())}')

    # Save vocab so generate.py / chat.py can reload it later
    with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump({'stoi': stoi, 'itos': {str(k): v for k, v in itos.items()}}, f)
    print(f'Vocabulary saved : {VOCAB_PATH}')

    # ── 3. Encode entire dataset ─────────────────────────────────────────────
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    n    = int(TRAIN_RATIO * len(data))
    train_data = data[:n]
    val_data   = data[n:]

    print(f'Train tokens     : {len(train_data):,}')
    print(f'Val   tokens     : {len(val_data):,}')

    # ── 4. Quick sanity check ────────────────────────────────────────────────
    sample      = text[:12]
    encoded     = encode(sample, stoi)
    decoded     = decode(encoded, itos)
    assert decoded == sample, 'Round-trip encode→decode failed!'
    print(f'Encode/decode OK : "{sample}" -> {encoded} -> "{decoded}"')

    # ── 5. Save tensors ──────────────────────────────────────────────────────
    torch.save(train_data, TRAIN_PATH)
    torch.save(val_data,   VAL_PATH)
    print(f'Saved: {TRAIN_PATH}, {VAL_PATH}')


if __name__ == '__main__':
    prepare()