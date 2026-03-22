"""
generate.py — Load a trained model and generate text from a prompt.

Usage:
    python generate.py
    python generate.py --prompt "ROMEO:" --tokens 300 --temp 0.8 --topk 40

Make sure you have already run:
    python train.py
"""
import argparse
import json
import torch

from model.gpt import SimpleGPT
from config import get_device


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_vocab(path: str = 'data/vocab.json') -> tuple[dict, dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    stoi = data['stoi']
    itos = {int(k): v for k, v in data['itos'].items()}
    return stoi, itos


def load_model(checkpoint_path: str, device: str) -> SimpleGPT:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg  = ckpt['config']
    model = SimpleGPT(
        vocab_size  = ckpt['vocab_size'],
        n_embd      = cfg['n_embd'],
        n_head      = cfg['n_head'],
        n_layer     = cfg['n_layer'],
        block_size  = cfg['block_size'],
        dropout     = cfg['dropout'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def generate_text(
    model: SimpleGPT,
    stoi: dict,
    itos: dict,
    prompt: str = '',
    max_tokens: int = 500,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = 'cpu',
) -> str:
    """Generate text continuing from prompt."""
    if prompt:
        ids = [stoi[c] for c in prompt if c in stoi]
        context = torch.tensor([ids], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    output_ids = model.generate(context, max_tokens, temperature, top_k)
    return ''.join(itos[i] for i in output_ids[0].tolist())


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description='Generate text with SimpleGPT')
    parser.add_argument('--checkpoint', default='model/simple_gpt.pt')
    parser.add_argument('--vocab',      default='data/vocab.json')
    parser.add_argument('--prompt',     default='',       help='Seed text')
    parser.add_argument('--tokens',     default=500, type=int)
    parser.add_argument('--temp',       default=0.8, type=float,
                        help='Temperature: higher = more creative (default 0.8)')
    parser.add_argument('--topk',       default=40,  type=int,
                        help='Top-k sampling (default 40, 0 = disabled)')
    args = parser.parse_args()

    device = get_device()
    print(f'Device: {device}')

    stoi, itos = load_vocab(args.vocab)
    model      = load_model(args.checkpoint, device)

    top_k = args.topk if args.topk > 0 else None
    text  = generate_text(
        model, stoi, itos,
        prompt      = args.prompt,
        max_tokens  = args.tokens,
        temperature = args.temp,
        top_k       = top_k,
        device      = device,
    )

    print('\n' + '=' * 60)
    print(text)
    print('=' * 60)


if __name__ == '__main__':
    main()