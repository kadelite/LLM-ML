"""
chat.py - Interactive chat interface for SimpleGPT.

Usage:
    python chat.py
    python chat.py --temp 0.9 --topk 50 --tokens 200

How it works:
  • You type a prompt (any text).
  • The model continues your text for `--tokens` characters.
  • This is a generative language model, not a question-answering model -
    it produces text that *follows* your prompt in the style of its training data.
  • For Shakespeare-trained model, try prompts like:
      "HAMLET:" or "To be," or "JULIET:\nO,"

Type 'quit' or 'exit' to leave.
Type 'help' for tips.
"""
import argparse
import json
import sys
import torch

from model.gpt import SimpleGPT
from config import get_device


HELP_TEXT = """
-----------------------------------------------------------------
 SimpleGPT Chat - Tips
-----------------------------------------------------------------
 This model CONTINUES your text - it's not a Q&A chatbot.
 The output will be in the style of the training data.

 For a Shakespeare-trained model, try:
   -> ROMEO:
   -> HAMLET:\nTo be or not to be,
   -> KING LEAR:\n

 Commands:
   help   - show this message
   quit   - exit
   exit   - exit
   /temp  0.8   - change temperature on the fly
   /topk  40    - change top-k sampling on the fly
   /len   300   - change output length on the fly
-----------------------------------------------------------------
"""


def load_vocab(path: str = 'data/vocab.json') -> tuple[dict, dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    stoi = data['stoi']
    itos = {int(k): v for k, v in data['itos'].items()}
    return stoi, itos


def load_model(checkpoint_path: str, device: str) -> SimpleGPT:
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg   = ckpt['config']
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


def generate(
    model: SimpleGPT,
    stoi: dict,
    itos: dict,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int | None,
    device: str,
) -> str:
    # Filter out unknown characters silently
    ids = [stoi[c] for c in prompt if c in stoi]
    if not ids:
        ids = [0]   # fallback to first token if prompt has no known chars

    context    = torch.tensor([ids], dtype=torch.long, device=device)
    output_ids = model.generate(context, max_tokens, temperature, top_k)

    # Return only the newly generated portion (strip the prompt)
    generated_ids = output_ids[0].tolist()[len(ids):]
    return ''.join(itos.get(i, '?') for i in generated_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description='Chat with SimpleGPT')
    parser.add_argument('--checkpoint', default='model/simple_gpt.pt')
    parser.add_argument('--vocab',      default='data/vocab.json')
    parser.add_argument('--temp',       default=0.8,  type=float)
    parser.add_argument('--topk',       default=40,   type=int)
    parser.add_argument('--tokens',     default=300,  type=int)
    args = parser.parse_args()

    device = get_device()

    # Load model
    print('Loading model ...', end=' ', flush=True)
    try:
        stoi, itos = load_vocab(args.vocab)
        model      = load_model(args.checkpoint, device)
    except FileNotFoundError as e:
        print(f'\nError: {e}')
        print('Have you run:  python train.py  ?')
        sys.exit(1)
    print('done.\n')

    # Session state (mutable so /temp /topk /len can change them)
    state = {
        'temperature' : args.temp,
        'top_k'       : args.topk if args.topk > 0 else None,
        'max_tokens'  : args.tokens,
    }

    print('=' * 60)
    print(' SimpleGPT Chat')
    print(f' Device: {device}  |  Temperature: {state["temperature"]}  |  '
          f'Top-k: {state["top_k"]}  |  Tokens: {state["max_tokens"]}')
    print(' Type "help" for tips, "quit" to exit.')
    print('=' * 60 + '\n')

    while True:
        try:
            user_input = input('You: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nGoodbye!')
            break

        if not user_input:
            continue

        lower = user_input.lower()

        # -- Built-in commands ----------------------------------------------
        if lower in ('quit', 'exit'):
            print('Goodbye!')
            break

        if lower == 'help':
            print(HELP_TEXT)
            continue

        if lower.startswith('/temp '):
            try:
                state['temperature'] = float(lower.split()[1])
                print(f'Temperature set to {state["temperature"]}')
            except ValueError:
                print('Usage: /temp 0.8')
            continue

        if lower.startswith('/topk '):
            try:
                val = int(lower.split()[1])
                state['top_k'] = val if val > 0 else None
                print(f'Top-k set to {state["top_k"]}')
            except ValueError:
                print('Usage: /topk 40')
            continue

        if lower.startswith('/len '):
            try:
                state['max_tokens'] = int(lower.split()[1])
                print(f'Output length set to {state["max_tokens"]} tokens')
            except ValueError:
                print('Usage: /len 300')
            continue

        # -- Generate ------------------------------------------------------
        print('\nModel:', end=' ', flush=True)
        response = generate(
            model, stoi, itos,
            prompt      = user_input,
            max_tokens  = state['max_tokens'],
            temperature = state['temperature'],
            top_k       = state['top_k'],
            device      = device,
        )
        print(response)
        print()


if __name__ == '__main__':
    main()