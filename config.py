"""
config.py — Device detection and shared configuration.

Run this file directly to check your hardware:
    python config.py
"""
import torch


def get_device():
    """Return the best available compute device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'    # Apple Silicon GPU
    return 'cpu'


DEVICE = get_device()


if __name__ == '__main__':
    print(f'Using device : {DEVICE}')
    print(f'PyTorch      : {torch.__version__}')
    if DEVICE == 'cuda':
        print(f'GPU          : {torch.cuda.get_device_name(0)}')
        print(f'VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    elif DEVICE == 'cpu':
        print('No GPU found - training will run on CPU (slower but works fine)')