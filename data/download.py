"""
data/download.py — Download training text data.

Usage:
    python data/download.py

Downloads Tiny Shakespeare (~1 MB) by default — perfect for a first run.
Edit DATASET_URL to use any plain-text URL instead.
"""
import os
import requests

# Tiny Shakespeare: ~1 MB, clean prose, fast to train on
DATASET_URL = (
    'https://raw.githubusercontent.com/karpathy/char-rnn/'
    'master/data/tinyshakespeare/input.txt'
)
OUTPUT_PATH = 'data/input.txt'


def download(url: str = DATASET_URL, dest: str = OUTPUT_PATH) -> None:
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        print(f'File already exists: {dest}  ({size:,} bytes) - skipping download.')
        return

    print(f'Downloading from:\n  {url}')
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'w', encoding='utf-8') as f:
        f.write(response.text)

    print(f'Saved {len(response.text):,} characters to {dest}')
    print(f'Preview:\n{"-"*40}\n{response.text[:300]}\n{"-"*40}')


if __name__ == '__main__':
    download()