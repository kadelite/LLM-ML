# Simple LLM From Scratch

A ~10 million parameter GPT-style language model you can **build, train, and chat with on your own laptop** — no expensive cloud GPUs required.

---

## Table of Contents

1. [What You Are Building](#1-what-you-are-building)
2. [How Everything Fits Together](#2-how-everything-fits-together)
3. [Project Structure](#3-project-structure)
4. [Prerequisites](#4-prerequisites)
5. [Step-by-Step Setup](#5-step-by-step-setup)
6. [Understanding Each File](#6-understanding-each-file)
7. [Understanding the Model Architecture](#7-understanding-the-model-architecture)
8. [Training the Model](#8-training-the-model)
9. [Chatting With the Model](#9-chatting-with-the-model)
10. [Generating Text](#10-generating-text)
11. [Reading the Training Output](#11-reading-the-training-output)
12. [Troubleshooting](#12-troubleshooting)
13. [Improving the Model](#13-improving-the-model)
14. [Concepts Glossary](#14-concepts-glossary)

---

## 1. What You Are Building

You are building a **decoder-only Transformer language model** — the same fundamental architecture as GPT-2, GPT-3, and GPT-4, just much smaller.

| Property        | Value                          |
|-----------------|-------------------------------|
| Architecture    | GPT-style Transformer         |
| Parameters      | ~10 million                   |
| Layers          | 6 Transformer blocks          |
| Attention heads | 6 per block                   |
| Embedding dim   | 384                           |
| Context length  | 256 tokens                    |
| Vocabulary      | ~65 characters (character-level) |
| Training data   | Tiny Shakespeare (~1 MB)      |
| Training time   | 30 min – 4 hours on CPU       |

**What the model does:** Given a sequence of characters, it predicts what character comes next. By chaining these predictions together, it generates text in the style of what it was trained on.

---

## 2. How Everything Fits Together

```
Your text file (input.txt)
        │
        ▼
  data/download.py        ← downloads the text
        │
        ▼
  data/prepare.py         ← tokenises text → train.pt + val.pt + vocab.json
        │
        ▼
    train.py              ← builds model, runs training loop
        │
        ▼
  model/simple_gpt.pt     ← saved trained weights
        │
     ┌──┴──┐
     ▼     ▼
 chat.py  generate.py     ← talk to your model
```

The model learns by reading millions of overlapping windows of text, and at each position trying to predict the next character. After thousands of such updates, the model's internal weights encode the patterns of the language.

---

## 3. Project Structure

```
LLM-YT/
├── data/
│   ├── __init__.py         Python package marker
│   ├── download.py         Download training text from the internet
│   ├── prepare.py          Tokenise + split text into train/val tensors
│   ├── loader.py           Random mini-batch generator used during training
│   ├── input.txt           Raw training text (created by download.py)
│   ├── train.pt            Encoded training tokens (created by prepare.py)
│   ├── val.pt              Encoded validation tokens (created by prepare.py)
│   └── vocab.json          Character ↔ integer mappings (created by prepare.py)
│
├── model/
│   ├── __init__.py         Python package marker
│   ├── attention.py        Head, MultiHeadAttention, FeedForward, TransformerBlock
│   ├── gpt.py              Complete SimpleGPT model class
│   └── simple_gpt.pt       Saved model checkpoint (created by train.py)
│
├── config.py               Detects best device (CPU / CUDA / MPS)
├── train.py                Full training script with loss curve
├── generate.py             Command-line text generator
├── chat.py                 Interactive chat REPL
└── requirements.txt        Python dependencies
```

---

## 4. Prerequisites

**Python version:** 3.10 or newer (required for `X | None` type hints)

**Hardware:**
- Any laptop with 8 GB+ RAM
- A CUDA GPU will speed up training ~10×, but is not required
- Apple Silicon (M1/M2/M3) Macs benefit from Metal GPU acceleration automatically

**Knowledge:** Basic Python. You do not need to understand deep learning theory to follow this guide — concepts are explained as you encounter them.

---

## 5. Step-by-Step Setup

### Step 5.1 — Create a virtual environment

A virtual environment keeps this project's dependencies isolated from your system Python.

```bash
# Navigate to the project folder
cd /path/to/LLM-YT

# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac / Linux:
source venv/bin/activate
```

You will see `(venv)` at the start of your terminal prompt when it is active.

### Step 5.2 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `torch` — PyTorch, the deep learning framework everything is built on
- `numpy` — numerical computing (used internally by PyTorch)
- `tiktoken` — OpenAI's tokeniser (optional, for future BPE upgrades)
- `matplotlib` — plots the loss curve after training
- `tqdm` — progress bars
- `requests` — downloads the training data

### Step 5.3 — Verify your setup

```bash
python config.py
```

Expected output on a CPU-only machine:
```
Using device : cpu
PyTorch      : 2.x.x
No GPU found — training will run on CPU (slower but works fine)
```

Expected output with a CUDA GPU:
```
Using device : cuda
PyTorch      : 2.x.x
GPU          : NVIDIA GeForce RTX 3060
VRAM         : 6.0 GB
```

### Step 5.4 — Download training data

```bash
python data/download.py
```

This downloads **Tiny Shakespeare** (~1 MB of play text) to `data/input.txt`.

Expected output:
```
Downloading from:
  https://raw.githubusercontent.com/karpathy/...
Saved 1,115,394 characters to data/input.txt
Preview:
----------------------------------------
First Citizen:
Before we proceed any further, hear me speak.
...
```

> **Using your own data:** Simply place any plain `.txt` file at `data/input.txt` and skip this step. More text generally means better quality, but even 100 KB is enough to see something interesting.

### Step 5.5 — Prepare the data

```bash
python data/prepare.py
```

This script:
1. Reads `data/input.txt`
2. Builds a character vocabulary (every unique character gets an integer ID)
3. Encodes the entire text as a sequence of integers
4. Splits it 90% train / 10% validation
5. Saves `data/train.pt`, `data/val.pt`, and `data/vocab.json`

Expected output:
```
Total characters : 1,115,394
Vocabulary size  : 65
Characters       :
 !"&'()*,-.:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
Train tokens     : 1,003,854
Val   tokens     :   111,540
Encode/decode OK : "First Citiz" → [18, 47, ...] → "First Citiz"
Saved: data/train.pt, data/val.pt
```

### Step 5.6 — Train the model

```bash
python train.py
```

This is the main event. It will print a table like this every 500 steps:

```
Step   Train loss    Val loss        LR    Time
───────────────────────────────────────────────────────
     0      4.1743      4.1762  3.00e-04      2s
   500      2.0154      2.0712  2.94e-04     38s
  1000      1.6289      1.7842  2.74e-04     74s
  ...
  5000      1.1564      1.4783  3.00e-05    370s
```

When finished, the model is saved to `model/simple_gpt.pt`.

### Step 5.7 — Chat with the model

```bash
python chat.py
```

You will see an interactive prompt. Type any text and the model will continue it.

### Step 5.8 — (Optional) Generate text non-interactively

```bash
python generate.py --prompt "ROMEO:" --tokens 300
```

---

## 6. Understanding Each File

### `config.py`

Detects the best available hardware device. The result (`DEVICE`) is imported by `train.py`, `generate.py`, and `chat.py` so the model automatically uses your GPU if one is available.

```python
def get_device():
    if torch.cuda.is_available():     # NVIDIA GPU
        return 'cuda'
    elif torch.backends.mps.is_available():  # Apple Silicon GPU
        return 'mps'
    return 'cpu'
```

### `data/download.py`

Downloads text from a URL and saves it to `data/input.txt`. Already-downloaded files are not re-downloaded. Edit `DATASET_URL` to use any other public text file.

### `data/prepare.py`

**Character-level tokenisation** — the simplest possible approach:
- Build a mapping from every unique character to an integer (its "token ID")
- Encode the whole text as a list of integers
- Save as PyTorch tensors for fast loading during training

The mapping is saved to `data/vocab.json` so `generate.py` and `chat.py` can decode the model's integer outputs back into readable text.

### `data/loader.py`

During training, the model does not read the whole text at once. It reads random **windows** (chunks) of length `block_size`. The `get_batch()` function samples many such windows in parallel (one per item in the batch).

```
Text:   H  e  l  l  o     W  o  r  l  d
         ─── block_size = 5 ───
x:      [H, e, l, l, o]   (the context)
y:      [e, l, l, o, ' '] (what to predict at each position)
```

### `model/attention.py`

Contains four classes, each building on the previous:

| Class | What it does |
|---|---|
| `Head` | One self-attention head — each token looks at all previous tokens and gathers information |
| `MultiHeadAttention` | Runs several `Head`s in parallel and combines their outputs |
| `FeedForward` | A small two-layer neural network applied to each token independently |
| `TransformerBlock` | Combines `MultiHeadAttention` + `FeedForward` with skip connections and layer normalisation |

### `model/gpt.py`

The complete `SimpleGPT` model:
1. Converts token IDs to dense vectors via `token_embedding`
2. Adds positional information via `position_embedding`
3. Passes through 6 stacked `TransformerBlock`s
4. Projects to vocabulary logits via `lm_head`
5. The `generate()` method loops to produce one token at a time

### `train.py`

The training loop:
1. Sample a random batch of text windows
2. Forward pass: compute logits and cross-entropy loss
3. Backward pass: compute gradients (how to adjust each weight)
4. Optimizer step: update weights in the direction that reduces loss
5. Repeat thousands of times

Also includes a **learning rate schedule**: warms up gradually for the first 200 steps, then decays gently to a minimum over the rest of training.

### `generate.py`

Loads a saved checkpoint and generates text from the command line. Supports `--prompt`, `--tokens`, `--temp`, and `--topk` arguments.

### `chat.py`

Interactive REPL (Read-Eval-Print Loop) for talking to the model. Features:
- Live commands: `/temp 0.9`, `/topk 50`, `/len 200`
- Graceful handling of characters not in the training vocabulary
- Only the generated portion is printed (the prompt is not repeated)

---

## 7. Understanding the Model Architecture

### What is a Transformer?

A Transformer is a neural network that processes sequences of tokens. Unlike older recurrent networks (RNNs), it processes all positions in parallel, which makes it much faster to train.

### Embeddings

The model cannot work with raw characters. First, each character is converted to a **vector** (a list of 384 numbers). This vector is learned during training — similar characters end up with similar vectors.

Additionally, a **position embedding** is added so the model knows *where* in the sequence each token is (position 0, 1, 2, ...).

### Self-Attention

Self-attention lets each token gather information from all previous tokens. For each token:
- A **Query** vector asks "what do I need?"
- A **Key** vector broadcasts "here is what I offer"
- A **Value** vector carries the actual information to share

The model computes how much each token's key matches each query, uses those scores to weight the values, and produces a new mixed representation for each token.

The **causal mask** ensures each token can only attend to itself and earlier tokens — never future ones. This is essential because during generation, the future is unknown.

### Multi-Head Attention

Instead of one attention operation, we run 6 in parallel. Each head can specialise in different patterns — one might track subject-verb agreement, another tracks when a character is speaking.

### Feed-Forward Network

After attention (the "communication" phase), each token passes through a small neural network independently (the "thinking" phase). It expands to 4× the embedding dimension, applies GELU activation, then projects back.

### Residual Connections

Every sub-layer adds its output *back* to its input: `x = x + sublayer(x)`. This creates a "highway" for gradients during training — they can skip over layers rather than having to pass through every transformation. Without this, training 6 layers deep would be very difficult.

### Layer Normalisation

Normalises the activations to have zero mean and unit variance. Applied before each sub-layer (Pre-LN). This stabilises training — especially in the early steps when weights are random.

---

## 8. Training the Model

### What the training loop does

```
for each step:
    1. Pick a random batch of text windows
    2. Run the model forward → get logits (scores for each next token)
    3. Compare logits to actual next tokens → compute loss (cross-entropy)
    4. Backpropagate: compute gradients
    5. Update weights: move each weight in the direction that reduces loss
```

### What the loss means

**Cross-entropy loss** measures how surprised the model is by the actual next token. Lower is better.

- **Starting loss** (~4.17 for 65 vocab): the model is completely random — it assigns equal probability to all 65 characters. `-ln(1/65) ≈ 4.17`
- **Good loss** (~1.4–1.5 val loss): the model has learned real patterns and generates recognisable text
- **Overfit warning**: if train loss keeps dropping but val loss starts rising, the model is memorising training data rather than learning generalizable patterns

### Configuration options

Edit the `CONFIG` dict at the top of `train.py`:

| Setting | Default | Notes |
|---|---|---|
| `batch_size` | 64 | Reduce to 32 or 16 if you get out-of-memory errors |
| `block_size` | 256 | Context length. Reduce to 128 to save memory |
| `n_embd` | 384 | Embedding dim. Reduce to 256 for smaller model |
| `n_layer` | 6 | Depth. More layers = more capacity but slower |
| `max_steps` | 5000 | Increase to 10000–20000 for better quality |
| `learning_rate` | 3e-4 | Safe default for AdamW + this model size |

---

## 9. Chatting With the Model

```bash
python chat.py
```

### What the model does with your input

The model receives your typed text as a **prompt** and generates tokens that continue it. It does not understand your question as a question — it generates text in the style of its training data that plausibly follows your prompt.

For a Shakespeare-trained model, the best results come from prompts that look like the training data:

```
You: HAMLET:
Model: What means this shouting? I trust the king keeps his promise.

You: To be or not to be
Model: , that is the question: Whether 'tis nobler in the mind to suffer...
```

### Live controls

While chatting, you can adjust generation parameters without restarting:

| Command | Effect |
|---|---|
| `/temp 0.5` | More focused, less creative output |
| `/temp 1.2` | More creative, more random output |
| `/topk 10` | Sample only from top 10 most likely tokens |
| `/topk 0` | Disable top-k (sample from full distribution) |
| `/len 500` | Generate 500 tokens per response |
| `help` | Show usage tips |
| `quit` | Exit |

### Temperature explained

Temperature controls how "peaked" the probability distribution is:

- `temp = 0.5` → top token gets most probability → model repeats common patterns → safe but boring
- `temp = 1.0` → raw distribution unchanged → balanced creativity
- `temp = 1.5` → probability is flattened → more surprising, often less coherent

Start with `0.8` and adjust to taste.

### Top-k explained

Top-k sampling first keeps only the `k` most likely next tokens, sets all others to zero probability, then samples. This prevents the model from ever generating very unlikely (often nonsensical) tokens even at high temperatures. `k = 40` is a good default.

---

## 10. Generating Text

Non-interactive text generation:

```bash
# Default (no prompt, 500 tokens, temp 0.8, top-k 40)
python generate.py

# Custom prompt
python generate.py --prompt "ROMEO:" --tokens 300

# More focused output
python generate.py --prompt "JULIET:" --temp 0.6 --topk 20

# Wilder output
python generate.py --prompt "KING LEAR:" --temp 1.1 --topk 80

# Disable top-k entirely
python generate.py --prompt "ALL:" --topk 0
```

All flags:

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | `model/simple_gpt.pt` | Path to saved model |
| `--vocab` | `data/vocab.json` | Path to vocabulary file |
| `--prompt` | (empty) | Seed text for generation |
| `--tokens` | 500 | Number of tokens to generate |
| `--temp` | 0.8 | Temperature |
| `--topk` | 40 | Top-k cutoff (0 = disabled) |

---

## 11. Reading the Training Output

```
Step   Train loss    Val loss        LR    Time
───────────────────────────────────────────────────────
     0      4.1743      4.1762  3.00e-04      2s
   500      2.0154      2.0712  2.94e-04     38s
  1000      1.6289      1.7842  2.74e-04     74s
  2000      1.3845      1.5691  2.07e-04    148s
  3000      1.2843      1.5102  1.18e-04    222s
  4000      1.2121      1.4890  4.35e-05    296s
  5000      1.1564      1.4783  3.00e-05    370s
```

**Train loss** — average loss over recent training batches. Should always decrease.

**Val loss** — loss on data the model has never trained on. This is the true measure of quality. Should decrease, then plateau. If it rises while train loss drops — you are overfitting.

**LR** — current learning rate. It warms up over the first 200 steps then decays smoothly via cosine annealing.

**Time** — elapsed wall-clock seconds since training started.

A loss curve plot is saved automatically to `model/loss_curve.png` when training finishes (requires matplotlib).

---

## 12. Troubleshooting

### `FileNotFoundError: data/input.txt not found`

Run `python data/download.py` first.

### `FileNotFoundError: data/train.pt not found`

Run `python data/prepare.py` first.

### `FileNotFoundError: model/simple_gpt.pt not found`

Run `python train.py` first.

### `RuntimeError: CUDA out of memory`

Your GPU does not have enough VRAM. In `train.py`, reduce:
```python
'batch_size' : 32,   # was 64
'block_size' : 128,  # was 256
```
Or just use CPU — it trains the same, just slower.

### Training is very slow (CPU, hours estimated)

This is normal on CPU. Options:
- Reduce `max_steps` to 2000 for a quick test run
- Reduce `n_layer` to 4 and `n_embd` to 256 for a faster smaller model
- Use Google Colab (free GPU) — upload the project folder and run the same commands

### Generated text is complete gibberish

The model may not be trained enough yet. Check the val loss — you want to see it below 1.8 before the output looks coherent. Try running more steps.

### Generated text repeats the same phrase forever

Temperature is too low. Try `--temp 0.9` or `--temp 1.0`.

### `ModuleNotFoundError: No module named 'torch'`

Your virtual environment is not activated. Run:
```bash
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```
Then re-run the command.

### Characters in prompt are unknown / skipped

The model was trained on a specific set of characters. If you include emoji, non-ASCII letters, or other characters not in the training data, they are silently skipped in `chat.py`. Stick to ASCII for Shakespeare-trained models.

---

## 13. Improving the Model

### Quick wins (no code changes)

- **Train longer:** change `max_steps` to `10000` or `20000`
- **More data:** append another text file to `data/input.txt` before running `prepare.py`
- **Tune learning rate:** try `2e-4` or `5e-4` and compare val loss

### Architecture upgrades

| Change | How | Effect |
|---|---|---|
| Bigger model | `n_embd=512`, `n_layer=8`, `n_head=8` | More capacity, slower training |
| Smaller model | `n_embd=256`, `n_layer=4` | Faster, less capable |
| Less dropout | `dropout=0.1` | Faster learning, risk of overfitting |
| More dropout | `dropout=0.3` | Slower learning, less overfitting |

### BPE tokenisation (medium difficulty)

Character-level tokenisation treats every character as one token. BPE (Byte Pair Encoding) merges common character sequences into single tokens — so "the" or "ing" become one token instead of three. This lets the model read more text per step and learn longer-range patterns.

```python
import tiktoken
enc = tiktoken.get_encoding('gpt2')   # 50,257 token vocabulary
```

After switching to BPE, update `data/prepare.py` to use `enc.encode()` / `enc.decode()` instead of the character mappings. The rest of the code stays the same.

### Scale up on Google Colab

Once your model works locally, you can train a 10× bigger version on Colab's free T4 GPU:
1. Upload this folder to Google Drive
2. Open a Colab notebook, mount Drive, navigate to the folder
3. Change `n_embd=768`, `n_layer=12`, `batch_size=128`, `max_steps=20000`
4. Run `python train.py`

---

## 14. Concepts Glossary

| Term | Meaning |
|---|---|
| **Token** | The basic unit the model reads and predicts. Here: a single character. |
| **Vocabulary** | All unique tokens the model knows about. ~65 chars for this model. |
| **Embedding** | A dense vector representation of a token. Learned during training. |
| **Context window** | The maximum number of tokens the model can see at once (256 here). |
| **Attention** | A mechanism that lets each token gather information from other tokens. |
| **Causal mask** | Prevents each token from seeing future tokens (essential for generation). |
| **Multi-head attention** | Running many attention operations in parallel, each learning different patterns. |
| **Feed-forward network** | A small MLP applied to each token's representation independently. |
| **Residual connection** | Adding a layer's input to its output: `x = x + f(x)`. Helps training deep networks. |
| **Layer normalisation** | Normalises values within a layer to stabilise training. |
| **Cross-entropy loss** | Measures how well the model predicts the correct next token. Lower = better. |
| **Backpropagation** | Algorithm that computes how much to adjust each weight to reduce loss. |
| **AdamW** | The optimizer that updates weights based on gradients. Standard for Transformers. |
| **Learning rate** | How large each weight update step is. Too high = unstable. Too low = slow. |
| **Batch size** | How many text windows are processed together in one training step. |
| **Overfitting** | When the model memorises training data and performs poorly on unseen data. |
| **Temperature** | Controls randomness in generation. Higher = more creative. Lower = more focused. |
| **Top-k sampling** | Only consider the k most likely next tokens when generating. |
| **Checkpoint** | A saved copy of the model's weights at a point in training. |

---

## Quick Reference Card

```bash
# Full pipeline from scratch:
python data/download.py    # 1. Get training data
python data/prepare.py     # 2. Tokenise it
python train.py            # 3. Train the model (30 min – 4 hours)
python chat.py             # 4. Chat with your model!

# Generate text:
python generate.py --prompt "HAMLET:" --tokens 500

# Check your hardware:
python config.py
```

---

*Built following Andrej Karpathy's nanoGPT approach. Good luck, Evansify!*