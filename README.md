# Pearl-Chat: Native Luganda language model in pure JAX

Pearl-Chat is a decoder-only transformer implemented from scratch in pure JAX using
Flax NNX, Optax, Orbax, and Grain. The model is trained on Luganda text with a custom
byte-level BPE tokenizer.

## System verification

Hardware and backend verified during project setup:

```
Backend: cpu
Devices: [CpuDevice(id=0)]
Python version: 3.11.15
Framework: JAX + Flax NNX (no PyTorch, no TensorFlow)
```

For distributed training and device mesh demonstration, the framework configures
eight logical devices via XLA host platform flags.

## Project metrics and real data

- Raw Luganda corpus: 4,048 articles from Luganda Wikipedia (`wikimedia/wikipedia 20231101.lg`)
- Processed dataset: 33,307 deduplicated lines
  - Training split: 31,642 lines (`data/processed/train.txt`)
  - Validation split: 1,665 lines (`data/processed/val.txt`)
- Tokenizer: custom Luganda byte-level BPE (`vocab.json`, `merges.txt`)
  - Vocabulary size: 4,096 tokens
  - Compression benchmark: 71 characters condensed into 16 native tokens
- Model configuration:
  - Architecture: GPT-2 style decoder-only transformer
  - Layers: 4
  - Attention heads: 4
  - Embedding dimension: 128
  - Feed-forward dimension: 512
  - Context length: 128 tokens
  - Parameter count: 1,331,986 parameters
- Performance on laptop CPU:
  - Training throughput: approximately 16,500 tokens per second
  - Training loss: decreased from 8.88 to 7.20 in 60 steps
  - Live workshop resumption: runs in under 20 seconds for 20 steps

## Repository structure

```
pearl-chat/
  pyproject.toml               # project metadata and dependencies
  Makefile                     # workflow targets
  README.md                    # project documentation and benchmarks
  requirements.md              # specification and requirements
  AGENTS.md                    # agent guidelines and formatting constraints
  data/
    raw/                       # raw Luganda text downloads
    processed/                 # cleaned and split Luganda corpus
    tokenizer/                 # trained BPE vocab and merge files
  src/pearlchat/
    config.py                  # dataclasses for model, training, and sharding
    tokenizer.py               # LugandaTokenizer wrapper
    data_pipeline.py           # deterministic Grain pipeline and data sources
    model.py                   # Flax NNX transformer and causal attention
    checkpointing.py           # Orbax save and restore management
    train.py                   # training step, schedule, and metrics logger
    sharding.py                # JAX mesh and array sharding utilities
    generate.py                # autoregressive sampling (top-k, top-p)
  scripts/
    01_fetch_data.py           # dataset download with fallback support
    02_train_tokenizer.py      # text normalization and tokenizer training
    03_prepare_dataset.py      # dataset splitting into train and val
    04_train_pearlchat.py      # training in full or live workshop mode
    05_run_chat_demo.py        # terminal inference interface
    06_sharding_visual_demo.py # visual 8-device sharding demonstration
  notebooks/
    workshop_walkthrough.ipynb # interactive spine walkthrough for the session
    colab_training.ipynb       # Google Colab GPU / TPU training notebook
  checkpoints/
    pearlchat-warm-start/      # pre-trained warm-start checkpoint
    logs/                      # training metric logs (CSV)
  tests/
    test_tokenizer.py          # round-trip encoding and special token tests
    test_data_pipeline.py      # Grain batching and checkpoint resumption tests
    test_model_shapes.py       # shapes and causal masking tests
    test_train_step.py         # training loss drop and reproduction tests
    test_sharding.py           # mesh reassembly tests
  ui/
    chat_app.py                # Gradio chat interface
```

## Setup instructions

1. Create a Python 3.11 virtual environment and install dependencies:

```bash
uv venv --python /opt/homebrew/bin/python3.11 .venv
source .venv/bin/activate
uv pip install -e .
```

2. Run the automated test suite:

```bash
pytest -v tests
```

## Quick start pipeline

The project includes standard Makefile targets to execute each stage:

```bash
# Fetch raw corpus and train tokenizer
make tokenizer

# Prepare train and validation splits
make dataset

# Train warm-start model
make train

# Run live workshop continuation (resumes from checkpoint)
make train-live

# Launch web chat interface
make chat
```

## Running the sharding demo

The device mesh demo runs on any machine by simulating eight logical CPU devices:

```bash
python scripts/06_sharding_visual_demo.py
```

This displays an ASCII layout of data-parallel batch partitioning across the `data`
axis and tensor-parallel weight projection partitioning across the `model` axis.

## Terminal chat demonstration

To generate Luganda responses directly in your terminal:

```bash
python scripts/05_run_chat_demo.py --prompt "Oli otya?" --max-tokens 48
```

## Training in Google Colab (GPU or TPU)

For accelerated pre-training on Google Colab:

1. Open `notebooks/colab_training.ipynb` in Google Colab.
2. Select a GPU runtime (T4, V100, A100, or L4) or a TPU v2/v3 runtime.
3. If using gated datasets such as `Sunbird/salt`, store your Hugging Face access
   token in Colab Secrets under `HF_TOKEN`.
4. Run the notebook top to bottom. The script will automatically detect the
   accelerator backend, pull the dataset, train the tokenizer, and run full
   pre-training with larger batch sizes.

## Troubleshooting

- No GPU detected: JAX defaults to CPU execution automatically. The live track
  is tuned so that CPU training completes in under five minutes.
- Out of memory: decrease `batch_size` in `scripts/04_train_pearlchat.py` or
  reduce `context_length` from 128 to 64.
- Checkpoint directory: Orbax requires absolute paths for tensorstore storage.
  The `CheckpointManager` class handles path resolution automatically.
- Grain iterator determinism: ensure the dataset files under `data/processed/`
  remain unmodified between checkpoint save and resumption.
