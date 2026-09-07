"""Configuration dataclasses for Pearl-Chat model, training, and data pipelines."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    """Hyperparameters for the Pearl-Chat transformer model."""

    vocab_size: int = 8192
    context_length: int = 256
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    feed_forward_dim: int = 1024
    dropout_rate: float = 0.1


@dataclass
class DataConfig:
    """Paths and settings for dataset preparation and tokenization."""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    tokenizer_dir: str = "data/tokenizer"
    context_length: int = 256
    val_ratio: float = 0.05
    seed: int = 42


@dataclass
class TrainConfig:
    """Hyperparameters and runtime settings for model training."""

    batch_size: int = 16
    learning_rate: float = 5e-4
    min_learning_rate: float = 5e-5
    warmup_steps: int = 50
    total_steps: int = 500
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    eval_every: int = 25
    eval_steps: int = 10
    checkpoint_every: int = 100
    checkpoint_dir: str = "checkpoints/pearlchat-warm-start"
    log_dir: str = "checkpoints/logs"
    seed: int = 42
    live_steps: int = 30


@dataclass
class ShardingConfig:
    """Configuration for device mesh and array sharding demonstration."""

    num_devices: int = 8
    data_axis_name: str = "data"
    model_axis_name: str = "model"
    mesh_shape: Tuple[int, int] = (4, 2)
