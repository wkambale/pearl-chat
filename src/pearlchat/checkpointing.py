"""Orbax checkpoint saving and restoration for Pearl-Chat."""

import json
from pathlib import Path
from typing import Any, Optional, Tuple, Union

from flax import nnx
import orbax.checkpoint as ocp

from pearlchat.config import ModelConfig
from pearlchat.model import PearlChatModel


def _restore_int_keys(tree: Any) -> Any:
    """Recursively convert digit string dictionary keys back to integers."""
    if isinstance(tree, dict):
        return {
            int(k) if isinstance(k, str) and k.isdigit() else k: _restore_int_keys(v)
            for k, v in tree.items()
        }
    return tree


class CheckpointManager:
    """Manages saving and restoring Flax NNX model states using Orbax."""

    def __init__(self, checkpoint_dir: Union[str, Path]):
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpointer = ocp.StandardCheckpointer()

    def save_checkpoint(
        self,
        model: PearlChatModel,
        step: int,
        config: Optional[ModelConfig] = None,
        extra_metadata: Optional[dict] = None,
    ) -> Path:
        """Save model state and configuration metadata."""
        step_dir = self.checkpoint_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        state = nnx.to_pure_dict(nnx.state(model))
        self.checkpointer.save(step_dir / "model_state", state)
        self.checkpointer.wait_until_finished()

        metadata = {
            "step": step,
            "vocab_size": model.config.vocab_size,
            "context_length": model.config.context_length,
            "embed_dim": model.config.embed_dim,
            "num_heads": model.config.num_heads,
            "num_layers": model.config.num_layers,
            "feed_forward_dim": model.config.feed_forward_dim,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        meta_file = step_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Update latest pointer
        latest_file = self.checkpoint_dir / "latest_step.txt"
        with open(latest_file, "w", encoding="utf-8") as f:
            f.write(str(step))

        return step_dir

    def restore_latest_checkpoint(
        self,
        model: PearlChatModel,
    ) -> Tuple[PearlChatModel, int]:
        """Restore model from latest available step checkpoint."""
        latest_file = self.checkpoint_dir / "latest_step.txt"
        if latest_file.exists():
            with open(latest_file, "r", encoding="utf-8") as f:
                step = int(f.read().strip())
            step_dir = self.checkpoint_dir / f"step_{step}"
        else:
            # Look for step directories
            step_dirs = [d for d in self.checkpoint_dir.glob("step_*") if d.is_dir()]
            if not step_dirs:
                # Check if checkpoint_dir itself contains model_state
                if (self.checkpoint_dir / "model_state").exists():
                    step_dir = self.checkpoint_dir
                    step = 0
                else:
                    raise FileNotFoundError(
                        f"No checkpoints found in {self.checkpoint_dir}"
                    )
            else:
                step_dirs.sort(key=lambda d: int(d.name.split("_")[-1]))
                step_dir = step_dirs[-1]
                step = int(step_dir.name.split("_")[-1])

        state_path = step_dir / "model_state"
        if not state_path.exists():
            raise FileNotFoundError(f"State file not found at {state_path}")

        restored_state = self.checkpointer.restore(state_path)
        restored_state = _restore_int_keys(restored_state)
        nnx.update(model, restored_state)

        return model, step
