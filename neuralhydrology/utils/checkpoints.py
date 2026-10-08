"""Select and load the same saved model in evaluation and analysis commands."""

from pathlib import Path

import torch


def select_checkpoint(run_dir: Path = None, checkpoint: Path = None, epoch: int = None) -> Path:
    """Accept a run folder or exact weight file; order saved epochs numerically."""
    if (run_dir is None) == (checkpoint is None):
        raise ValueError("Specify either --run-dir or --checkpoint.")
    if checkpoint is not None:
        checkpoint = Path(checkpoint).expanduser().resolve()
        if checkpoint.is_dir():
            run_dir = checkpoint
        else:
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint file or run directory not found: {checkpoint}")
            if epoch is not None:
                raise ValueError("--epoch applies to a run folder; omit it when selecting an exact checkpoint file.")
            return checkpoint

    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not run_dir.is_dir():
        raise ValueError("--run-dir requires a folder. Use --checkpoint to select a .pt file.")
    if epoch is not None:
        if epoch < 0:
            raise ValueError("--epoch must be non-negative.")
        weight_file = run_dir / f"model_epoch{epoch:03d}.pt"
    else:
        weight_files = [path for path in run_dir.glob("model_epoch*.pt")
                        if path.is_file() and path.stem[len("model_epoch"):].isdigit()]
        if not weight_files:
            raise FileNotFoundError(f"No model_epoch*.pt checkpoints found in {run_dir}")
        weight_file = max(weight_files, key=lambda path: int(path.stem[len("model_epoch"):]))
    if not weight_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weight_file}")
    return weight_file


def load_model_weights(model: torch.nn.Module, checkpoint: Path, device):
    """Reject invalid weights before inference, including in the gating network."""
    try:
        state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:  # PyTorch versions without weights_only.
        state_dict = torch.load(checkpoint, map_location=device)
    invalid = [name for name, value in state_dict.items()
               if isinstance(value, torch.Tensor) and not torch.isfinite(value).all()]
    if invalid:
        raise ValueError(f"Checkpoint contains non-finite parameters: {', '.join(invalid)}. "
                         "Select a checkpoint with finite weights.")
    model.load_state_dict(state_dict)
