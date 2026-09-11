"""Plot one watershed's mean expert gating weights directly from a MoE-tau checkpoint."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.evaluation.utils import load_basin_id_encoding
from neuralhydrology.modelzoo import get_model
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import NoEvaluationDataError


BASIN_NAMES = {
    "1": "SHA", "3": "ORO", "4": "YRS", "5": "FOL", "6": "MKM", "7": "NHG", "8": "NML",
    "9": "TLG", "10": "MRC", "11": "MIL", "12": "PNF", "13": "TRM", "14": "SCC", "15": "ISB",
}


def _resolve_watershed(watershed: str) -> tuple:
    watershed = str(watershed).strip().upper()
    for basin_id, name in BASIN_NAMES.items():
        if watershed in {basin_id, name}:
            return basin_id, name
    raise ValueError(f"Unknown watershed '{watershed}'. Use a CDEC name or ID: "
                     + ", ".join(f"{name}/{basin_id}" for basin_id, name in BASIN_NAMES.items()))


def _select_checkpoint(run_dir: Path = None, checkpoint: Path = None, epoch: int = None) -> Path:
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


def extract_mean_gate_weights(checkpoint: Path,
                              watershed: str,
                              period: str = "test",
                              data_dir: Path = BASE_DIR / "processed_data",
                              device: str = "cpu",
                              batch_size: int = 512) -> np.ndarray:
    """Average gates over valid input sequences using the run's saved training scaler."""
    basin_id, basin_name = _resolve_watershed(watershed)
    if period not in {"train", "validation", "test"}:
        raise ValueError("period must be train, validation, or test.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    checkpoint = Path(checkpoint).expanduser().resolve()
    run_dir = checkpoint.parent
    cfg = Config(run_dir / "config.yml")
    if cfg.model == "moe_lstm_attn_learnable_temp_gating":
        # The MoE-tau rename preserved model parameters and checkpoint keys.
        cfg.update_config({"model": "moe_tau"})
    if cfg.model != "moe_tau":
        raise ValueError(f"Expected model: moe_tau in {run_dir / 'config.yml'}, found {cfg.model}.")
    cfg.update_config({"run_dir": run_dir, "train_dir": run_dir / "train_data",
                       "data_dir": Path(data_dir).expanduser().resolve(), "device": device})
    model = get_model(cfg).to(device)
    try:
        state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:  # PyTorch versions without weights_only.
        state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    scaler = load_scaler(run_dir)
    id_to_int = load_basin_id_encoding(run_dir) if cfg.use_basin_id_encoding else {}
    try:
        dataset = get_dataset(cfg=cfg, is_train=False, period=period, basin=basin_id,
                              scaler=scaler, id_to_int=id_to_int)
    except NoEvaluationDataError as exc:
        raise ValueError(f"No usable {period} samples for {basin_name}; check the data and configured dates.") from exc
    if len(dataset) == 0:
        raise ValueError(f"No usable {period} samples for {basin_name}.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    print(f"Checkpoint: {checkpoint}", flush=True)
    print(f"Computing {period} gates for {basin_name} ({basin_id}): {len(dataset)} samples.", flush=True)
    gate_sum = torch.zeros(model.num_experts, dtype=torch.float64)
    sample_count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                     for key, value in batch.items()}
            batch = model.pre_model_hook(batch, is_train=False)
            # The mean-usage plot only needs the embedding and gating network.
            gates = model.gating_net(model.embedding_net(batch))
            if not torch.isfinite(gates).all():
                raise ValueError(f"Non-finite gating weights for {basin_name}; check the checkpoint and inputs.")
            gate_sum += gates.to(device="cpu", dtype=torch.float64).sum(dim=0)
            sample_count += gates.shape[0]
    return (gate_sum / sample_count).numpy()


def plot_single_watershed_expert_activation_distribution(
        watershed: str,
        run_dir: Path = None,
        checkpoint: Path = None,
        epoch: int = None,
        period: str = "test",
        data_dir: Path = BASE_DIR / "processed_data",
        device: str = "cpu",
        batch_size: int = 512,
        output_dir: Path = BASE_DIR / "output",
        file_prefix: str = "single_watershed_expert_usage",
        dpi: int = 150) -> Path:
    """Extract and plot one watershed's mean gates from a selected model checkpoint."""
    _, basin_name = _resolve_watershed(watershed)
    checkpoint = _select_checkpoint(run_dir=run_dir, checkpoint=checkpoint, epoch=epoch)
    mean_gates = extract_mean_gate_weights(checkpoint, basin_name, period, data_dir, device, batch_size)
    expert_positions = np.arange(len(mean_gates))
    expert_labels = [f"E{idx}" for idx in range(1, len(mean_gates) + 1)]

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.bar(expert_positions,
           mean_gates,
           color="#9ecae1",
           edgecolor="black",
           linewidth=0.8,
           width=0.72)

    ax.set_xticks(expert_positions)
    ax.set_xticklabels(expert_labels, fontweight="bold")
    ax.set_xlabel("Expert", fontweight="bold")
    ax.set_ylabel("Mean gate weight", fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{file_prefix}_{basin_name.lower()}_{period}_{checkpoint.stem}.png"
    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved expert usage figure to {output_file}")
    return output_file


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="Run folder with config.yml, train_data/, and model checkpoints.")
    source.add_argument("--checkpoint", type=Path,
                        help="Exact .pt file or a run folder containing model_epoch*.pt checkpoints.")
    parser.add_argument("--epoch", type=int,
                        help="Epoch to load when selecting a run folder (default: latest saved checkpoint).")
    parser.add_argument("--watershed", required=True, help="One CDEC watershed name or ID, e.g. YRS or 4.")
    parser.add_argument("--period", choices=["train", "validation", "test"], default="test",
                        help="Date range from the saved run configuration (default: test).")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "processed_data",
                        help="Prepared time_series/ and attributes/ directory (default: project processed_data/).")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda:0 (default: cpu).")
    parser.add_argument("--batch-size", type=int, default=512, help="Inference batch size (default: 512).")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "output", help="Directory for the PNG figure.")
    parser.add_argument("--dpi", type=int, default=150, help="Figure resolution (default: 150).")
    args = parser.parse_args()
    try:
        plot_single_watershed_expert_activation_distribution(**vars(args))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    _main()
