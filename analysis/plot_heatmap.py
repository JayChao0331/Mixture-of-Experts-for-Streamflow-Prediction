"""Plot Figure 5 training/testing gate heatmaps directly from a MoE-tau checkpoint."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from neuralhydrology.datautils.utils import load_basin_file, load_scaler
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.evaluation.utils import load_basin_id_encoding
from neuralhydrology.modelzoo import get_model
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import NoEvaluationDataError


REGIME_ORDER = ["low", "middle", "high"]
REGIME_LABELS = ["Low", "Mid", "High"]


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


def _target_scaler_value(values, target: str) -> float:
    if hasattr(values, "data_vars"):
        values = values[target]
    elif hasattr(values, "coords") and "variable" in values.coords:
        values = values.sel(variable=target)
    elif isinstance(values, dict):
        values = values[target]
    raw_values = values.values if hasattr(values, "values") else values
    return float(np.asarray(raw_values, dtype=np.float32).reshape(-1)[0])


def extract_train_test_gate_tables(checkpoint: Path,
                                  data_dir: Path = BASE_DIR / "processed_data",
                                  basin_file: Path = BASE_DIR / "cdec_nondet_basin.txt",
                                  device: str = "cpu",
                                  batch_size: int = 512) -> tuple:
    """Collect observed discharge and gates for both periods with one saved model/scaler."""
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    basin_ids = list(dict.fromkeys(load_basin_file(Path(basin_file).expanduser().resolve())))
    if not basin_ids:
        raise ValueError(f"No watershed IDs found in {basin_file}.")
    checkpoint = Path(checkpoint).expanduser().resolve()
    run_dir = checkpoint.parent
    cfg = Config(run_dir / "config.yml")
    if cfg.model == "moe_lstm_attn_learnable_temp_gating":
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
    invalid_parameters = [name for name, value in state_dict.items()
                          if isinstance(value, torch.Tensor) and not torch.isfinite(value).all()]
    if invalid_parameters:
        raise ValueError(f"Checkpoint contains non-finite parameters: {', '.join(invalid_parameters)}. "
                         "Select a checkpoint with finite weights.")
    model.load_state_dict(state_dict)
    model.eval()

    scaler = load_scaler(run_dir)
    target = cfg.target_variables[0]
    target_scale = _target_scaler_value(scaler["xarray_feature_scale"], target)
    target_center = _target_scaler_value(scaler["xarray_feature_center"], target)
    id_to_int = load_basin_id_encoding(run_dir) if cfg.use_basin_id_encoding else {}
    tables = []
    print(f"Checkpoint: {checkpoint}", flush=True)
    for period in ("train", "test"):
        print(f"{period.capitalize()} dates: {getattr(cfg, period + '_start_date').date()} to "
              f"{getattr(cfg, period + '_end_date').date()}", flush=True)
        frames = []
        for basin_id in basin_ids:
            try:
                # Even the training period uses evaluation mode and the saved training scaler.
                dataset = get_dataset(cfg=cfg, is_train=False, period=period, basin=basin_id,
                                      scaler=scaler, id_to_int=id_to_int)
            except NoEvaluationDataError as exc:
                raise ValueError(f"No usable {period} samples for basin {basin_id}; check the data and configured dates.") from exc
            if len(dataset) == 0:
                raise ValueError(f"No usable {period} samples for basin {basin_id}.")
            print(f"Computing {period} gates for basin {basin_id}: {len(dataset)} samples.", flush=True)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
            skipped_samples = 0
            used_samples = 0
            invalid_inputs = set()
            with torch.inference_mode():
                for batch in loader:
                    dates = pd.to_datetime(batch["date"][:, -1])
                    obs_norm = batch["y"][:, -1, 0].cpu().numpy()
                    gate_cols = [f"expert_{i}_gate" for i in range(model.num_experts)]
                    rows = pd.DataFrame(np.full((len(dates), model.num_experts), np.nan, dtype=np.float32),
                                        columns=gate_cols)
                    rows["qobs"] = obs_norm * target_scale + target_center
                    rows["basin_id"] = basin_id
                    rows["date"] = dates
                    # Evaluation retains NaN-padded history and gaps to preserve its date axis.
                    # A gate can only be computed when the entire input sequence is finite.
                    # Do not filter on y: missing observations are handled when assigning regimes.
                    valid = torch.ones(batch["x_d"].shape[0], dtype=torch.bool)
                    for key in ("x_d", "x_s", "x_one_hot"):
                        if key in batch:
                            finite = torch.isfinite(batch[key]).reshape(len(valid), -1).all(dim=1)
                            valid &= finite
                            if not finite.all():
                                invalid_inputs.add(key)
                    skipped_samples += int((~valid).sum())
                    if not valid.any():
                        # Keep observed flows for the global training percentiles, even without gates.
                        frames.append(rows)
                        continue
                    if not valid.all():
                        batch = {key: value[valid] if isinstance(value, torch.Tensor) else value[valid.numpy()]
                                 for key, value in batch.items()}
                    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                             for key, value in batch.items()}
                    batch = model.pre_model_hook(batch, is_train=False)
                    # Figure 5 needs only the embedding and gating network, not expert predictions.
                    gates = model.gating_net(model.embedding_net(batch))
                    if not torch.isfinite(gates).all():
                        raise ValueError(f"Non-finite {period} gating weights for basin {basin_id} despite finite inputs "
                                         f"({dates.min().date()} to {dates.max().date()}). "
                                         "Check the checkpoint and saved normalization for numerical problems.")
                    rows.loc[valid.numpy(), gate_cols] = gates.cpu().numpy()
                    frames.append(rows)
                    used_samples += int(valid.sum())
            if skipped_samples:
                print(f"Excluded {skipped_samples} {period} sequences for basin {basin_id} from gate inference due to "
                      f"missing or non-finite inputs ({', '.join(sorted(invalid_inputs))}); retained {used_samples}. "
                      "Observed flows remain available for training thresholds.", flush=True)
            if not used_samples:
                raise ValueError(f"No finite {period} input sequences for basin {basin_id}. "
                                 "Check the available input history, prepared data, and saved training scaler.")
        tables.append(pd.concat(frames, ignore_index=True))
    return tuple(tables)


def _gate_columns(table: pd.DataFrame):
    return [col for col in table.columns if col.startswith("expert_") and col.endswith("_gate")]


def _global_flow_thresholds(table: pd.DataFrame,
                            flow_col: str = "qobs",
                            low_quantile: float = 0.30,
                            high_quantile: float = 0.98):
    if not 0 <= low_quantile < high_quantile <= 1:
        raise ValueError("Quantiles must satisfy 0 <= low_quantile < high_quantile <= 1.")
    flows = table.loc[np.isfinite(table[flow_col]), flow_col]
    if flows.empty:
        raise ValueError(f"No finite values found in flow column '{flow_col}'.")

    return {
        "low": float(flows.quantile(low_quantile)),
        "high": float(flows.quantile(high_quantile)),
    }


def _assign_global_regime(table: pd.DataFrame, flow_col: str, thresholds: dict) -> pd.Series:
    values = table[flow_col]
    regimes = pd.Series(
        np.where(values <= thresholds["low"],
                 "low",
                 np.where(values >= thresholds["high"], "high", "middle")),
        index=table.index,
        dtype="object",
    )
    # Missing observations have no flow regime and must not be counted as mid-flow.
    return regimes.where(np.isfinite(values))


def _mean_gate_by_regime(table: pd.DataFrame, regime_col: str = "global_qobs_fhv_flv_regime") -> pd.DataFrame:
    gate_cols = _gate_columns(table)
    valid = table.loc[np.isfinite(table[gate_cols]).all(axis=1)]
    return valid.groupby(regime_col)[gate_cols].mean().reindex(REGIME_ORDER)


def _print_regime_counts(name: str, table: pd.DataFrame, regime_col: str):
    table = table.loc[np.isfinite(table[_gate_columns(table)]).all(axis=1)]
    counts = table[regime_col].value_counts().reindex(REGIME_ORDER).fillna(0).astype(int)
    print(f"{name} regime sample counts:")
    for regime, count in counts.items():
        print(f"  {regime}: {count}")
    missing_count = int(table[regime_col].isna().sum())
    if missing_count:
        print(f"  Excluded {missing_count} samples with missing or non-finite observed discharge.")


def summarize_train_test_gates(train_table: pd.DataFrame, test_table: pd.DataFrame,
                               low_quantile: float = 0.30, high_quantile: float = 0.98) -> tuple:
    """Pool daily samples across basins and apply training discharge thresholds to both sets."""
    if not _gate_columns(train_table) or _gate_columns(train_table) != _gate_columns(test_table):
        raise ValueError("Training and testing tables must have the same expert gate columns.")
    thresholds = _global_flow_thresholds(train_table, low_quantile=low_quantile, high_quantile=high_quantile)
    if not np.isfinite(test_table["qobs"]).any():
        raise ValueError("No finite observed discharge in the testing set.")
    regime_col = "global_qobs_fhv_flv_regime"
    train_table = train_table.assign(**{regime_col: _assign_global_regime(train_table, "qobs", thresholds)})
    test_table = test_table.assign(**{regime_col: _assign_global_regime(test_table, "qobs", thresholds)})

    train_summary = _mean_gate_by_regime(train_table, regime_col=regime_col)
    test_summary = _mean_gate_by_regime(test_table, regime_col=regime_col)

    print("Global observed-discharge thresholds from training samples (also applied to testing):")
    print(f"  Low <= {thresholds['low']:.6g} ({low_quantile:.0%} quantile)")
    print(f"  High >= {thresholds['high']:.6g} ({high_quantile:.0%} quantile)")
    _print_regime_counts("Training set", train_table, regime_col)
    _print_regime_counts("Testing set", test_table, regime_col)
    return train_summary, test_summary, thresholds


def _plot_train_test_summaries(train_summary: pd.DataFrame, test_summary: pd.DataFrame,
                               output_file: Path, dpi: int):
    """Draw two heatmaps with one shared scale and mark regimes with no samples."""
    gate_cols = train_summary.columns.tolist()
    expert_labels = [f"E{idx}" for idx in range(1, len(gate_cols) + 1)]
    vmax = max(0.01, np.nanmax(np.concatenate([train_summary.to_numpy(dtype=float),
                                             test_summary.to_numpy(dtype=float)])))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharex=True, sharey=True, constrained_layout=True)
    heatmaps = [
        (axes[0], train_summary, "Training set"),
        (axes[1], test_summary, "Testing set"),
    ]

    im = None
    for ax, table, set_label in heatmaps:
        values = table.to_numpy(dtype=float)
        im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        for row in range(len(REGIME_ORDER)):
            if not np.isfinite(values[row]).any():
                ax.text((len(expert_labels) - 1) / 2, row, "No samples", ha="center", va="center", fontsize=10)

        ax.set_xticks(np.arange(len(expert_labels)))
        ax.set_xticklabels(expert_labels, fontweight="bold")
        ax.set_yticks(np.arange(len(REGIME_LABELS)))
        ax.set_yticklabels(REGIME_LABELS, fontweight="bold")
        ax.tick_params(axis="y", labelleft=True)
        ax.set_xlabel("Expert", fontweight="bold")
        ax.set_ylabel(f"{set_label} global streamflow regime", fontweight="bold")
        ax.tick_params(axis="both", labelsize=11)

        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight("bold")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical", fraction=0.035, pad=0.03)
    cbar.set_label("Mean gate weight", fontweight="bold")
    cbar.ax.tick_params(labelsize=11)
    for tick_label in cbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    output_file = Path(output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved global-regime heatmap to {output_file}")
    return output_file


def plot_train_test_global_qobs_fhv_flv_gate_heatmaps(
        run_dir: Path = None,
        checkpoint: Path = None,
        epoch: int = None,
        data_dir: Path = BASE_DIR / "processed_data",
        basin_file: Path = BASE_DIR / "cdec_nondet_basin.txt",
        device: str = "cpu",
        batch_size: int = 512,
        output_file: Path = BASE_DIR / "output" / "mean_gate_by_global_qobs_fhv_flv_regime_train_test.png",
        dpi: int = 150) -> Path:
    """Compute and plot Figure 5 from one checkpoint, without pre-generated gate tables."""
    if dpi < 1:
        raise ValueError("--dpi must be positive.")
    checkpoint = _select_checkpoint(run_dir=run_dir, checkpoint=checkpoint, epoch=epoch)
    train_table, test_table = extract_train_test_gate_tables(checkpoint, data_dir, basin_file, device, batch_size)
    train_summary, test_summary, _ = summarize_train_test_gates(train_table, test_table)
    return _plot_train_test_summaries(train_summary, test_summary, output_file, dpi)


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="Run folder with config.yml, train_data/, and model checkpoints.")
    source.add_argument("--checkpoint", type=Path,
                        help="Exact .pt file or a run folder containing model_epoch*.pt checkpoints.")
    parser.add_argument("--epoch", type=int,
                        help="Epoch to load from a run folder (default: latest saved checkpoint).")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "processed_data",
                        help="Prepared time_series/ and attributes/ directory (default: project processed_data/).")
    parser.add_argument("--basin-file", type=Path, default=BASE_DIR / "cdec_nondet_basin.txt",
                        help="Watershed IDs used in both heatmaps (default: all 14 project watersheds).")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda:0 (default: cpu).")
    parser.add_argument("--batch-size", type=int, default=512, help="Inference batch size (default: 512).")
    parser.add_argument("--output-file", type=Path,
                        default=BASE_DIR / "output" / "mean_gate_by_global_qobs_fhv_flv_regime_train_test.png",
                        help="PNG file for the combined training/testing heatmaps.")
    parser.add_argument("--dpi", type=int, default=150, help="Figure resolution (default: 150).")
    args = parser.parse_args()
    try:
        plot_train_test_global_qobs_fhv_flv_gate_heatmaps(**vars(args))
    except (FileNotFoundError, ValueError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    _main()
