"""Plot watershed KS distances versus expert-weight L1 distances from a MoE-tau checkpoint."""

import argparse
from pathlib import Path
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from matplotlib.ticker import MaxNLocator
from scipy.stats import ks_2samp
from torch.utils.data import DataLoader

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from neuralhydrology.datautils.utils import load_basin_file, load_scaler
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.evaluation.utils import load_basin_id_encoding
from neuralhydrology.modelzoo import get_model
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import NoEvaluationDataError


BASIN_NAMES = {
    "1": "SHA", "3": "ORO", "4": "YRS", "5": "FOL", "6": "MKM", "7": "NHG", "8": "NML",
    "9": "TLG", "10": "MRC", "11": "MIL", "12": "PNF", "13": "TRM", "14": "SCC", "15": "ISB",
}
DISTRIBUTION_VARIABLES = ("qobs", "pr_x", "pr_7d", "pr_30d", "tmean")


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


def _target_scaler_value(values, target: str) -> float:
    if hasattr(values, "data_vars"):
        values = values[target]
    elif hasattr(values, "coords") and "variable" in values.coords:
        values = values.sel(variable=target)
    elif isinstance(values, dict):
        values = values[target]
    raw_values = values.values if hasattr(values, "values") else values
    return float(np.asarray(raw_values, dtype=np.float32).reshape(-1)[0])


def _load_hydroclimate(data_dir: Path, basin_id: str) -> pd.DataFrame:
    with xr.open_dataset(data_dir / "time_series" / f"{basin_id}.nc") as dataset:
        raw = dataset[["pr_x", "tmax_x", "tmin_x"]].to_dataframe().reset_index()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date")
    # Include antecedent precipitation before the selected evaluation period.
    raw["pr_7d"] = raw["pr_x"].rolling(7, min_periods=1).sum()
    raw["pr_30d"] = raw["pr_x"].rolling(30, min_periods=1).sum()
    raw["tmean"] = (raw["tmax_x"] + raw["tmin_x"]) / 2
    return raw[["date", "pr_x", "pr_7d", "pr_30d", "tmean"]]


def extract_gate_and_hydroclimate_table(checkpoint: Path,
                                      basin_ids: Iterable[str],
                                      period: str = "test",
                                      data_dir: Path = BASE_DIR / "processed_data",
                                      device: str = "cpu",
                                      batch_size: int = 512) -> pd.DataFrame:
    """Load a checkpoint once and collect gates and observed data for valid sequences."""
    if period not in {"train", "validation", "test"}:
        raise ValueError("period must be train, validation, or test.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    checkpoint = Path(checkpoint).expanduser().resolve()
    data_dir = Path(data_dir).expanduser().resolve()
    run_dir = checkpoint.parent
    cfg = Config(run_dir / "config.yml")
    if cfg.model == "moe_lstm_attn_learnable_temp_gating":
        cfg.update_config({"model": "moe_tau"})
    if cfg.model != "moe_tau":
        raise ValueError(f"Expected model: moe_tau in {run_dir / 'config.yml'}, found {cfg.model}.")
    cfg.update_config({"run_dir": run_dir, "train_dir": run_dir / "train_data",
                       "data_dir": data_dir, "device": device})
    model = get_model(cfg).to(device)
    try:
        state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:  # PyTorch versions without weights_only.
        state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    scaler = load_scaler(run_dir)
    target = cfg.target_variables[0]
    target_scale = _target_scaler_value(scaler["xarray_feature_scale"], target)
    target_center = _target_scaler_value(scaler["xarray_feature_center"], target)
    id_to_int = load_basin_id_encoding(run_dir) if cfg.use_basin_id_encoding else {}
    frames = []
    print(f"Checkpoint: {checkpoint}", flush=True)
    for basin_id in basin_ids:
        basin_id = str(basin_id)
        basin_name = BASIN_NAMES.get(basin_id, basin_id)
        try:
            dataset = get_dataset(cfg=cfg, is_train=False, period=period, basin=basin_id,
                                  scaler=scaler, id_to_int=id_to_int)
        except NoEvaluationDataError as exc:
            raise ValueError(f"No usable {period} samples for {basin_name}; check the data and configured dates.") from exc
        if len(dataset) == 0:
            raise ValueError(f"No usable {period} samples for {basin_name}.")
        print(f"Computing {period} gates for {basin_name} ({basin_id}): {len(dataset)} samples.", flush=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        batches = []
        with torch.inference_mode():
            for batch in loader:
                dates = pd.to_datetime(batch["date"][:, -1])
                obs_norm = batch["y"][:, -1, 0].cpu().numpy()
                batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                         for key, value in batch.items()}
                batch = model.pre_model_hook(batch, is_train=False)
                gates = model.gating_net(model.embedding_net(batch))
                if not torch.isfinite(gates).all():
                    raise ValueError(f"Non-finite gating weights for {basin_name}; check the checkpoint and inputs.")
                rows = pd.DataFrame(gates.cpu().numpy(), columns=[f"expert_{i}_gate" for i in range(model.num_experts)])
                rows["date"] = dates
                rows["qobs"] = obs_norm * target_scale + target_center
                batches.append(rows)
        basin_frame = pd.concat(batches, ignore_index=True)
        basin_frame["basin_id"] = basin_id
        basin_frame["basin_name"] = basin_name
        frames.append(basin_frame.merge(_load_hydroclimate(data_dir, basin_id), on="date", how="left",
                                       validate="many_to_one"))
    if not frames:
        raise ValueError("No watersheds supplied for comparison.")
    return pd.concat(frames, ignore_index=True)


def calculate_watershed_distances(table: pd.DataFrame, reference_watershed: str) -> pd.DataFrame:
    """Compare each other basin to the reference using the original five-variable KS mean and gate L1."""
    reference_id, reference_name = _resolve_watershed(reference_watershed)
    gate_cols = [col for col in table if col.endswith("_gate")]
    mean_gates = table.groupby("basin_id")[gate_cols].mean().astype(float)
    if reference_id not in mean_gates.index:
        raise ValueError(f"Reference watershed {reference_name} is missing from the comparison basin list.")
    if len(mean_gates) < 3:
        raise ValueError("Spearman correlation requires at least three watersheds: a reference and two comparisons.")
    # Each basin's mean gate vector is normalized exactly as in the original analysis.
    mean_gates = mean_gates.div(mean_gates.sum(axis=1), axis=0)
    reference = table[table["basin_id"] == reference_id]
    rows = []
    for basin_id in sorted(mean_gates.index, key=int):
        if basin_id == reference_id:
            continue
        candidate = table[table["basin_id"] == basin_id]
        variable_distances = {}
        for variable in DISTRIBUTION_VARIABLES:
            reference_values = reference[variable].dropna().to_numpy(dtype=float)
            candidate_values = candidate[variable].dropna().to_numpy(dtype=float)
            variable_distances[f"{variable}_ks"] = (
                float(ks_2samp(reference_values, candidate_values).statistic)
                if len(reference_values) and len(candidate_values) else np.nan)
        distances = np.asarray(list(variable_distances.values()))
        rows.append({
            "reference_basin_id": reference_id, "reference_basin_name": reference_name,
            "basin_id": basin_id, "basin_name": BASIN_NAMES.get(basin_id, basin_id),
            "distribution_distance": float(np.nanmean(distances)) if np.isfinite(distances).any() else np.nan,
            "expert_l1_distance": float(np.abs(mean_gates.loc[reference_id] - mean_gates.loc[basin_id]).sum()),
            **variable_distances,
        })
    return pd.DataFrame(rows).sort_values(["distribution_distance", "expert_l1_distance"]).reset_index(drop=True)


def _plot_single_reference_spearman(similarity: pd.DataFrame,
                                    output_file: Path,
                                    reference_name: str,
                                    dpi: int):
    if adjust_text is None:
        raise ImportError("The adjustText package is required for this plot. Install it with: pip install adjustText")

    required_columns = {"distribution_distance", "expert_l1_distance", "basin_name"}
    missing_columns = required_columns - set(similarity.columns)
    if missing_columns:
        raise ValueError(f"Missing column(s) for {reference_name}: {sorted(missing_columns)}")

    plot_df = similarity[["distribution_distance", "expert_l1_distance", "basin_name"]].replace(
        [np.inf, -np.inf], np.nan).dropna()
    if len(plot_df) < 2:
        raise ValueError(f"At least two valid comparison watersheds are needed to plot {reference_name}.")

    spearman = plot_df["distribution_distance"].corr(plot_df["expert_l1_distance"], method="spearman")

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.scatter(plot_df["distribution_distance"],
               plot_df["expert_l1_distance"],
               s=70,
               color="#9ecae1",
               edgecolor="black",
               linewidth=0.8,
               alpha=0.9)

    texts = []
    for _, row in plot_df.iterrows():
        texts.append(ax.text(row["distribution_distance"],
                             row["expert_l1_distance"],
                             row["basin_name"],
                             fontsize=10,
                             color="black",
                             fontweight="bold"))

    ax.set_xlabel("Average KS distance", fontweight="bold")
    ax.set_ylabel("Mean expert-weight L1 distance", fontweight="bold")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="both", labelsize=11)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    x_values = plot_df["distribution_distance"].to_numpy(dtype=float)
    y_values = plot_df["expert_l1_distance"].to_numpy(dtype=float)
    x_pad = max((np.nanmax(x_values) - np.nanmin(x_values)) * 0.12, 0.01)
    y_pad = max((np.nanmax(y_values) - np.nanmin(y_values)) * 0.12, 0.01)
    ax.set_xlim(np.nanmin(x_values) - x_pad, np.nanmax(x_values) + x_pad)
    ax.set_ylim(np.nanmin(y_values) - y_pad, np.nanmax(y_values) + y_pad)

    adjust_text(texts,
                x=x_values,
                y=y_values,
                ax=ax,
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#666666",
                    "lw": 0.7,
                    "alpha": 0.8,
                })

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"{reference_name} Spearman r: {spearman:.3f}")


def plot_ks_vs_expert_l1_spearman_figures(
        reference_watersheds: Iterable[str] = ("ISB", "NML", "YRS"),
        run_dir: Path = None,
        checkpoint: Path = None,
        epoch: int = None,
        period: str = "test",
        data_dir: Path = BASE_DIR / "processed_data",
        basin_file: Path = BASE_DIR / "cdec_nondet_basin.txt",
        device: str = "cpu",
        batch_size: int = 512,
        output_dir: Path = BASE_DIR / "output",
        file_prefix: str = "ks_distance_vs_expert_l1_distance",
        dpi: int = 150):
    """Compute all watershed comparisons and plot directly, without pre-generated CSV files."""
    if adjust_text is None:
        raise ImportError("The adjustText package is required. Install it with: python3 -m pip install adjustText")
    if dpi < 1:
        raise ValueError("--dpi must be positive.")
    if isinstance(reference_watersheds, str):
        reference_watersheds = [reference_watersheds]
    references = list(dict.fromkeys(_resolve_watershed(value) for value in reference_watersheds))
    if not references:
        raise ValueError("Select at least one reference watershed.")
    basin_ids = list(dict.fromkeys(load_basin_file(Path(basin_file).expanduser().resolve())))
    if len(basin_ids) < 3:
        raise ValueError("Spearman correlation requires at least three watersheds: a reference and two comparisons.")
    for basin_id, name in references:
        if basin_id not in basin_ids:
            raise ValueError(f"Reference watershed {name} is missing from the comparison basin list: {basin_file}")
    checkpoint = _select_checkpoint(run_dir=run_dir, checkpoint=checkpoint, epoch=epoch)
    table = extract_gate_and_hydroclimate_table(checkpoint, basin_ids, period, data_dir, device, batch_size)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for _, reference_name in references:
        similarity = calculate_watershed_distances(table, reference_name)
        output_file = output_dir / f"{file_prefix}_{reference_name.lower()}_{period}_{checkpoint.stem}.png"
        _plot_single_reference_spearman(similarity=similarity,
                                        output_file=output_file,
                                        reference_name=reference_name,
                                        dpi=dpi)
        similarity.to_csv(output_file.with_suffix(".csv"), index=False)
        figures.append(output_file)
        print(f"Saved Spearman figure to {output_file}")
        print(f"Saved comparison data to {output_file.with_suffix('.csv')}")
    return figures


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="Run folder with config.yml, train_data/, and model checkpoints.")
    source.add_argument("--checkpoint", type=Path,
                        help="Exact .pt file or a run folder containing model_epoch*.pt checkpoints.")
    parser.add_argument("--epoch", type=int,
                        help="Epoch to load from a run folder (default: latest saved checkpoint).")
    parser.add_argument("--watershed", "--reference-watersheds", dest="reference_watersheds", nargs="+",
                        default=["ISB", "NML", "YRS"],
                        help="Reference watershed names or IDs; one figure each (default: ISB NML YRS).")
    parser.add_argument("--period", choices=["train", "validation", "test"], default="test",
                        help="Date range from the saved run configuration (default: test).")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "processed_data",
                        help="Prepared time_series/ and attributes/ directory (default: project processed_data/).")
    parser.add_argument("--basin-file", type=Path, default=BASE_DIR / "cdec_nondet_basin.txt",
                        help="Comparison basin IDs, including references (default: all 14 project watersheds).")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda:0 (default: cpu).")
    parser.add_argument("--batch-size", type=int, default=512, help="Inference batch size (default: 512).")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "output", help="Directory for PNGs and comparison CSVs.")
    parser.add_argument("--dpi", type=int, default=150, help="Figure resolution (default: 150).")
    args = parser.parse_args()
    try:
        plot_ks_vs_expert_l1_spearman_figures(**vars(args))
    except (FileNotFoundError, ValueError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    _main()
