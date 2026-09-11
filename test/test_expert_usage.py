"""Check direct checkpoint plotting against the existing gate-analysis calculation."""
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from analysis import plot_selected_watershed_expert_usage as usage
from analysis.analyze_moe_tau_experts import extract_gate_table
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.modelzoo import get_model
from neuralhydrology.utils.config import Config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def saved_run(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("expert_usage_run")
    train_dir = run_dir / "train_data"
    train_dir.mkdir()
    cfg = Config(ROOT / "runs/moe_tau.yml")
    cfg.update_config({
        "run_dir": run_dir, "train_dir": train_dir, "data_dir": ROOT / "processed_data",
        "train_basin_file": ROOT / "cdec_nondet_basin.txt",
        "validation_basin_file": ROOT / "cdec_nondet_basin.txt",
        "test_basin_file": ROOT / "cdec_nondet_basin.txt",
        "train_start_date": "01/01/2000", "train_end_date": "14/01/2000",
        "validation_start_date": "15/01/2000", "validation_end_date": "28/01/2000",
        "test_start_date": "01/02/2000", "test_end_date": "14/02/2000",
        "device": "cpu", "seq_length": 7, "hidden_size": 4, "verbose": 0,
        "number_of_basins": 14,
    })
    get_dataset(cfg, is_train=True, period="train")  # Save normalization learned on training data.
    torch.manual_seed(42)
    torch.save(get_model(cfg).state_dict(), run_dir / "model_epoch000.pt")
    cfg.dump_config(run_dir, filename="config.yml")
    return run_dir, cfg


@pytest.mark.parametrize("period", ["train", "validation", "test"])
def test_direct_means_match_existing_analysis(saved_run, period):
    run_dir, cfg = saved_run
    scaler_file = run_dir / "train_data/train_data_scaler.yml"
    original_scaler = scaler_file.read_bytes()
    # Five-sample batches leave a partial final batch: every sample must count equally.
    actual = usage.extract_mean_gate_weights(run_dir / "model_epoch000.pt", "yrs",
                                             period=period, batch_size=5)
    reference = extract_gate_table(cfg, run_dir, ["4"], period, 0, "cpu", 14, False)
    expected = reference[[f"expert_{i}_gate" for i in range(8)]].mean().to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual.sum(), 1.0, atol=1e-6)
    assert scaler_file.read_bytes() == original_scaler


def test_checkpoint_selection(tmp_path):
    for name in ("model_epoch000.pt", "model_epoch999.pt", "model_epoch1000.pt", "optimizer_state9999.pt"):
        (tmp_path / name).touch()
    assert usage._select_checkpoint(run_dir=tmp_path).name == "model_epoch1000.pt"
    assert usage._select_checkpoint(checkpoint=tmp_path).name == "model_epoch1000.pt"
    assert usage._select_checkpoint(run_dir=tmp_path, epoch=0).name == "model_epoch000.pt"
    assert usage._select_checkpoint(checkpoint=tmp_path, epoch=999).name == "model_epoch999.pt"
    assert usage._select_checkpoint(checkpoint=tmp_path / "model_epoch999.pt").name == "model_epoch999.pt"
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        usage._select_checkpoint(run_dir=tmp_path, epoch=5)
    with pytest.raises(ValueError, match="omit it when selecting an exact checkpoint file"):
        usage._select_checkpoint(checkpoint=tmp_path / "model_epoch000.pt", epoch=0)
    with pytest.raises(FileNotFoundError, match="file or run directory not found"):
        usage._select_checkpoint(checkpoint=tmp_path / "missing")
    with pytest.raises(ValueError, match="requires a folder"):
        usage._select_checkpoint(run_dir=tmp_path / "model_epoch000.pt")
    empty_run = tmp_path / "empty_run"
    empty_run.mkdir()
    with pytest.raises(FileNotFoundError, match="No model_epoch"):
        usage._select_checkpoint(checkpoint=empty_run)


@pytest.mark.parametrize("checkpoint_directory", [False, True])
def test_cli_creates_one_figure_without_analysis_csv(saved_run, tmp_path, monkeypatch, checkpoint_directory):
    run_dir, _ = saved_run
    monkeypatch.setenv("MPLBACKEND", "Agg")
    result = subprocess.run([
        sys.executable, str(ROOT / "analysis/plot_selected_watershed_expert_usage.py"),
        "--checkpoint", str(run_dir if checkpoint_directory else run_dir / "model_epoch000.pt"),
        "--watershed", "5",
        "--batch-size", "5", "--output-dir", str(tmp_path / "figures"),
    ], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    figure, = (tmp_path / "figures").iterdir()
    assert figure.name == "single_watershed_expert_usage_fol_test_model_epoch000.png"
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not (tmp_path / "moe_tau_expert_analysis").exists()


def test_original_model_name_loads_without_modifying_saved_config(saved_run, tmp_path):
    run_dir, _ = saved_run
    legacy_run = tmp_path / "legacy_run"
    shutil.copytree(run_dir, legacy_run)
    config_file = legacy_run / "config.yml"
    config_file.write_text(config_file.read_text().replace("model: moe_tau\n",
                                                          "model: moe_lstm_attn_learnable_temp_gating\n"))
    original_config = config_file.read_bytes()
    actual = usage.extract_mean_gate_weights(legacy_run / "model_epoch000.pt", "SCC")
    expected = usage.extract_mean_gate_weights(run_dir / "model_epoch000.pt", "SCC")
    np.testing.assert_array_equal(actual, expected)
    assert config_file.read_bytes() == original_config
