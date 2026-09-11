"""Verify standalone Spearman plotting against the original scientific analysis."""

from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from analysis import plot_spearman as spearman
from analysis.analyze_moe_tau_experts import (
    _watershed_distribution_expert_similarity,
    add_hydrologic_context,
    extract_gate_table,
)
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.modelzoo import get_model
from neuralhydrology.utils.config import Config


ROOT = Path(__file__).resolve().parents[1]
BASINS = ["4", "5", "8", "14", "15"]


@pytest.fixture(scope="module")
def saved_run(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("spearman_run")
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
    get_dataset(cfg, is_train=True, period="train")
    torch.manual_seed(42)
    torch.save(get_model(cfg).state_dict(), run_dir / "model_epoch000.pt")
    cfg.dump_config(run_dir, filename="config.yml")
    return run_dir, cfg


@pytest.mark.parametrize("period", ["train", "validation", "test"])
def test_distances_match_original_analysis(saved_run, period):
    run_dir, cfg = saved_run
    scaler_file = run_dir / "train_data/train_data_scaler.yml"
    original_scaler = scaler_file.read_bytes()
    actual = spearman.extract_gate_and_hydroclimate_table(
        run_dir / "model_epoch000.pt", BASINS, period=period, batch_size=5)
    original = extract_gate_table(cfg, run_dir, BASINS, period, 0, "cpu", 14, False)
    expected = add_hydrologic_context(original, ROOT / "processed_data", BASINS)
    columns = ["qobs", "pr_x", "pr_7d", "pr_30d", "tmean"] + [f"expert_{i}_gate" for i in range(8)]
    pd.testing.assert_frame_equal(actual[["basin_id", "date"]], expected[["basin_id", "date"]])
    np.testing.assert_allclose(actual[columns], expected[columns], rtol=1e-6, atol=1e-7, equal_nan=True)
    for reference_id in ["4", "8", "15"]:
        actual_distances = spearman.calculate_watershed_distances(actual, reference_id)
        expected_distances = _watershed_distribution_expert_similarity(
            expected, reference_id, list(spearman.DISTRIBUTION_VARIABLES))
        assert actual_distances["basin_id"].tolist() == expected_distances["basin_id"].tolist()
        distance_columns = ["distribution_distance", "expert_l1_distance"] + [f"{v}_ks" for v in spearman.DISTRIBUTION_VARIABLES]
        np.testing.assert_allclose(actual_distances[distance_columns], expected_distances[distance_columns],
                                   rtol=1e-6, atol=1e-7)
        actual_r = actual_distances["distribution_distance"].corr(actual_distances["expert_l1_distance"], method="spearman")
        expected_r = expected_distances["distribution_distance"].corr(expected_distances["expert_l1_distance"], method="spearman")
        assert actual_r == pytest.approx(expected_r)
        assert reference_id not in actual_distances["basin_id"].tolist()
    assert scaler_file.read_bytes() == original_scaler


@pytest.mark.parametrize("single_reference", [True, False])
def test_cli_generates_figures_and_tables_without_csv_inputs(saved_run, tmp_path, monkeypatch, single_reference):
    run_dir, _ = saved_run
    # A moved run with its original model name and obsolete saved data paths must work.
    legacy_run = tmp_path / "legacy_run"
    shutil.copytree(run_dir, legacy_run)
    config_file = legacy_run / "config.yml"
    legacy_config = Config(config_file)
    legacy_config.update_config({"model": "moe_lstm_attn_learnable_temp_gating",
                                 "run_dir": Path("/old/location"), "train_dir": Path("/old/location/train_data"),
                                 "data_dir": Path("/old/location/processed_data"),
                                 "test_basin_file": Path("/old/location/basins.txt")})
    config_file.unlink()
    legacy_config.dump_config(legacy_run, filename="config.yml")
    original_config = config_file.read_bytes()
    monkeypatch.setenv("MPLBACKEND", "Agg")
    command = [sys.executable, str(ROOT / "analysis/plot_spearman.py"),
               "--checkpoint", str(legacy_run / "model_epoch000.pt") if single_reference else str(legacy_run),
               "--batch-size", "5", "--output-dir", str(tmp_path / "figures")]
    if single_reference:
        command.extend(["--watershed", "4"])
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    references = ["yrs"] if single_reference else ["isb", "nml", "yrs"]
    assert len(list((tmp_path / "figures").iterdir())) == 2 * len(references)
    for reference in references:
        figure = tmp_path / "figures" / f"ks_distance_vs_expert_l1_distance_{reference}_test_model_epoch000.png"
        assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        table = pd.read_csv(figure.with_suffix(".csv"))
        assert len(table) == 13
        assert reference.upper() not in table["basin_name"].tolist()
        assert np.isfinite(table[["distribution_distance", "expert_l1_distance"]]).all().all()
        assert f"{reference.upper()} Spearman r:" in result.stdout
    assert config_file.read_bytes() == original_config
    assert not (tmp_path / "moe_tau_expert_analysis").exists()


def test_reference_validation_happens_before_inference(tmp_path):
    basin_file = tmp_path / "basins.txt"
    basin_file.write_text("4\n5\n8\n")
    with pytest.raises(ValueError, match="Unknown watershed"):
        spearman.plot_ks_vs_expert_l1_spearman_figures(["unknown"], basin_file=basin_file)
    with pytest.raises(ValueError, match="ISB is missing"):
        spearman.plot_ks_vs_expert_l1_spearman_figures(["ISB"], basin_file=basin_file)
    basin_file.write_text("4\n5\n")
    with pytest.raises(ValueError, match="at least three watersheds"):
        spearman.plot_ks_vs_expert_l1_spearman_figures(["YRS"], basin_file=basin_file)


def test_checkpoint_selection(tmp_path):
    for name in ("model_epoch000.pt", "model_epoch999.pt", "model_epoch1000.pt", "optimizer_state9999.pt"):
        (tmp_path / name).touch()
    assert spearman._select_checkpoint(run_dir=tmp_path).name == "model_epoch1000.pt"
    assert spearman._select_checkpoint(checkpoint=tmp_path).name == "model_epoch1000.pt"
    assert spearman._select_checkpoint(run_dir=tmp_path, epoch=0).name == "model_epoch000.pt"
    with pytest.raises(ValueError, match="omit it when selecting an exact checkpoint file"):
        spearman._select_checkpoint(checkpoint=tmp_path / "model_epoch000.pt", epoch=0)
