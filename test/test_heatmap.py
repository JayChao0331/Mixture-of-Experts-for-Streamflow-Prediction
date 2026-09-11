"""Verify checkpoint-driven Figure 5 inference and global regime averaging."""

from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from analysis import plot_heatmap as heatmap
from analysis.analyze_moe_tau_experts import extract_gate_table
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.modelzoo import get_model
from neuralhydrology.utils.config import Config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def saved_run(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("heatmap_run")
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


def test_inference_and_regime_means_match_existing_analysis(saved_run, tmp_path, monkeypatch):
    run_dir, cfg = saved_run
    basin_ids = ["4", "8", "14", "15"]
    basin_file = tmp_path / "basins.txt"
    basin_file.write_text("\n".join(basin_ids))
    scaler_file = run_dir / "train_data/train_data_scaler.yml"
    original_scaler = scaler_file.read_bytes()
    original_config = (run_dir / "config.yml").read_bytes()
    model_calls = []
    original_get_model = heatmap.get_model

    def count_model_loads(config):
        model_calls.append(config)
        return original_get_model(config)

    monkeypatch.setattr(heatmap, "get_model", count_model_loads)
    actual_tables = heatmap.extract_train_test_gate_tables(run_dir / "model_epoch000.pt", basin_file=basin_file,
                                                         batch_size=5)
    assert len(model_calls) == 1
    reference_tables = [extract_gate_table(cfg, run_dir, basin_ids, period, 0, "cpu", 14, False)
                        for period in ["train", "test"]]
    gate_cols = [f"expert_{i}_gate" for i in range(8)]
    for actual, expected in zip(actual_tables, reference_tables):
        actual = actual.sort_values(["basin_id", "date"]).reset_index(drop=True)
        expected = expected.sort_values(["basin_id", "date"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual[["basin_id", "date"]], expected[["basin_id", "date"]])
        np.testing.assert_allclose(actual[["qobs"] + gate_cols], expected[["qobs"] + gate_cols],
                                   rtol=1e-6, atol=1e-7, equal_nan=True)
    train_summary, test_summary, thresholds = heatmap.summarize_train_test_gates(*actual_tables)
    reference_low, reference_high = reference_tables[0]["qobs"].quantile([0.30, 0.98])
    assert thresholds == pytest.approx({"low": reference_low, "high": reference_high})
    for summary, table in zip([train_summary, test_summary], reference_tables):
        valid = table.loc[np.isfinite(table["qobs"])].copy()
        valid["regime"] = np.where(valid["qobs"] <= reference_low, "low",
                                    np.where(valid["qobs"] >= reference_high, "high", "middle"))
        expected = valid.groupby("regime")[gate_cols].mean().reindex(heatmap.REGIME_ORDER)
        np.testing.assert_allclose(summary, expected, rtol=1e-6, atol=1e-7, equal_nan=True)
    assert scaler_file.read_bytes() == original_scaler
    assert (run_dir / "config.yml").read_bytes() == original_config


def test_training_thresholds_pool_samples_and_exclude_unknown_flows():
    train = pd.DataFrame({"qobs": [0, 1, 2, 3, 10], "basin_id": ["A", "B", "A", "A", "A"],
                          "expert_0_gate": [1, 0, .2, .6, .9], "expert_1_gate": [0, 1, .8, .4, .1]})
    test = pd.DataFrame({"qobs": [0, 2, 20, 30, np.nan, np.inf],
                         "expert_0_gate": [.8, .3, .1, .5, 1, 1], "expert_1_gate": [.2, .7, .9, .5, 0, 0]})
    train_summary, test_summary, thresholds = heatmap.summarize_train_test_gates(train, test)
    assert thresholds == pytest.approx({"low": 1.2, "high": 9.44})
    np.testing.assert_allclose(train_summary, [[.5, .5], [.4, .6], [.9, .1]])
    np.testing.assert_allclose(test_summary, [[.8, .2], [.3, .7], [.3, .7]])
    boundary_flows = pd.DataFrame({"qobs": [thresholds["low"], thresholds["high"], 2, np.nan, np.inf]})
    regimes = heatmap._assign_global_regime(boundary_flows, "qobs", thresholds)
    assert regimes.iloc[:3].tolist() == ["low", "high", "middle"]
    assert regimes.iloc[3:].isna().all()


def test_empty_regimes_are_missing_not_zero():
    table = pd.DataFrame({"qobs": [1, 1], "expert_0_gate": [.2, .4], "expert_1_gate": [.8, .6]})
    train, test, _ = heatmap.summarize_train_test_gates(table, table)
    for summary in [train, test]:
        np.testing.assert_allclose(summary.loc["low"], [.3, .7])
        assert summary.loc[["middle", "high"]].isna().all().all()


@pytest.mark.parametrize("source", ["run-dir", "checkpoint-file", "checkpoint-folder"])
def test_cli_creates_combined_heatmap_without_analysis_csv(saved_run, tmp_path, monkeypatch, source):
    run_dir, _ = saved_run
    legacy_run = tmp_path / "legacy_run"
    shutil.copytree(run_dir, legacy_run)
    config_file = legacy_run / "config.yml"
    cfg = Config(config_file)
    cfg.update_config({"model": "moe_lstm_attn_learnable_temp_gating", "run_dir": Path("/old/location"),
                       "train_dir": Path("/old/location/train_data"), "data_dir": Path("/old/location/data"),
                       "train_basin_file": Path("/old/location/basins.txt"),
                       "test_basin_file": Path("/old/location/basins.txt")})
    config_file.unlink()
    cfg.dump_config(legacy_run, filename="config.yml")
    original_config = config_file.read_bytes()
    output_file = tmp_path / "figures" / "figure5.png"
    monkeypatch.setenv("MPLBACKEND", "Agg")
    flag = "--run-dir" if source == "run-dir" else "--checkpoint"
    selected = legacy_run / "model_epoch000.pt" if source == "checkpoint-file" else legacy_run
    result = subprocess.run([sys.executable, str(ROOT / "analysis/plot_heatmap.py"), flag, str(selected),
                             "--batch-size", "5", "--output-file", str(output_file)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert list(output_file.parent.iterdir()) == [output_file]
    assert output_file.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert result.stdout.count("Computing train gates for basin") == 14
    assert result.stdout.count("Computing test gates for basin") == 14
    assert "also applied to testing" in result.stdout
    assert config_file.read_bytes() == original_config
    assert not (tmp_path / "moe_tau_expert_analysis").exists()


def test_checkpoint_selection(tmp_path):
    for name in ["model_epoch000.pt", "model_epoch999.pt", "model_epoch1000.pt"]:
        (tmp_path / name).touch()
    assert heatmap._select_checkpoint(run_dir=tmp_path).name == "model_epoch1000.pt"
    assert heatmap._select_checkpoint(checkpoint=tmp_path, epoch=0).name == "model_epoch000.pt"
    with pytest.raises(ValueError, match="omit it when selecting an exact checkpoint file"):
        heatmap._select_checkpoint(checkpoint=tmp_path / "model_epoch000.pt", epoch=0)
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        heatmap._select_checkpoint(run_dir=tmp_path, epoch=7)


@pytest.mark.parametrize("batch_size", [128, 512])
def test_missing_initial_history_is_excluded_before_inference(saved_run, tmp_path, capsys, batch_size):
    original_run, _ = saved_run
    run_dir = tmp_path / "warmup_run"
    shutil.copytree(original_run, run_dir)
    config_file = run_dir / "config.yml"
    cfg = Config(config_file)
    cfg.update_config({"run_dir": run_dir, "train_dir": run_dir / "train_data", "seq_length": 365,
                       "train_start_date": "01/10/1987", "train_end_date": "03/10/1988"})
    config_file.unlink()
    cfg.dump_config(run_dir, filename="config.yml")
    original_config = config_file.read_bytes()
    scaler_file = run_dir / "train_data/train_data_scaler.yml"
    original_scaler = scaler_file.read_bytes()
    basin_file = tmp_path / "sha.txt"
    basin_file.write_text("1\n")

    # Data start on 1987-10-01: the first 364 daily windows lack antecedent inputs.
    # Batch size 128 also exercises batches containing no usable sequences.
    train, test = heatmap.extract_train_test_gate_tables(run_dir / "model_epoch000.pt", basin_file=basin_file,
                                                        batch_size=batch_size)
    gate_cols = [f"expert_{i}_gate" for i in range(8)]
    valid_train = train.loc[np.isfinite(train[gate_cols]).all(axis=1)].reset_index(drop=True)
    assert train["date"].tolist() == pd.date_range("1987-10-01", "1988-10-03").tolist()
    assert valid_train["date"].tolist() == pd.date_range("1988-09-29", "1988-10-03").tolist()
    assert train[gate_cols].iloc[:364].isna().all().all()
    assert test["date"].tolist() == pd.date_range("2000-02-01", "2000-02-14").tolist()
    assert "Excluded 364 train sequences for basin 1" in capsys.readouterr().out
    assert np.isfinite(test[gate_cols]).all().all()
    # The original analyzer retains observed flows even when padded windows yield NaN gates.
    expected = extract_gate_table(cfg, run_dir, ["1"], "train", 0, "cpu", 128, False)
    np.testing.assert_allclose(train[["qobs"] + gate_cols], expected[["qobs"] + gate_cols],
                               rtol=1e-6, atol=1e-7, equal_nan=True)
    summary, _, thresholds = heatmap.summarize_train_test_gates(train, test)
    expected_low, expected_high = expected["qobs"].quantile([.30, .98])
    assert thresholds == pytest.approx({"low": expected_low, "high": expected_high})
    assert thresholds != heatmap._global_flow_thresholds(valid_train)
    expected["regime"] = np.where(expected["qobs"] <= expected_low, "low",
                                  np.where(expected["qobs"] >= expected_high, "high", "middle"))
    expected = expected.loc[expected["qobs"].notna()]
    expected_summary = expected.groupby("regime")[gate_cols].mean().reindex(heatmap.REGIME_ORDER)
    np.testing.assert_allclose(summary, expected_summary, rtol=1e-6, atol=1e-7, equal_nan=True)
    assert config_file.read_bytes() == original_config
    assert scaler_file.read_bytes() == original_scaler


def test_nonfinite_checkpoint_weights_are_not_silently_skipped(saved_run, tmp_path):
    original_run, _ = saved_run
    run_dir = tmp_path / "invalid_run"
    shutil.copytree(original_run, run_dir)
    checkpoint = run_dir / "model_epoch000.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state["gating_net.proj.weight"][0, 0] = float("nan")
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match=r"Checkpoint contains non-finite parameters: gating_net.proj.weight"):
        heatmap.extract_train_test_gate_tables(checkpoint)


def test_period_with_no_complete_inputs_has_clear_error(saved_run, tmp_path):
    original_run, _ = saved_run
    run_dir = tmp_path / "no_history_run"
    shutil.copytree(original_run, run_dir)
    config_file = run_dir / "config.yml"
    cfg = Config(config_file)
    cfg.update_config({"seq_length": 365, "train_start_date": "01/10/1987", "train_end_date": "14/10/1987"})
    config_file.unlink()
    cfg.dump_config(run_dir, filename="config.yml")
    basin_file = tmp_path / "sha.txt"
    basin_file.write_text("1\n")
    with pytest.raises(ValueError, match="No finite train input sequences for basin 1"):
        heatmap.extract_train_test_gate_tables(run_dir / "model_epoch000.pt", basin_file=basin_file)
