"""Exercise the documented train/test-to-figures workflow using a real optimized model."""

import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from neuralhydrology.evaluation.tester import BaseTester
from neuralhydrology.modelzoo import get_model
from neuralhydrology.utils.config import Config
from runs.run_moe_tau import _get_weight_stem


ROOT = Path(__file__).resolve().parents[1]


def run_command(arguments, cwd=ROOT):
    env = {**os.environ, "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "2"}
    result = subprocess.run([sys.executable, *map(str, arguments)], cwd=cwd, env=env,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    work = tmp_path_factory.mktemp("moe_workflow")
    cfg = Config(ROOT / "runs/moe_tau.yml")
    cfg.update_config({
        "experiment_name": "moe_tau_smoke", "run_dir": work / "runs",
        "data_dir": ROOT / "processed_data",
        "train_basin_file": ROOT / "cdec_nondet_basin.txt",
        "validation_basin_file": ROOT / "cdec_nondet_basin.txt",
        "test_basin_file": ROOT / "cdec_nondet_basin.txt",
        "train_start_date": "01/01/2000", "train_end_date": "14/01/2000",
        "validation_start_date": "01/01/2001", "validation_end_date": "14/01/2001",
        "test_start_date": "01/02/2001", "test_end_date": "14/02/2001",
        "epochs": 2, "max_updates_per_epoch": 2, "batch_size": 8, "num_workers": 0,
        "validate_every": 1, "seed": 42, "verbose": 0,
    })
    # Keep the paper's 365-day window and 8-expert/32-hidden-unit architecture.
    cfg.dump_config(work, filename="smoke.yml")
    result = run_command([ROOT / "runs/run_moe_tau.py", "--config", work / "smoke.yml", "--gpu", "-1"])
    run_dir, = (work / "runs").iterdir()
    assert f"Finished MoE-tau training run: {run_dir}" in result.stdout
    assert "Epoch 2 average validation loss" in result.stdout
    assert list(run_dir.glob("loss_plot_*.png"))
    return run_dir


def test_training_optimizes_and_testing_saves_all_basins(trained_run):
    cfg = Config(trained_run / "config.yml")
    state = torch.load(trained_run / "model_epoch000.pt", weights_only=True)
    assert all(torch.isfinite(value).all() for value in state.values())
    initial_tau = get_model(cfg).gating_net.raw_scale.detach()
    assert not torch.equal(state["gating_net.raw_scale"], initial_tau)
    optimizer = torch.load(trained_run / "optimizer_state_epoch000.pt", weights_only=True)
    assert optimizer["state"]
    assert any(value["exp_avg"].abs().sum() > 0 for value in optimizer["state"].values())
    results_dir = trained_run / "test/model_epoch000"
    with (results_dir / "test_results.p").open("rb") as stream:
        results = pickle.load(stream)
    assert set(results) == set((ROOT / "cdec_nondet_basin.txt").read_text().split())
    for basin in results.values():
        predictions = basin["1D"]["xr"]["discharge_sim"].values
        assert len(predictions) == 14
        assert np.isfinite(predictions).all()
    metrics = pd.read_csv(results_dir / "test_metrics.csv")
    assert len(metrics) == 14
    assert np.isfinite(metrics[["NSE", "MSE"]]).all().all()


@pytest.mark.parametrize("script,options,expected_pngs", [
    ("plot_heatmap.py", [], 1),
    ("plot_dataset_distribution.py", [], 3),
    ("plot_selected_watershed_expert_usage.py", ["--watershed", "YRS"], 1),
    ("plot_spearman.py", ["--watershed", "YRS"], 1),
])
def test_documented_analyses_accept_training_output(trained_run, tmp_path, script, options, expected_pngs):
    config_before = (trained_run / "config.yml").read_bytes()
    scaler_before = (trained_run / "train_data/train_data_scaler.yml").read_bytes()
    arguments = [ROOT / "analysis" / script, *options]
    if script != "plot_dataset_distribution.py":
        arguments += ["--run-dir", trained_run]
    if script == "plot_heatmap.py":
        arguments += ["--output-file", tmp_path / "figure5.png"]
    else:
        arguments += ["--output-dir", tmp_path]
    run_command(arguments)
    figures = list(tmp_path.glob("*.png"))
    assert len(figures) == expected_pngs
    for figure in figures:
        with Image.open(figure) as img:
            assert min(img.size) > 300
            assert np.asarray(img).std() > 0
    if script == "plot_spearman.py":
        table = pd.read_csv(next(tmp_path.glob("*.csv")))
        assert len(table) == 13
        assert np.isfinite(table[["distribution_distance", "expert_l1_distance"]]).all().all()
    assert (trained_run / "config.yml").read_bytes() == config_before
    assert (trained_run / "train_data/train_data_scaler.yml").read_bytes() == scaler_before


def test_evaluation_of_relocated_legacy_run(trained_run, tmp_path):
    moved = tmp_path / "moved_run"
    moved.mkdir()
    shutil.copy2(trained_run / "model_epoch000.pt", moved)
    shutil.copytree(trained_run / "train_data", moved / "train_data")
    cfg = Config(trained_run / "config.yml")
    cfg.update_config({
        "model": "moe_lstm_attn_learnable_temp_gating",
        "run_dir": Path("/old/machine/run"), "train_dir": Path("/old/machine/run/train_data"),
        "data_dir": Path("/old/machine/processed_data"),
        "test_basin_file": Path("/old/machine/cdec_nondet_basin.txt"),
    })
    cfg.dump_config(moved, filename="config.yml")
    config_before = (moved / "config.yml").read_bytes()
    run_command([ROOT / "runs/run_moe_tau.py", "evaluate", "--run-dir", moved, "--gpu", "-1"], cwd=tmp_path)
    with (trained_run / "test/model_epoch000/test_results.p").open("rb") as stream:
        original = pickle.load(stream)
    with (moved / "test/model_epoch000/test_results.p").open("rb") as stream:
        actual = pickle.load(stream)
    for basin_id in original:
        np.testing.assert_array_equal(actual[basin_id]["1D"]["xr"]["discharge_sim"],
                                      original[basin_id]["1D"]["xr"]["discharge_sim"])
    assert (moved / "config.yml").read_bytes() == config_before


def test_evaluator_and_runner_select_checkpoints_numerically(tmp_path):
    for name in ("model_epoch999.pt", "model_epoch1000.pt", "model_epochinvalid.pt"):
        (tmp_path / name).touch()
    tester = object.__new__(BaseTester)
    tester.run_dir = tmp_path
    assert tester._get_weight_file(None).name == "model_epoch1000.pt"
    assert _get_weight_stem(tmp_path) == "model_epoch1000"
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        tester._get_weight_file(1)
