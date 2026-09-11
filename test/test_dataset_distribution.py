"""Check monthly climatologies computed directly from observed NetCDF data."""

from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from analysis import plot_dataset_distribution as distribution


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared_data(tmp_path):
    data_dir = tmp_path / "processed_data"
    series_dir = data_dir / "time_series"
    series_dir.mkdir(parents=True)
    dates = pd.to_datetime(["2012-12-31", "2013-01-01", "2013-01-31", "2013-02-01",
                            "2014-01-01", "2014-02-28", "2015-01-01"])
    dataset = xr.Dataset({
        "discharge": ("date", [999, 2, np.nan, 6, 10, np.nan, 999]),
        "pr_x": ("date", [999, 1, 3, 5, 8, 7, 999]),
        "tmax_x": ("date", [999, 10, np.nan, 18, 22, 30, 999]),
        "tmin_x": ("date", [999, 0, 2, 6, 10, 14, 999]),
    }, coords={"date": dates})
    # Unsorted records must produce the same monthly climatology.
    for basin_id in ["4", "8", "15"]:
        dataset.isel(date=[6, 0, 4, 2, 1, 5, 3]).to_netcdf(series_dir / f"{basin_id}.nc")
    return data_dir


def test_monthly_means_use_daily_values_and_inclusive_dates(prepared_data):
    monthly = distribution.load_monthly_climatology(prepared_data, ["yrs", "4"], "2013-01-01", "2014-01-01")
    assert monthly["basin_name"].tolist() == ["YRS"] * 12  # Duplicate name/ID is plotted once.
    monthly = monthly.set_index("month")
    # Missing discharge keeps the corresponding precipitation observation. January pools
    # all daily samples, rather than weighting unequal annual monthly means equally.
    np.testing.assert_allclose(monthly.loc[1, ["qobs", "pr_x", "tmean"]].astype(float), [6, 4, 10.5])
    np.testing.assert_allclose(monthly.loc[2, ["qobs", "pr_x", "tmean"]].astype(float), [6, 5, 12])
    assert monthly.loc[3:12, ["qobs", "pr_x", "tmean"]].isna().all().all()


@pytest.mark.parametrize("kwargs, message", [
    ({"start_date": "2014-01-01", "end_date": "2013-01-01"}, "on or before"),
    ({"start_date": "2020-01-01", "end_date": "2020-12-31"}, "No data for"),
    ({"watersheds": ["unknown"]}, "Unknown watershed"),
])
def test_invalid_selections_are_reported(prepared_data, kwargs, message):
    with pytest.raises(ValueError, match=message):
        distribution.load_monthly_climatology(prepared_data, **kwargs)


def test_all_missing_variable_is_reported(prepared_data):
    with pytest.raises(ValueError, match="No valid qobs observations"):
        distribution.load_monthly_climatology(prepared_data, ["YRS"], "2013-01-31", "2013-01-31")


def test_cli_runs_with_only_script_and_prepared_data(prepared_data, tmp_path, monkeypatch):
    script = tmp_path / "analysis" / "plot_dataset_distribution.py"
    script.parent.mkdir()
    shutil.copy2(ROOT / "analysis/plot_dataset_distribution.py", script)
    other_workdir = tmp_path / "other_workdir"
    other_workdir.mkdir()
    monkeypatch.setenv("MPLBACKEND", "Agg")
    # No flags, model package, config, checkpoint, scaler, attributes, or analysis CSV.
    result = subprocess.run([sys.executable, str(script)], cwd=other_workdir, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    figures = sorted((tmp_path / "output").glob("*.png"))
    assert {path.name for path in figures} == {
        f"selected_watershed_monthly_climatology_{variable}.png"
        for variable in ["discharge", "precipitation", "temperature"]
    }
    for figure in figures:
        assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    for name in ["ISB", "NML", "YRS"]:
        assert f"{name}:" in result.stdout
    assert not (tmp_path / "moe_tau_expert_analysis").exists()
