"""Plot Figure 6 monthly climatologies directly from prepared data, without a model or analysis CSV."""

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


BASE_DIR = Path(__file__).resolve().parents[1]
# Match the test-period data used by the original Figure 6 analysis.
DEFAULT_START_DATE = "2013-01-01"
DEFAULT_END_DATE = "2018-12-31"
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


def load_monthly_climatology(data_dir: Path = BASE_DIR / "processed_data",
                            watersheds: Iterable[str] = ("ISB", "NML", "YRS"),
                            start_date: str = DEFAULT_START_DATE,
                            end_date: str = DEFAULT_END_DATE) -> pd.DataFrame:
    """Average daily observations by calendar month across the selected years."""
    start = pd.to_datetime(start_date, format="%Y-%m-%d", errors="raise")
    end = pd.to_datetime(end_date, format="%Y-%m-%d", errors="raise")
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("Use valid dates with --start-date on or before --end-date (YYYY-MM-DD).")
    if isinstance(watersheds, str):
        watersheds = [watersheds]
    basins = list(dict.fromkeys(_resolve_watershed(watershed) for watershed in watersheds))
    if not basins:
        raise ValueError("Select at least one watershed.")

    data_dir = Path(data_dir).expanduser().resolve()
    frames = []
    required = ["discharge", "pr_x", "tmax_x", "tmin_x"]
    for basin_id, basin_name in basins:
        data_file = data_dir / "time_series" / f"{basin_id}.nc"
        if not data_file.is_file():
            raise FileNotFoundError(f"Missing prepared data for {basin_name}: {data_file}")
        with xr.open_dataset(data_file) as dataset:
            missing = set(required) - set(dataset.data_vars)
            if missing:
                raise ValueError(f"Missing variable(s) in {data_file}: {sorted(missing)}")
            if "date" not in dataset.coords:
                raise ValueError(f"Missing date coordinate in {data_file}")
            daily = dataset[required].to_dataframe()
        daily.index = pd.to_datetime(daily.index)
        daily = daily.sort_index().loc[start:end].replace([np.inf, -np.inf], np.nan)
        if daily.empty:
            raise ValueError(f"No data for {basin_name} between {start.date()} and {end.date()}.")
        # Prepared discharge and precipitation are already in mm/day; temperature is in degrees C.
        daily = daily.rename(columns={"discharge": "qobs"})
        daily["tmean"] = (daily["tmax_x"] + daily["tmin_x"]) / 2
        variables = ["qobs", "pr_x", "tmean"]
        for variable in variables:
            if daily[variable].isna().all():
                raise ValueError(f"No valid {variable} observations for {basin_name} in the selected dates.")
        # Each variable uses its available observations; missing discharge does not remove climate data.
        monthly = daily[variables].groupby(daily.index.month).mean().reindex(range(1, 13))
        monthly.index.name = "month"
        monthly["basin_name"] = basin_name
        frames.append(monthly.reset_index())
        print(f"{basin_name}: {len(daily)} daily records, {daily.index.min().date()} to {daily.index.max().date()}.")
    return pd.concat(frames, ignore_index=True)


def plot_selected_watershed_monthly_climatology(
        data_dir: Path = BASE_DIR / "processed_data",
        watersheds: Iterable[str] = ("ISB", "NML", "YRS"),
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
        output_dir: Path = BASE_DIR / "output",
        file_prefix: str = "selected_watershed_monthly_climatology",
        dpi: int = 150):
    """Read observed data and plot one Figure 6 panel per hydroclimate variable."""
    if dpi < 1:
        raise ValueError("--dpi must be positive.")
    monthly = load_monthly_climatology(data_dir, watersheds, start_date, end_date)
    basin_names = monthly["basin_name"].drop_duplicates().tolist()
    variables = [
        ("qobs", "Observed discharge (mm/day)", "discharge"),
        ("pr_x", "Precipitation (mm/day)", "precipitation"),
        ("tmean", r"Mean temperature ($^\circ$C)", "temperature"),
    ]

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for variable, ylabel, file_suffix in variables:
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        for basin_name in basin_names:
            basin_monthly = monthly[monthly["basin_name"] == basin_name]
            ax.plot(basin_monthly["month"],
                    basin_monthly[variable],
                    marker="o",
                    linewidth=2,
                    markersize=4,
                    label=basin_name)

        ax.set_xlabel("Month", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_xticks(range(1, 13))
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(axis="both", color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight("bold")

        legend = ax.legend(frameon=False)
        for text in legend.get_texts():
            text.set_fontweight("bold")

        fig.tight_layout()
        output_file = output_dir / f"{file_prefix}_{file_suffix}.png"
        fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        figures.append(output_file)
        print(f"Saved Figure 6 panel to {output_file}")
    return figures


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "processed_data",
                        help="Prepared data directory containing time_series/ (default: project processed_data/).")
    parser.add_argument("--watersheds", nargs="+", default=["ISB", "NML", "YRS"],
                        help="CDEC watershed names or IDs (default: ISB NML YRS).")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE,
                        help=f"Inclusive start date in YYYY-MM-DD format (default: {DEFAULT_START_DATE}).")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE,
                        help=f"Inclusive end date in YYYY-MM-DD format (default: {DEFAULT_END_DATE}).")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "output", help="Directory for the three PNG panels.")
    parser.add_argument("--dpi", type=int, default=150, help="Figure resolution (default: 150).")
    args = parser.parse_args()
    try:
        plot_selected_watershed_monthly_climatology(**vars(args))
    except (FileNotFoundError, ValueError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    _main()
