"""Train and evaluate MoE-tau using the prepared data; run from the project root."""

import argparse
from contextlib import closing
from datetime import datetime
import os
import random
import socket
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import pickle
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Support direct execution with `python3 runs/run_moe_tau.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import eval_run
from neuralhydrology.training.train import start_training
from neuralhydrology.utils.config import Config


data_dir = "./final_data_stage1_training_val_071824"
climate_nondet_dir = "./dynamic_inputs/climate_historical_non_detrended_unsplit_100yr"
target_variable_dir = "./target_variables/historical_observed_cleaned_15CDEC"
processed_data_dir = "./processed_data"

WATERSHED_AREA_DICT = {"BND": 23051,
                       "FOL": 4882,
                       "ISB": 5372,
                       "MIL": 4338,
                       "MKM": 1409,
                       "MRC": 2748,
                       "NHG": 940,
                       "NML": 2331,
                       "ORO": 9342,
                       "PNF": 4002,
                       "SCC": 1018,
                       "SHA": 17262,
                       "TLG": 3983,
                       "TRM": 1453,
                       "YRS": 2870}


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def load_dynamic_static_pbm_data():
    # generate watersheds & indices mapping
    watershed_index_dict = {}
    watershed_index_dframe = pd.read_csv(os.path.join(data_dir, "FNF_key.csv"))

    for index, row in watershed_index_dframe.iterrows():
        watershed_index_dict[row['FNF']] = row['ID']

    start_date_lst = []
    end_date_lst = []
    for watershed in watershed_index_dict.keys():
        # Read dynamic input (avg)
        climate_nondet_avg_fname = "00_nonDet_100yr_ws_avg_{}_1.csv".format(watershed)
        climate_nondet_avg_dframe = pd.read_csv(os.path.join(data_dir, climate_nondet_dir, "mean", climate_nondet_avg_fname), index_col=0, parse_dates=True)
        climate_nondet_avg_dframe.index.name = None
        climate_nondet_avg_dframe = climate_nondet_avg_dframe.drop(['basin', 'year', 'month', 'day', 'scenario'], axis=1)

        # Read dynamic input (std)
        climate_nondet_std_fname = "00_nonDet_100yr_std_dev_{}_1.csv".format(watershed)
        climate_nondet_std_dframe = pd.read_csv(os.path.join(data_dir, climate_nondet_dir, "stddev", climate_nondet_std_fname), index_col=0, parse_dates=True)
        climate_nondet_std_dframe.index.name = None
        climate_nondet_std_dframe = climate_nondet_std_dframe.drop(['basin', 'year', 'month', 'day', 'scenario'], axis=1)
        climate_nondet_std_dframe = climate_nondet_std_dframe.fillna(0)

        # Read dynamic input (elband)
        climate_nondet_elband_fname = "00_nonDet_100yr_el_bands_{}_1.csv".format(watershed)
        climate_nondet_elband_dframe = pd.read_csv(os.path.join(data_dir, climate_nondet_dir, "elband", climate_nondet_elband_fname), index_col=0, parse_dates=True)
        climate_nondet_elband_dframe.index.name = None
        climate_nondet_elband_dframe = climate_nondet_elband_dframe.drop(['basin', 'year', 'month', 'day', 'scenario'], axis=1)
        climate_nondet_elband_dframe = climate_nondet_elband_dframe.fillna(0)


        # Read PBM outputs
        pbm_pred_fname = "simflow_sacsma_{}_short.txt".format(watershed)
        pbm_dframe = pd.read_csv(os.path.join(data_dir, "pbm_streamflow", "climate_historical", pbm_pred_fname), sep=r'\s+', header=None)
        pbm_dframe['date'] = pd.to_datetime(pbm_dframe[[0, 1, 2]].astype(str).agg('-'.join, axis=1))
        pbm_dframe = pbm_dframe[['date', 3]]
        pbm_dframe = pbm_dframe.rename(columns={3: 'discharge_pbm'})
        pbm_dframe.index = pd.to_datetime(pbm_dframe['date'])
        pbm_dframe = pbm_dframe.drop(columns='date')
        pbm_dframe.index.name = None


        # Read target variable
        target_variable_fname = "FNF_{}_cfs.txt".format(watershed)
        column_names = ['year', 'month', 'day', 'value', 'date']

        target_variable_dframe = pd.read_csv(
                    os.path.join(data_dir, target_variable_dir, target_variable_fname),
                    sep='\s+', 
                    header=None, 
                    names=column_names, 
                    na_values=['', ' '], 
                    keep_default_na=False, 
                    engine='python'
        )

        target_variable_dframe['date'] = pd.to_datetime(target_variable_dframe[['year', 'month', 'day']])
        target_variable_dframe.set_index('date', inplace=True)
        target_variable_dframe['value'] = pd.to_numeric(target_variable_dframe['value'], errors='coerce')
        target_variable_dframe.rename(columns={'value': 'discharge'}, inplace=True)
        target_variable_dframe.index.name = None
        target_variable_dframe = target_variable_dframe.drop(['year', 'month', 'day'], axis=1)
        target_variable_dframe = target_variable_dframe.map(lambda x: np.nan if pd.isna(x) else x)

        target_variable_dframe['discharge'] = target_variable_dframe['discharge'] * 2.446576 / WATERSHED_AREA_DICT[watershed]



        dates_df1 = pd.to_datetime(climate_nondet_avg_dframe.index)
        dates_df2 = pd.to_datetime(climate_nondet_std_dframe.index)
        dates_df3 = pd.to_datetime(climate_nondet_elband_dframe.index)
        dates_df4 = pd.to_datetime(pbm_dframe.index)
        dates_df5 = pd.to_datetime(target_variable_dframe.index)
        overlapping_dates = dates_df1.intersection(dates_df2).intersection(dates_df3).intersection(dates_df4).intersection(dates_df5)


        climate_nondet_avg_filtered_dframe = climate_nondet_avg_dframe.loc[overlapping_dates]
        climate_nondet_std_filtered_dframe = climate_nondet_std_dframe.loc[overlapping_dates]
        climate_nondet_elband_filtered_dframe = climate_nondet_elband_dframe.loc[overlapping_dates]
        pbm_filtered_dframe = pbm_dframe.loc[overlapping_dates]
        target_variable_filtered_dframe = target_variable_dframe.loc[overlapping_dates]


        merged_dframe = pd.merge(climate_nondet_avg_filtered_dframe, climate_nondet_std_filtered_dframe, left_index=True, right_index=True)
        merged_dframe = pd.merge(merged_dframe, climate_nondet_elband_filtered_dframe, left_index=True, right_index=True)
        merged_dframe = pd.merge(merged_dframe, pbm_filtered_dframe, left_index=True, right_index=True)
        merged_dframe = pd.merge(merged_dframe, target_variable_filtered_dframe, left_index=True, right_index=True)

        # Convert the DataFrame to an xarray Dataset
        dataset_nc = xr.Dataset.from_dataframe(merged_dframe)

        # Explicitly set the date index as a coordinate
        dataset_nc = dataset_nc.rename({'index': 'date'})
        dataset_nc = dataset_nc.set_coords('date')

        # Save the Dataset to a NetCDF file
        watershed_id = watershed_index_dict[watershed]
        dataset_nc.to_netcdf(os.path.join(processed_data_dir, 'time_series', '{}.nc'.format(watershed_id)))

    return watershed_index_dict



CONFIG_FILE = Path(__file__).resolve().with_name("moe_tau.yml")
HELDOUT_CONFIG_DIR = Path("heldout_configs")
HELDOUT_SPLITS = {
    "sha": {"name": "SHA", "id": "1"},
    "oro": {"name": "ORO", "id": "3"},
    "yrs": {"name": "YRS", "id": "4"},
    "fol": {"name": "FOL", "id": "5"},
    "nml": {"name": "NML", "id": "8"},
    "tlg": {"name": "TLG", "id": "9"},
    "mrc": {"name": "MRC", "id": "10"},
    "pnf": {"name": "PNF", "id": "12"},
    "trm": {"name": "TRM", "id": "13"},
    "scc": {"name": "SCC", "id": "14"},
    "isb": {"name": "ISB", "id": "15"},
}
AREA_FOLDS = {
    "area1": {"name": "Area 1", "watersheds": ["SHA", "ORO", "YRS", "FOL"], "ids": ["1", "3", "4", "5"]},
    "area2": {"name": "Area 2", "watersheds": ["MKM", "NHG", "NML", "TLG", "MRC", "MIL"], "ids": ["6", "7", "8", "9", "10", "11"]},
    "area3": {"name": "Area 3", "watersheds": ["PNF", "TRM", "SCC", "ISB"], "ids": ["12", "13", "14", "15"]},
}


def _get_watershed_index_dict() -> dict:
    watershed_index_dframe = pd.read_csv(os.path.join(data_dir, "FNF_key.csv"))
    return {str(row["FNF"]).upper(): str(row["ID"]) for _, row in watershed_index_dframe.iterrows()}


def _resolve_watershed_name(watershed: str) -> str:
    watershed = str(watershed).upper()
    watershed_index_dict = _get_watershed_index_dict()

    if watershed in watershed_index_dict:
        return watershed

    id_to_watershed = {basin_id: name for name, basin_id in watershed_index_dict.items()}
    if watershed in id_to_watershed:
        return id_to_watershed[watershed]

    valid = ", ".join(sorted(watershed_index_dict.keys()))
    raise ValueError(f"Unknown watershed '{watershed}'. Use a watershed name or ID. Known names: {valid}.")


def _load_observed_discharge_mm_per_day(watershed: str) -> pd.DataFrame:
    watershed = _resolve_watershed_name(watershed)
    target_variable_fname = f"FNF_{watershed}_cfs.txt"
    column_names = ["year", "month", "day", "value", "date"]

    discharge = pd.read_csv(os.path.join(data_dir, target_variable_dir, target_variable_fname),
                            sep=r"\s+",
                            header=None,
                            names=column_names,
                            na_values=["", " "],
                            keep_default_na=False,
                            engine="python")
    discharge["date"] = pd.to_datetime(discharge[["year", "month", "day"]])
    discharge["discharge_cfs"] = pd.to_numeric(discharge["value"], errors="coerce")
    discharge["discharge_mm_per_day"] = discharge["discharge_cfs"] * 2.446576 / WATERSHED_AREA_DICT[watershed]
    return discharge[["date", "discharge_cfs", "discharge_mm_per_day"]].set_index("date")


def print_observed_discharge_values(watershed: str,
                                    start_date: str = None,
                                    end_date: str = None,
                                    max_rows: int = 20) -> pd.DataFrame:
    """Print observed discharge for one watershed in the transformed mm/day scale.

    Parameters
    ----------
    watershed : str
        Watershed name, e.g. "SHA", or basin ID, e.g. "1".
    start_date : str, optional
        Optional first date to print, parseable by pandas.
    end_date : str, optional
        Optional last date to print, parseable by pandas.
    max_rows : int, optional
        Maximum number of daily rows to print. Set to None to print the full selected record.
    """
    watershed_name = _resolve_watershed_name(watershed)
    discharge = _load_observed_discharge_mm_per_day(watershed_name)

    if start_date is not None:
        discharge = discharge.loc[pd.Timestamp(start_date):]
    if end_date is not None:
        discharge = discharge.loc[:pd.Timestamp(end_date)]

    summary = discharge["discharge_mm_per_day"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"\nObserved discharge for {watershed_name} in transformed scale (mm/day)")
    print(summary.to_string())

    rows_to_print = discharge if max_rows is None else discharge.head(max_rows)
    print("\nDaily values:")
    print(rows_to_print.to_string())

    if max_rows is not None and len(discharge) > max_rows:
        print(f"\nPrinted first {max_rows} of {len(discharge)} rows. Set max_rows=None to print all rows.")

    return discharge


def _get_first_gpu_id(cfg: Config) -> int:
    if cfg.device is not None and cfg.device.startswith("cuda:"):
        return int(cfg.device.split(":")[-1])
    return 0


def _get_gpu_ids(cfg: Config):
    gpu_ids = cfg.as_dict().get("gpu_ids", None)
    if gpu_ids is not None:
        gpu_ids = [int(gpu_id) for gpu_id in gpu_ids]
        if not gpu_ids:
            raise ValueError("gpu_ids must not be empty when provided.")
        return gpu_ids

    num_gpus = cfg.num_gpus
    first_gpu_id = _get_first_gpu_id(cfg)
    return list(range(first_gpu_id, first_gpu_id + num_gpus))


def _configure_nccl_environment(cfg: Config = None):
    """Set safer NCCL defaults for single-node multi-GPU training."""
    cfg_dict = cfg.as_dict() if cfg is not None else {}

    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

    nccl_debug = cfg_dict.get("nccl_debug", None)
    if nccl_debug is not None:
        os.environ["NCCL_DEBUG"] = str(nccl_debug)

    flag_to_env = {
        "nccl_p2p_disable": "NCCL_P2P_DISABLE",
        "nccl_ib_disable": "NCCL_IB_DISABLE",
        "nccl_shm_disable": "NCCL_SHM_DISABLE",
    }
    for cfg_key, env_key in flag_to_env.items():
        if cfg_key in cfg_dict:
            os.environ[env_key] = "1" if bool(cfg_dict[cfg_key]) else "0"


def _get_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


def _make_run_name(cfg: Config) -> str:
    now = datetime.now()
    day = f"{now.day}".zfill(2)
    month = f"{now.month}".zfill(2)
    hour = f"{now.hour}".zfill(2)
    minute = f"{now.minute}".zfill(2)
    second = f"{now.second}".zfill(2)
    return f"{cfg.experiment_name}_{day}{month}_{hour}{minute}{second}"


def _get_expected_run_dir(cfg: Config, run_name: str) -> Path:
    if cfg.run_dir is None:
        return Path().cwd() / "runs" / run_name
    return cfg.run_dir / run_name


def _write_config_file(cfg: Config, config_file: Path):
    config_file.parent.mkdir(parents=True, exist_ok=True)
    if config_file.exists():
        config_file.unlink()
    cfg.dump_config(folder=config_file.parent, filename=config_file.name)


def _start_training_from_config(config_file: Path, seed: int = None, gpu: int = None) -> Path:
    cfg = Config(config_file)
    if gpu is not None:
        cfg.update_config({"device": "cpu" if gpu < 0 else f"cuda:{gpu}", "num_gpus": 1, "gpu_ids": None},
                          dev_mode=True)
    gpu_ids = _get_gpu_ids(cfg)
    num_gpus = len(gpu_ids)

    if num_gpus < 1:
        raise ValueError("num_gpus must be at least 1.")

    run_name = _make_run_name(cfg)
    run_dir = _get_expected_run_dir(cfg, run_name)

    if num_gpus == 1:
        cfg.update_config({"run_name": run_name})
        if seed is not None:
            cfg.update_config({"seed": int(seed)})
        start_training(cfg)
        return run_dir

    if not torch.cuda.is_available():
        raise RuntimeError("num_gpus > 1 requires CUDA GPUs.")

    available_gpus = torch.cuda.device_count()
    invalid_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= available_gpus]
    if invalid_gpu_ids:
        raise RuntimeError(f"Requested invalid GPU id(s) {invalid_gpu_ids}, "
                           f"but this machine only has {available_gpus} CUDA device(s).")

    master_port = _get_free_port()
    _configure_nccl_environment(cfg)
    mp.spawn(_distributed_worker,
             args=(gpu_ids, str(config_file), run_name, master_port, seed),
             nprocs=num_gpus,
             join=True)

    return run_dir


def _distributed_worker(rank: int,
                        gpu_ids,
                        config_file: str,
                        run_name: str,
                        master_port: int,
                        seed: int = None):
    world_size = len(gpu_ids)
    gpu_id = int(gpu_ids[rank])
    torch.cuda.set_device(gpu_id)
    print(f"[rank {rank}] using cuda:{gpu_id} ({torch.cuda.get_device_name(gpu_id)})", flush=True)

    init_kwargs = {
        "backend": "nccl",
        "init_method": f"tcp://127.0.0.1:{master_port}",
        "world_size": world_size,
        "rank": rank,
    }
    try:
        dist.init_process_group(**init_kwargs, device_id=torch.device(f"cuda:{gpu_id}"))
    except TypeError:
        # Older PyTorch versions do not support the device_id argument.
        dist.init_process_group(**init_kwargs)

    try:
        cfg = Config(Path(config_file))
        cfg.device = f"cuda:{gpu_id}"
        cfg.update_config({
            "distributed_rank": rank,
            "distributed_local_rank": gpu_id,
            "distributed_world_size": world_size,
            "distributed_backend": "nccl",
            "run_name": run_name,
        })
        if seed is not None:
            cfg.update_config({"seed": int(seed)})

        start_training(cfg)
    finally:
        dist.destroy_process_group()


def debug_distributed_gpu_setup(num_gpus: int = None, gpu_ids=None):
    """Run a minimal NCCL all-reduce test without loading data or training the model."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    cfg = None
    if gpu_ids is None:
        if num_gpus is None:
            cfg = Config(CONFIG_FILE)
            gpu_ids = _get_gpu_ids(cfg)
        else:
            gpu_ids = list(range(num_gpus))
    else:
        gpu_ids = [int(gpu_id) for gpu_id in gpu_ids]

    available_gpus = torch.cuda.device_count()
    invalid_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= available_gpus]
    if invalid_gpu_ids:
        raise RuntimeError(f"Requested invalid GPU id(s) {invalid_gpu_ids}, "
                           f"but this machine only has {available_gpus} CUDA device(s).")

    master_port = _get_free_port()
    _configure_nccl_environment(cfg)
    mp.spawn(_distributed_debug_worker,
             args=(gpu_ids, master_port),
             nprocs=len(gpu_ids),
             join=True)


def _distributed_debug_worker(rank: int, gpu_ids, master_port: int):
    world_size = len(gpu_ids)
    gpu_id = int(gpu_ids[rank])
    torch.cuda.set_device(gpu_id)
    print(f"[debug rank {rank}] using cuda:{gpu_id} ({torch.cuda.get_device_name(gpu_id)})", flush=True)

    init_kwargs = {
        "backend": "nccl",
        "init_method": f"tcp://127.0.0.1:{master_port}",
        "world_size": world_size,
        "rank": rank,
    }
    try:
        dist.init_process_group(**init_kwargs, device_id=torch.device(f"cuda:{gpu_id}"))
    except TypeError:
        dist.init_process_group(**init_kwargs)

    try:
        tensor = torch.tensor([rank + 1.0], device=f"cuda:{gpu_id}")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        try:
            dist.barrier(device_ids=[gpu_id])
        except TypeError:
            dist.barrier()
        print(f"[debug rank {rank}] all_reduce result: {tensor.item()}", flush=True)
    finally:
        dist.destroy_process_group()


def run_experiments(times=10, gpu: int = None):
    """Train repeated runs from the preserved MoE-tau configuration."""
    if times < 1:
        raise ValueError("times must be at least 1.")
    run_dirs = []
    for _ in range(times):
        run_dir = _start_training_from_config(CONFIG_FILE, gpu=gpu)
        run_dirs.append(run_dir)
        print(f"Finished MoE-tau training run: {run_dir}")
    return run_dirs


def _read_basin_ids(basin_file: Path):
    with basin_file.open("r") as fp:
        return [line.strip() for line in fp if line.strip()]


def _write_basin_ids(basin_file: Path, basin_ids):
    basin_file.write_text("\n".join(basin_ids) + "\n")


def _get_heldout_split(split_key: str) -> tuple:
    split_key = str(split_key).lower()
    aliases = {}
    for key, split_info in HELDOUT_SPLITS.items():
        aliases[key] = key
        aliases[split_info["id"]] = key

    if split_key not in aliases:
        valid = ", ".join(sorted(aliases.keys()))
        raise ValueError(f"Unknown held-out split '{split_key}'. Use one of: {valid}.")

    canonical_key = aliases[split_key]
    return canonical_key, HELDOUT_SPLITS[canonical_key]


def _prepare_heldout_config(split_key: str, split_info: dict) -> Path:
    all_basin_file = Path("cdec_nondet_basin.txt")
    all_basin_ids = _read_basin_ids(all_basin_file)
    heldout_id = split_info["id"]

    if heldout_id not in all_basin_ids:
        raise ValueError(f"Held-out basin id {heldout_id} is not listed in {all_basin_file}.")

    train_ids = [basin_id for basin_id in all_basin_ids if basin_id != heldout_id]
    train_basin_file = Path(f"cdec_nondet_basin_train_without_{split_key}.txt")
    test_basin_file = Path(f"cdec_nondet_basin_test_{split_key}.txt")

    _write_basin_ids(train_basin_file, train_ids)
    _write_basin_ids(test_basin_file, [heldout_id])

    cfg = Config(CONFIG_FILE)
    cfg.update_config({
        "experiment_name": f"{cfg.experiment_name}_holdout_{split_info['name']}",
        "train_basin_file": train_basin_file,
        "validation_basin_file": train_basin_file,
        "test_basin_file": test_basin_file,
    })

    heldout_config_file = HELDOUT_CONFIG_DIR / f"moe_tau_holdout_{split_key}.yml"
    _write_config_file(cfg, heldout_config_file)
    return heldout_config_file


def _get_area_fold(fold_key: str) -> tuple:
    fold_key = str(fold_key).lower().replace("_", "").replace("-", "")
    aliases = {
        "area1": "area1",
        "1": "area1",
        "area2": "area2",
        "2": "area2",
        "area3": "area3",
        "3": "area3",
    }

    if fold_key not in aliases:
        valid = ", ".join(sorted(aliases.keys()))
        raise ValueError(f"Unknown area fold '{fold_key}'. Use one of: {valid}.")

    canonical_key = aliases[fold_key]
    return canonical_key, AREA_FOLDS[canonical_key]


def _prepare_area_fold_config(fold_key: str, fold_info: dict) -> Path:
    all_basin_file = Path("cdec_nondet_basin.txt")
    all_basin_ids = _read_basin_ids(all_basin_file)
    test_ids = fold_info["ids"]
    missing_ids = [basin_id for basin_id in test_ids if basin_id not in all_basin_ids]

    if missing_ids:
        raise ValueError(f"Area fold basin id(s) {missing_ids} are not listed in {all_basin_file}.")

    train_ids = [basin_id for basin_id in all_basin_ids if basin_id not in test_ids]
    train_basin_file = Path(f"cdec_nondet_basin_train_without_{fold_key}.txt")
    test_basin_file = Path(f"cdec_nondet_basin_test_{fold_key}.txt")

    _write_basin_ids(train_basin_file, train_ids)
    _write_basin_ids(test_basin_file, test_ids)

    cfg = Config(CONFIG_FILE)
    cfg.update_config({
        "experiment_name": f"{cfg.experiment_name}_area_fold_{fold_info['name'].replace(' ', '')}",
        "train_basin_file": train_basin_file,
        "validation_basin_file": train_basin_file,
        "test_basin_file": test_basin_file,
    })

    area_config_file = HELDOUT_CONFIG_DIR / f"moe_tau_area_fold_{fold_key}.yml"
    _write_config_file(cfg, area_config_file)
    return area_config_file


def _get_weight_stem(run_dir: Path, epoch: int = None) -> str:
    if epoch is None:
        return sorted(list(run_dir.glob("model_epoch*.pt")))[-1].stem
    return f"model_epoch{epoch:03d}"


def _evaluate_single_heldout_basin(run_dir: Path, basin_id: str, epoch: int = None) -> dict:
    eval_run(run_dir=run_dir, period="test", epoch=epoch)

    weight_stem = _get_weight_stem(run_dir=run_dir, epoch=epoch)
    with open(run_dir / "test" / weight_stem / "test_results.p", "rb") as fp:
        results = pickle.load(fp)

    qobs = results[basin_id]["1D"]["xr"]["discharge_obs"]
    qsim = results[basin_id]["1D"]["xr"]["discharge_sim"]
    return metrics.calculate_all_metrics(qobs.isel(time_step=-1), qsim.isel(time_step=-1))


def _evaluate_area_fold(run_dir: Path, basin_ids, epoch: int = None) -> pd.DataFrame:
    eval_run(run_dir=run_dir, period="test", epoch=epoch)

    weight_stem = _get_weight_stem(run_dir=run_dir, epoch=epoch)
    with open(run_dir / "test" / weight_stem / "test_results.p", "rb") as fp:
        results = pickle.load(fp)

    basin_metrics = []
    for basin_id in basin_ids:
        qobs = results[basin_id]["1D"]["xr"]["discharge_obs"]
        qsim = results[basin_id]["1D"]["xr"]["discharge_sim"]
        values = metrics.calculate_all_metrics(qobs.isel(time_step=-1), qsim.isel(time_step=-1))
        values["basin_id"] = basin_id
        basin_metrics.append(values)

    return pd.DataFrame(basin_metrics).set_index("basin_id")


def _print_metric_table(title: str, table: pd.DataFrame):
    print(f"\n{title}")
    if table.empty:
        print("No metrics available.")
    else:
        print(table.to_string())


def _load_heldout_results(run_dir: Path, basin_id: str, epoch: int = None, force_eval: bool = False):
    weight_stem = _get_weight_stem(run_dir=run_dir, epoch=epoch)
    result_file = run_dir / "test" / weight_stem / "test_results.p"

    if force_eval or not result_file.exists():
        eval_run(run_dir=run_dir, period="test", epoch=epoch)

    with open(result_file, "rb") as fp:
        results = pickle.load(fp)

    return results[basin_id]["1D"]["xr"], weight_stem


def _add_water_year_day(df: pd.DataFrame) -> pd.DataFrame:
    # Use a leap water-year template so Feb 29 has its own slot and dates after Feb 29 are not shifted.
    water_year_days = pd.date_range("1999-10-01", "2000-09-30", freq="D")
    month_day_to_wy_day = {date.strftime("%m-%d"): idx + 1 for idx, date in enumerate(water_year_days)}
    df = df.copy()
    df["water_year_day"] = df["date"].dt.strftime("%m-%d").map(month_day_to_wy_day)
    return df


def plot_heldout_seasonal_hydrograph(split_key: str,
                                     run_dir: Path,
                                     epoch: int = None,
                                     output_file: Path = None,
                                     force_eval: bool = False,
                                     clip_negative_predictions: bool = True,
                                     show: bool = False) -> pd.DataFrame:
    """Plot observed vs. MoE-tau seasonal hydrograph for a held-out watershed.

    The daily series is aggregated over the full held-out test record by day of water year,
    where day 1 is Oct 1 and the final day is Sep 30. Lines show the daily mean and shaded
    bands show the 5th to 95th percentile range.
    """
    _, split_info = _get_heldout_split(split_key)
    run_dir = Path(run_dir)
    heldout_id = split_info["id"]
    basin_results, weight_stem = _load_heldout_results(run_dir=run_dir,
                                                       basin_id=heldout_id,
                                                       epoch=epoch,
                                                       force_eval=force_eval)

    # These arrays are already denormalized to the transformed target scale, e.g. mm/day.
    qobs = basin_results["discharge_obs"].isel(time_step=-1)
    qsim = basin_results["discharge_sim"].isel(time_step=-1)
    moe_tau_values = qsim.values
    if clip_negative_predictions:
        n_negative = int(np.sum(moe_tau_values < 0))
        if n_negative > 0:
            print(f"Clipped {n_negative} negative MoE-\u03c4 predictions to zero before plotting.")
        moe_tau_values = np.maximum(moe_tau_values, 0.0)

    hydrograph = pd.DataFrame({
        "date": pd.to_datetime(qobs["date"].values),
        "observed": qobs.values,
        "MoE-\u03c4": moe_tau_values,
    })
    hydrograph = hydrograph.dropna(subset=["observed", "MoE-\u03c4"], how="all")
    hydrograph = _add_water_year_day(hydrograph)

    grouped = hydrograph.groupby("water_year_day")
    seasonal = pd.DataFrame({
        "observed": grouped["observed"].mean(),
        "observed_5%": grouped["observed"].quantile(0.05),
        "observed_95%": grouped["observed"].quantile(0.95),
        "MoE-\u03c4": grouped["MoE-\u03c4"].mean(),
        "MoE-\u03c4_5%": grouped["MoE-\u03c4"].quantile(0.05),
        "MoE-\u03c4_95%": grouped["MoE-\u03c4"].quantile(0.95),
    })
    seasonal = seasonal.reindex(range(1, 367))

    if output_file is None:
        output_file = f"./heldout_{split_info['name'].lower()}_seasonal_hydrograph_{weight_stem}.png"
    else:
        output_file = Path(output_file)

    month_ticks = {
        1: "Oct",
        32: "Nov",
        62: "Dec",
        93: "Jan",
        124: "Feb",
        153: "Mar",
        184: "Apr",
        214: "May",
        245: "Jun",
        275: "Jul",
        306: "Aug",
        337: "Sep",
    }

    fig, ax = plt.subplots(figsize=(11, 5))
    x_values = seasonal.index.to_numpy()
    observed_color = "grey"
    model_color = "green"
    ax.fill_between(x_values,
                    seasonal["observed_5%"].to_numpy(dtype=float),
                    seasonal["observed_95%"].to_numpy(dtype=float),
                    color=observed_color,
                    alpha=0.18,
                    linewidth=0)
    ax.fill_between(x_values,
                    seasonal["MoE-\u03c4_5%"].to_numpy(dtype=float),
                    seasonal["MoE-\u03c4_95%"].to_numpy(dtype=float),
                    color=model_color,
                    alpha=0.20,
                    linewidth=0)
    ax.plot(seasonal.index, seasonal["observed"], label="Observed", color=observed_color, linewidth=2)
    ax.plot(seasonal.index, seasonal["MoE-\u03c4"], label="MoE-\u03c4", color=model_color, linewidth=2)
    ax.set_xlim(1, 366)
    ax.set_ylim(bottom=0)
    ax.set_xticks(list(month_ticks.keys()))
    ax.set_xticklabels(list(month_ticks.values()), fontweight="bold")
    ax.set_xlabel("Day of water year", fontweight="bold")
    ax.set_ylabel("Daily discharge (mm/day)", fontweight="bold")
    ax.set_title(f"Held-out {split_info['name']} seasonal hydrograph", fontweight="bold")
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    ax.grid(True, alpha=0.3)
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved seasonal hydrograph to {output_file}")
    return seasonal


def train_heldout_watershed_experiment(split_key: str,
                                       times: int = 1,
                                       vary_seed: bool = True,
                                       base_seed: int = SEED):
    """Train one held-out watershed experiment.

    Parameters
    ----------
    split_key : str
        Held-out split to train. Use "sha"/"1", "oro"/"3", or "tlg"/"9".
    times : int, optional
        Number of independent training runs to launch for the selected split.
    vary_seed : bool, optional
        If True, use a different deterministic seed for each repeated run.
    base_seed : int, optional
        Base seed used when vary_seed is True. Run i uses base_seed + i.
    """
    split_key, split_info = _get_heldout_split(split_key)
    heldout_config_file = _prepare_heldout_config(split_key=split_key, split_info=split_info)
    run_dirs = []

    for run_idx in range(times):
        run_seed = int(base_seed + run_idx) if vary_seed else None
        if run_seed is not None:
            print(f"Starting held-out {split_info['name']} training run {run_idx + 1}/{times} with seed {run_seed}")
        run_dir = _start_training_from_config(heldout_config_file, seed=run_seed)
        run_dirs.append(run_dir)
        print(f"Finished held-out {split_info['name']} training run: {run_dir}")

    return run_dirs


def test_heldout_watershed_experiment(split_key: str, run_dir: Path, epoch: int = None):
    """Evaluate one selected held-out run directory.

    Parameters
    ----------
    split_key : str
        Held-out split to evaluate. Use "sha"/"1", "oro"/"3", or "tlg"/"9".
    run_dir : Path
        Run directory containing the selected checkpoint files.
    epoch : int, optional
        Specific model epoch to evaluate. If None, eval_run uses the latest model_epoch*.pt file.
    """
    _, split_info = _get_heldout_split(split_key)
    run_dir = Path(run_dir)
    values = _evaluate_single_heldout_basin(run_dir=run_dir, basin_id=split_info["id"], epoch=epoch)

    print(f"\nHeld-out {split_info['name']} ({split_info['id']}) results")
    print(f"Run dir: {run_dir}")
    print(f"Checkpoint: {_get_weight_stem(run_dir=run_dir, epoch=epoch)}")
    for metric_name, metric_value in values.items():
        print(f"{metric_name}: {metric_value}")

    return values


def test_all_heldout_watershed_checkpoints(split_key: str,
                                           runs_dir: Path = "./runs",
                                           epoch: int = None,
                                           run_name_contains: str = None,
                                           skip_failed: bool = True):
    """Evaluate all checkpoint folders in a runs directory for one held-out watershed.

    Parameters
    ----------
    split_key : str
        Held-out split to evaluate. Use "sha"/"1", "oro"/"3", or "tlg"/"9".
    runs_dir : Path, optional
        Directory containing run/checkpoint folders. Defaults to "./runs".
    epoch : int, optional
        Specific model epoch to evaluate for every run. If None, eval_run uses the latest checkpoint.
    run_name_contains : str, optional
        Optional substring filter for run folder names, e.g. "holdout_ORO".
    skip_failed : bool, optional
        If True, print failed runs and continue. If False, raise the first error.
    """
    _, split_info = _get_heldout_split(split_key)
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")

    run_dirs = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_name_contains is not None and run_name_contains not in run_dir.name:
            continue
        if not list(run_dir.glob("model_epoch*.pt")):
            continue
        run_dirs.append(run_dir)

    if not run_dirs:
        print(f"No checkpoint folders found in {runs_dir}.")
        return pd.DataFrame(), pd.DataFrame()

    metric_rows = []
    failed_runs = []

    for run_dir in run_dirs:
        print(f"\nEvaluating held-out {split_info['name']} checkpoint folder: {run_dir}")
        try:
            values = _evaluate_single_heldout_basin(run_dir=run_dir, basin_id=split_info["id"], epoch=epoch)
        except Exception as exc:
            if not skip_failed:
                raise
            failed_runs.append({"run_dir": str(run_dir), "error": str(exc)})
            print(f"Skipping failed run: {exc}")
            continue

        row = values.copy()
        row.update({
            "run_dir": str(run_dir),
            "checkpoint": _get_weight_stem(run_dir=run_dir, epoch=epoch),
        })
        metric_rows.append(row)

    if not metric_rows:
        print("\nNo runs were evaluated successfully.")
        if failed_runs:
            _print_metric_table("Failed runs", pd.DataFrame(failed_runs))
        return pd.DataFrame(), pd.DataFrame()

    run_metrics = pd.DataFrame(metric_rows).set_index("run_dir")
    metric_columns = run_metrics.select_dtypes(include=[np.number]).columns
    std_ddof = 1 if len(run_metrics) > 1 else 0
    summary = pd.DataFrame({
        "mean": run_metrics[metric_columns].mean(),
        "std": run_metrics[metric_columns].std(ddof=std_ddof),
    })

    _print_metric_table(f"Held-out {split_info['name']} metrics for each checkpoint folder", run_metrics)
    _print_metric_table(f"Held-out {split_info['name']} summary across checkpoint folders", summary)

    if failed_runs:
        _print_metric_table("Failed runs", pd.DataFrame(failed_runs))

    return run_metrics, summary


def test_all_heldout_watershed_checkpoints_by_metric(split_key: str,
                                                     runs_dir: Path = "./runs",
                                                     epoch: int = None,
                                                     run_name_contains: str = None,
                                                     skip_failed: bool = True):
    """Evaluate all checkpoint folders and print each metric's checkpoint values as a list."""
    _, split_info = _get_heldout_split(split_key)
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")

    run_dirs = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_name_contains is not None and run_name_contains not in run_dir.name:
            continue
        if not list(run_dir.glob("model_epoch*.pt")):
            continue
        run_dirs.append(run_dir)

    if not run_dirs:
        print(f"No checkpoint folders found in {runs_dir}.")
        return

    metric_rows = []
    failed_runs = []

    for run_dir in run_dirs:
        print(f"\nEvaluating held-out {split_info['name']} checkpoint folder: {run_dir}")
        try:
            values = _evaluate_single_heldout_basin(run_dir=run_dir, basin_id=split_info["id"], epoch=epoch)
        except Exception as exc:
            if not skip_failed:
                raise
            failed_runs.append({"run_dir": str(run_dir), "error": str(exc)})
            print(f"Skipping failed run: {exc}")
            continue

        row = values.copy()
        row.update({
            "run_dir": str(run_dir),
            "checkpoint": _get_weight_stem(run_dir=run_dir, epoch=epoch),
        })
        metric_rows.append(row)

    if not metric_rows:
        print("\nNo runs were evaluated successfully.")
        if failed_runs:
            _print_metric_table("Failed runs", pd.DataFrame(failed_runs))
        return

    run_metrics = pd.DataFrame(metric_rows).set_index("run_dir")
    metric_columns = run_metrics.select_dtypes(include=[np.number]).columns

    print(f"\nHeld-out {split_info['name']} MoE-tau metric values across checkpoint folders")
    for metric_name in metric_columns:
        values = run_metrics[metric_name].tolist()
        print(f"{metric_name}: {values}")

    if failed_runs:
        _print_metric_table("Failed runs", pd.DataFrame(failed_runs))


def print_checkpoint_nse_fhv_flv(split_key: str,
                                 runs_dir: Path = "./runs",
                                 epoch: int = None,
                                 run_name_contains: str = None,
                                 skip_failed: bool = True):
    """Evaluate all checkpoint folders and print each checkpoint's NSE, FHV, and FLV."""
    _, split_info = _get_heldout_split(split_key)
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")

    run_dirs = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_name_contains is not None and run_name_contains.lower() not in run_dir.name.lower():
            continue
        if not list(run_dir.glob("model_epoch*.pt")):
            continue
        run_dirs.append(run_dir)

    if not run_dirs:
        print(f"No checkpoint folders found in {runs_dir}.")
        return

    rows = []
    failed_runs = []

    for run_dir in run_dirs:
        print(f"\nEvaluating held-out {split_info['name']} MoE-tau checkpoint folder: {run_dir}")
        try:
            values = _evaluate_single_heldout_basin(run_dir=run_dir, basin_id=split_info["id"], epoch=epoch)
        except Exception as exc:
            if not skip_failed:
                raise
            failed_runs.append({"run_dir": str(run_dir), "error": str(exc)})
            print(f"Skipping failed run: {exc}")
            continue

        missing_metrics = [metric for metric in ["NSE", "FHV", "FLV"] if metric not in values]
        if missing_metrics:
            raise KeyError(f"Missing metric(s) {missing_metrics} for {run_dir}. Available metrics: {list(values.keys())}")

        rows.append({
            "checkpoint_folder": run_dir.name,
            "checkpoint": _get_weight_stem(run_dir=run_dir, epoch=epoch),
            "NSE": values["NSE"],
            "FHV": values["FHV"],
            "FLV": values["FLV"],
        })

    if not rows:
        print("\nNo runs were evaluated successfully.")
        if failed_runs:
            _print_metric_table("Failed runs", pd.DataFrame(failed_runs))
        return

    result_table = pd.DataFrame(rows)
    _print_metric_table(f"Held-out {split_info['name']} MoE-tau checkpoint NSE/FHV/FLV", result_table)

    if failed_runs:
        _print_metric_table("Failed runs", pd.DataFrame(failed_runs))


def test_top_nse_heldout_watershed_checkpoints(split_key: str,
                                               runs_dir: Path = "./runs",
                                               epoch: int = None,
                                               run_name_contains: str = None,
                                               top_k: int = 6,
                                               skip_failed: bool = True):
    """Evaluate all checkpoint folders and summarize the top-k checkpoints by NSE/FHV/FLV score.

    Ranking score:
    - higher NSE is better.
    - smaller absolute FHV is better.
    - smaller absolute FLV is better.

    Each component is min-max normalized across the evaluated checkpoints so that higher is better,
    then the three normalized components are averaged.
    """
    _, split_info = _get_heldout_split(split_key)
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")

    run_dirs = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_name_contains is not None and run_name_contains.lower() not in run_dir.name.lower():
            continue
        if not list(run_dir.glob("model_epoch*.pt")):
            continue
        run_dirs.append(run_dir)

    if not run_dirs:
        print(f"No checkpoint folders found in {runs_dir}.")
        return

    metric_rows = []
    failed_runs = []

    for run_dir in run_dirs:
        print(f"\nEvaluating held-out {split_info['name']} checkpoint folder: {run_dir}")
        try:
            values = _evaluate_single_heldout_basin(run_dir=run_dir, basin_id=split_info["id"], epoch=epoch)
        except Exception as exc:
            if not skip_failed:
                raise
            failed_runs.append({"run_dir": str(run_dir), "error": str(exc)})
            print(f"Skipping failed run: {exc}")
            continue

        row = values.copy()
        row.update({
            "run_dir": str(run_dir),
            "checkpoint": _get_weight_stem(run_dir=run_dir, epoch=epoch),
        })
        metric_rows.append(row)

    print(f"\nTotal checkpoint folders found: {len(run_dirs)}")
    print(f"Successfully evaluated checkpoint folders: {len(metric_rows)}")
    print(f"Failed checkpoint folders: {len(failed_runs)}")

    if not metric_rows:
        print("\nNo runs were evaluated successfully.")
        if failed_runs:
            _print_metric_table("Failed runs", pd.DataFrame(failed_runs))
        return

    run_metrics = pd.DataFrame(metric_rows).set_index("run_dir")
    ranking_metrics = ["NSE", "FHV", "FLV"]
    missing_metrics = [metric for metric in ranking_metrics if metric not in run_metrics.columns]
    if missing_metrics:
        raise KeyError(f"Missing ranking metric(s) {missing_metrics}. Available metrics: {list(run_metrics.columns)}")

    def _normalize_higher_is_better(values: pd.Series) -> pd.Series:
        values = values.astype(float)
        value_range = values.max() - values.min()
        if value_range == 0:
            return pd.Series(1.0, index=values.index)
        return (values - values.min()) / value_range

    def _normalize_closer_to_zero_is_better(values: pd.Series) -> pd.Series:
        errors = values.astype(float).abs()
        error_range = errors.max() - errors.min()
        if error_range == 0:
            return pd.Series(1.0, index=errors.index)
        return 1.0 - (errors - errors.min()) / error_range

    score_table = pd.DataFrame(index=run_metrics.index)
    score_table["checkpoint"] = run_metrics["checkpoint"]
    score_table["NSE"] = run_metrics["NSE"]
    score_table["FHV"] = run_metrics["FHV"]
    score_table["FLV"] = run_metrics["FLV"]
    score_table["NSE_score"] = _normalize_higher_is_better(run_metrics["NSE"])
    score_table["FHV_abs"] = run_metrics["FHV"].astype(float).abs()
    score_table["FHV_score"] = _normalize_closer_to_zero_is_better(run_metrics["FHV"])
    score_table["FLV_abs"] = run_metrics["FLV"].astype(float).abs()
    score_table["FLV_score"] = _normalize_closer_to_zero_is_better(run_metrics["FLV"])
    score_table["ranking_score"] = score_table[["NSE_score", "FHV_score", "FLV_score"]].mean(axis=1)

    n_selected = min(top_k, len(run_metrics))
    top_score_table = score_table.sort_values("ranking_score", ascending=False).head(n_selected)
    top_metrics = run_metrics.loc[top_score_table.index].copy()
    metric_columns = top_metrics.select_dtypes(include=[np.number]).columns
    std_ddof = 1 if len(top_metrics) > 1 else 0
    summary = pd.DataFrame({
        "mean": top_metrics[metric_columns].mean(),
        "std": top_metrics[metric_columns].std(ddof=std_ddof),
    })

    print(f"\nSelected top {n_selected} checkpoint folders by combined NSE/FHV/FLV score "
          f"out of {len(run_metrics)} successfully evaluated folders.")
    print("Ranking score = mean(normalized NSE, normalized closeness of FHV to 0, "
          "normalized closeness of FLV to 0).")
    print(f"Top {n_selected} ranking scores: {top_score_table['ranking_score'].tolist()}")

    _print_metric_table(f"Held-out {split_info['name']} MoE-tau top-{n_selected} checkpoint ranking details",
                        top_score_table)
    _print_metric_table(f"Held-out {split_info['name']} MoE-tau top-{n_selected} checkpoint metrics",
                        top_metrics)
    _print_metric_table(f"Held-out {split_info['name']} MoE-tau top-{n_selected} NSE/FHV/FLV-score summary",
                        summary)

    if failed_runs:
        _print_metric_table("Failed runs", pd.DataFrame(failed_runs))


def run_heldout_watershed_experiments(split_key: str,
                                      times: int = 1,
                                      vary_seed: bool = True,
                                      base_seed: int = SEED):
    """Backward-compatible wrapper for held-out training only.

    Testing is intentionally separate. Use test_heldout_watershed_experiment() with the run directory you select.
    """
    return train_heldout_watershed_experiment(split_key=split_key,
                                              times=times,
                                              vary_seed=vary_seed,
                                              base_seed=base_seed)


def run_area_fold_train_test_experiments(fold_keys=None):
    """Run 3-fold area holdout experiments.

    Each fold trains on two areas and immediately evaluates on the remaining area. The summary
    reports mean and standard deviation across fold-level area-average metrics.

    Parameters
    ----------
    fold_keys : list, optional
        Optional subset/order of folds to run. Use "area1"/"1", "area2"/"2", or "area3"/"3".
        If None, all three folds are run.
    """
    if fold_keys is None:
        fold_keys = ["area1", "area2", "area3"]

    fold_rows = []

    for fold_key in fold_keys:
        fold_key, fold_info = _get_area_fold(fold_key)
        area_config_file = _prepare_area_fold_config(fold_key=fold_key, fold_info=fold_info)

        print(f"\nRunning {fold_info['name']} holdout experiment")
        print(f"Training on all watersheds except: {', '.join(fold_info['watersheds'])}")
        print(f"Testing on: {', '.join(fold_info['watersheds'])} ({', '.join(fold_info['ids'])})")

        run_dir = _start_training_from_config(area_config_file)

        basin_metrics = _evaluate_area_fold(run_dir=run_dir, basin_ids=fold_info["ids"])
        _print_metric_table(f"{fold_info['name']} basin-level test metrics", basin_metrics)

        fold_mean = basin_metrics.mean(numeric_only=True)
        fold_row = fold_mean.to_dict()
        fold_row.update({
            "fold": fold_key,
            "heldout_area": fold_info["name"],
            "heldout_watersheds": ", ".join(fold_info["watersheds"]),
            "run_dir": str(run_dir),
        })
        fold_rows.append(fold_row)

    fold_metrics = pd.DataFrame(fold_rows).set_index("fold")
    metric_columns = fold_metrics.select_dtypes(include=[np.number]).columns
    summary = pd.DataFrame({
        "mean": fold_metrics[metric_columns].mean(),
        "std": fold_metrics[metric_columns].std(ddof=1),
    })

    _print_metric_table("Area-fold mean metrics for each held-out area", fold_metrics)
    _print_metric_table("3-fold area holdout summary (mean/std across held-out areas)", summary)

def evaluate_lstm():
    ########## Evaluation ##########
    run_dirs = sorted(path for path in Path("runs").iterdir()
                      if path.is_dir() and any(path.glob("model_epoch*.pt")))

    NSE_mean_lst = []
    MSE_mean_lst = []
    RMSE_mean_lst = []
    KGE_mean_lst = []
    FHV_mean_lst = []
    FLV_mean_lst = []
    Pearson_mean_lst = []
    PeakMape_mean_lst = []
    PeakTiming_mean_lst = []

    for run_dir_path in run_dirs:
        eval_run(run_dir=run_dir_path, period="test")

        ########## Load and inspect model predictions ##########
        with open(run_dir_path / "test" / "model_epoch000" / "test_results.p", "rb") as fp:
            results = pickle.load(fp)

        basin_ids = ['1', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
        
        qobs_lst = []
        qsim_lst = []
        for basin_id in basin_ids:
            qobs_lst.append(results[basin_id]['1D']['xr']['discharge_obs'])
            qsim_lst.append(results[basin_id]['1D']['xr']['discharge_sim'])


        NSE_lst = []
        MSE_lst = []
        RMSE_lst = []
        KGE_lst = []
        FHV_lst = []
        FLV_lst = []
        Pearson_lst = []
        PeakMape_lst = []
        PeakTiming_lst = []

        for id in range(len(basin_ids)):
            qobs = qobs_lst[id]
            qsim = qsim_lst[id]

            values = metrics.calculate_all_metrics(qobs.isel(time_step=-1), qsim.isel(time_step=-1))

            NSE_lst.append(values['NSE'])
            MSE_lst.append(values['MSE'])
            RMSE_lst.append(values['RMSE'])
            KGE_lst.append(values['KGE'])
            FHV_lst.append(values['FHV'])
            FLV_lst.append(values['FLV'])
            Pearson_lst.append(values['Pearson-r'])
            PeakMape_lst.append(values['Peak-MAPE'])
            PeakTiming_lst.append(values['Peak-Timing'])

        NSE_mean_lst.append(np.mean(NSE_lst))
        MSE_mean_lst.append(np.mean(MSE_lst))
        RMSE_mean_lst.append(np.mean(RMSE_lst))
        KGE_mean_lst.append(np.mean(KGE_lst))
        FHV_mean_lst.append(np.mean(FHV_lst))
        FLV_mean_lst.append(np.mean(FLV_lst))
        Pearson_mean_lst.append(np.mean(Pearson_lst))
        PeakMape_mean_lst.append(np.mean(PeakMape_lst))
        PeakTiming_mean_lst.append(np.mean(PeakTiming_lst))

    # Calculate mean & std of the runs
    print("NSE: {} ± {}".format(np.mean(NSE_mean_lst), np.std(NSE_mean_lst)))
    print("MSE: {} ± {}".format(np.mean(MSE_mean_lst), np.std(MSE_mean_lst)))
    print("RMSE: {} ± {}".format(np.mean(RMSE_mean_lst), np.std(RMSE_mean_lst)))
    print("KGE: {} ± {}".format(np.mean(KGE_mean_lst), np.std(KGE_mean_lst)))
    print("FHV: {} ± {}".format(np.mean(FHV_mean_lst), np.std(FHV_mean_lst)))
    print("FLV: {} ± {}".format(np.mean(FLV_mean_lst), np.std(FLV_mean_lst)))
    print("Pearson-r: {} ± {}".format(np.mean(Pearson_mean_lst), np.std(Pearson_mean_lst)))
    print("PeakMAPE: {} ± {}".format(np.mean(PeakMape_mean_lst), np.std(PeakMape_mean_lst)))
    print("PeakTiming: {} ± {}".format(np.mean(PeakTiming_mean_lst), np.std(PeakTiming_mean_lst)))


def evaluate_lstm_train():
    """Evaluate all run folders on the training period for all watersheds.

    This is analogous to ``evaluate_lstm()``, but evaluates ``period="train"`` and reads
    ``train_results.p``. Metrics are first averaged across all basins for each run, then
    summarized across run folders with mean and standard deviation.
    """
    runs_dir = Path("./runs")
    basin_file = Path("cdec_nondet_basin.txt")
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")

    basin_ids = _read_basin_ids(basin_file)
    run_dirs = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if not list(run_dir.glob("model_epoch*.pt")):
            continue
        run_dirs.append(run_dir)

    if not run_dirs:
        print(f"No checkpoint folders found in {runs_dir}.")
        return

    run_rows = []
    failed_runs = []

    for run_dir in run_dirs:
        print(f"\nEvaluating training period for checkpoint folder: {run_dir}")
        try:
            eval_run(run_dir=run_dir, period="train")
            weight_stem = _get_weight_stem(run_dir=run_dir)
            results_file = run_dir / "train" / weight_stem / "train_results.p"
            with results_file.open("rb") as fp:
                results = pickle.load(fp)

            basin_rows = []
            for basin_id in basin_ids:
                if basin_id not in results:
                    continue
                qobs = results[basin_id]["1D"]["xr"]["discharge_obs"]
                qsim = results[basin_id]["1D"]["xr"]["discharge_sim"]
                values = metrics.calculate_all_metrics(qobs.isel(time_step=-1), qsim.isel(time_step=-1))
                basin_rows.append(values)

            if not basin_rows:
                raise ValueError(f"No requested basin IDs were found in {results_file}.")

            basin_metrics = pd.DataFrame(basin_rows)
            run_row = basin_metrics.mean(numeric_only=True).to_dict()
            run_row.update({
                "run_dir": str(run_dir),
                "checkpoint": weight_stem,
                "n_basins": len(basin_rows),
            })
            run_rows.append(run_row)

        except Exception as exc:
            failed_runs.append({"run_dir": str(run_dir), "error": str(exc)})
            print(f"Skipping failed run: {exc}")

    if not run_rows:
        print("\nNo runs were evaluated successfully.")
        if failed_runs:
            _print_metric_table("Failed runs", pd.DataFrame(failed_runs))
        return

    run_metrics = pd.DataFrame(run_rows).set_index("run_dir")
    metric_columns = run_metrics.select_dtypes(include=[np.number]).columns.drop("n_basins", errors="ignore")
    std_ddof = 1 if len(run_metrics) > 1 else 0
    summary = pd.DataFrame({
        "mean": run_metrics[metric_columns].mean(),
        "std": run_metrics[metric_columns].std(ddof=std_ddof),
    })

    _print_metric_table("Training-period metrics averaged across all watersheds for each checkpoint folder",
                        run_metrics)
    _print_metric_table("Training-period summary across checkpoint folders", summary)

    if failed_runs:
        _print_metric_table("Failed runs", pd.DataFrame(failed_runs))


def save_prediction(watershed_index_dict, basin_id='1'):
    ########## Evaluation ##########
    run_dirs = sorted(path for path in Path("runs").iterdir()
                      if path.is_dir() and any(path.glob("model_epoch*.pt")))
    if not run_dirs:
        raise FileNotFoundError("No trained checkpoint directories found in runs/.")
    run_dir_path = run_dirs[0]

    eval_run(run_dir=run_dir_path, period="test")

    ########## Load and inspect model predictions ##########
    with open(run_dir_path / "test" / "model_epoch000" / "test_results.p", "rb") as fp:
        results = pickle.load(fp)

    basin_id = str(basin_id)
    qobs = results[basin_id]['1D']['xr']['discharge_obs']
    qsim = results[basin_id]['1D']['xr']['discharge_sim']

    encoding_sim = {"discharge_sim": {"zlib": True, "dtype": "float32"}}
    qsim.to_netcdf("moe_tau_sim_{}.nc".format(basin_id), encoding=encoding_sim)

    encoding_obs = {"discharge_obs": {"zlib": True, "dtype": "float32"}}
    qobs.to_netcdf("obs_{}.nc".format(basin_id), encoding=encoding_obs)

    id_to_basin = {str(v): k for k, v in watershed_index_dict.items()}
    basin_name = id_to_basin[basin_id]
    pbm_pred_fname = f"simflow_sacsma_{basin_name}_short.txt"
    pbm_path = Path(data_dir) / "pbm_streamflow" / "climate_historical" / pbm_pred_fname
    pbm_df = pd.read_csv(pbm_path, sep=r"\s+", header=None)
    pbm_df["date"] = pd.to_datetime(pbm_df[[0, 1, 2]].astype(str).agg("-".join, axis=1), format="%Y-%m-%d")

    # Build 1D DataArray over 'date' so it matches qobs/qsim structure
    pbm_da = xr.DataArray(
        data=pbm_df[3].to_numpy(dtype=np.float32),
        coords={"date": pbm_df["date"].to_numpy()},
        dims=("date",),
        name="discharge_sim"   # <- save PBM as 'discharge_sim' so loaders pick it up directly
    )

    # Align PBM to the exact test date range used by qobs/qsim
    start_date = pd.Timestamp(qobs["date"].min().item())
    end_date   = pd.Timestamp(qobs["date"].max().item())
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Reindex to full range; if PBM is missing some days, fill with NaN (keeps lengths consistent)
    pbm_da = pbm_da.reindex(date=full_range)

    # Save PBM in the same format as sim
    pbm_da.to_netcdf(f"pbm_sim_{basin_id}.nc", encoding={"discharge_sim": {"zlib": True, "dtype": "float32"}})


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=["train-test", "train", "evaluate"], default="train-test",
                        help="Action to perform (default: train-test, trains then evaluates on the test period).")
    parser.add_argument("--times", type=int, default=1, help="Number of training runs (default: 1).")
    parser.add_argument("--run-dir", type=Path, help="Trained run directory to evaluate.")
    parser.add_argument("--period", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--epoch", type=int, help="Checkpoint epoch to evaluate (default: latest).")
    parser.add_argument("--gpu", type=int, help="Override the configured GPU; use -1 for CPU.")
    args = parser.parse_args()

    if args.times < 1:
        parser.error("--times must be at least 1.")
    if args.mode in {"train-test", "train"}:
        if args.run_dir is not None:
            parser.error("--run-dir is only used with evaluate.")
        run_dirs = run_experiments(times=args.times, gpu=args.gpu)
        if args.mode == "train-test":
            for run_dir in run_dirs:
                eval_run(run_dir=run_dir, period=args.period, epoch=args.epoch, gpu=args.gpu)
    else:
        if args.run_dir is None:
            parser.error("evaluate requires --run-dir.")
        eval_run(run_dir=args.run_dir, period=args.period, epoch=args.epoch, gpu=args.gpu)


if __name__ == '__main__':
    _main()
