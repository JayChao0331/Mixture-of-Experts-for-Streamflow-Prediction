"""Independent daily inference from raw CDEC arrays for analysis regression tests."""

import numpy as np
import pandas as pd
import torch
import xarray as xr

from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.datasetzoo.genericdataset import load_attributes
from neuralhydrology.modelzoo import get_model


def reference_gate_table(cfg, run_dir, basin_ids, period):
    """Compare plotting's batched inference to the full model on manually built inputs."""
    scaler = load_scaler(run_dir)
    model = get_model(cfg).eval()
    model.load_state_dict(torch.load(run_dir / "model_epoch000.pt", map_location="cpu", weights_only=True))
    attributes = load_attributes(cfg.data_dir)[sorted(cfg.static_attributes)]
    attributes = (attributes - scaler["attribute_means"]) / scaler["attribute_stds"]
    dates = pd.date_range(getattr(cfg, period + "_start_date"), getattr(cfg, period + "_end_date"))
    history = pd.date_range(dates[0] - pd.Timedelta(days=cfg.seq_length - 1), dates[-1])
    gate_output = []
    hook = model.gating_net.register_forward_hook(lambda module, inputs, output: gate_output.append(output))
    frames = []
    try:
        for basin_id in basin_ids:
            with xr.open_dataset(cfg.data_dir / "time_series" / f"{basin_id}.nc") as raw:
                data = raw.astype(np.float32).reindex(date=history).load()
            normalized = (data - scaler["xarray_feature_center"]) / scaler["xarray_feature_scale"]
            inputs = normalized[cfg.dynamic_inputs].to_array().transpose("date", "variable").values.astype(np.float32)
            windows = np.stack([inputs[i:i + cfg.seq_length] for i in range(len(dates))])
            statics = attributes.loc[basin_id].to_numpy(dtype=np.float32)
            valid = np.isfinite(windows).all(axis=(1, 2)) & np.isfinite(statics).all()
            gates = np.full((len(dates), model.num_experts), np.nan, dtype=np.float32)
            if valid.any():
                batch = {"x_d": torch.from_numpy(windows[valid]),
                         "x_s": torch.from_numpy(np.tile(statics, (valid.sum(), 1)))}
                with torch.inference_mode():
                    model(model.pre_model_hook(batch, is_train=False))
                gates[valid] = gate_output.pop().numpy()
            target = cfg.target_variables[0]
            # Match the float32 round trip used when evaluation returns observations.
            obs = normalized[target].sel(date=dates).values.astype(np.float32)
            obs = obs * np.float32(scaler["xarray_feature_scale"][target].item())
            obs += np.float32(scaler["xarray_feature_center"][target].item())
            frame = pd.DataFrame(gates, columns=[f"expert_{i}_gate" for i in range(model.num_experts)])
            frame["qobs"], frame["date"], frame["basin_id"] = obs, dates, basin_id
            frames.append(frame)
    finally:
        hook.remove()
    return pd.concat(frames, ignore_index=True)
