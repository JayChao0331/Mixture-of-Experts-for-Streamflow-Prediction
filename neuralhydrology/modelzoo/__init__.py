"""Model factory for the architectures included in this repository."""
import warnings
from typing import Dict, Tuple

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.arlstm import ARLSTM
from neuralhydrology.modelzoo.cudalstm import CudaLSTM
from neuralhydrology.modelzoo.customlstm import CustomLSTM
from neuralhydrology.modelzoo.ealstm import EALSTM
from neuralhydrology.modelzoo.embcudalstm import EmbCudaLSTM
from neuralhydrology.modelzoo.handoff_forecast_lstm import HandoffForecastLSTM
from neuralhydrology.modelzoo.hybridmodel import HybridModel
from neuralhydrology.modelzoo.mamba import Mamba
from neuralhydrology.modelzoo.mclstm import MCLSTM
from neuralhydrology.modelzoo.moe_tau import MoETau
from neuralhydrology.modelzoo.mtslstm import MTSLSTM
from neuralhydrology.modelzoo.multihead_forecast_lstm import MultiHeadForecastLSTM
from neuralhydrology.modelzoo.odelstm import ODELSTM
from neuralhydrology.modelzoo.sequential_forecast_lstm import SequentialForecastLSTM
from neuralhydrology.utils.config import Config


MODEL_REGISTRY = {
    "arlstm": ARLSTM,
    "cudalstm": CudaLSTM,
    "customlstm": CustomLSTM,
    "ealstm": EALSTM,
    "embcudalstm": EmbCudaLSTM,
    "handoff_forecast_lstm": HandoffForecastLSTM,
    "hybrid_model": HybridModel,
    "mamba": Mamba,
    "mclstm": MCLSTM,
    "moe_tau": MoETau,
    "mtslstm": MTSLSTM,
    "multihead_forecast_lstm": MultiHeadForecastLSTM,
    "odelstm": ODELSTM,
    "sequential_forecast_lstm": SequentialForecastLSTM
}
SINGLE_FREQ_MODELS = [name for name in MODEL_REGISTRY if name not in {"mtslstm", "odelstm"}]
AUTOREGRESSIVE_MODELS = ["arlstm"]


def get_model(cfg: Config) -> nn.Module:
    """Create the configured model and report its parameter counts.

    Model names are case insensitive. Unsupported names raise an error listing
    the architectures available in this repository.
    """
    model_name = cfg.model.lower()
    if model_name == "lstm":
        warnings.warn("The `lstm` model name is deprecated; use `customlstm` instead.",
                      FutureWarning, stacklevel=2)
        model_name = "customlstm"

    if model_name not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY)
        raise NotImplementedError(f"Model '{cfg.model}' is not available. Supported models: {supported}.")

    if model_name in SINGLE_FREQ_MODELS and len(cfg.use_frequencies) > 1:
        raise ValueError(f"Model {cfg.model} does not support multiple frequencies.")
    if model_name not in AUTOREGRESSIVE_MODELS and cfg.autoregressive_inputs:
        raise ValueError(f"Model {cfg.model} does not support autoregression.")
    if model_name != "mclstm" and cfg.mass_inputs:
        raise ValueError(f"The use of 'mass_inputs' with {cfg.model} is not supported.")

    model = MODEL_REGISTRY[model_name](cfg=cfg)
    summarize_model_params(model, trainable_only=True)
    return model


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Return the total number of (trainable) parameters in `model`."""
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if (not trainable_only) or p.requires_grad:
            total += p.numel()
    return total

def count_parameters_by_child(model: torch.nn.Module, trainable_only: bool = True) -> Dict[str, int]:
    """
    Return a dict mapping each *top-level* child module name to its parameter count.
    Parameters registered directly on `model` (not inside a child) are reported under '<root>'.
    """
    top_level = dict(model.named_children())

    # Count per child (avoid double-count across children if sharing happens)
    seen = set()
    per_child = {name: 0 for name in top_level.keys()}
    root_count = 0

    for name, child in top_level.items():
        for p in child.parameters(recurse=True):
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            if trainable_only and not p.requires_grad:
                continue
            per_child[name] += p.numel()

    # Now handle parameters registered directly on the root module
    for p in model.parameters(recurse=False):
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        if trainable_only and not p.requires_grad:
            continue
        root_count += p.numel()

    if root_count:
        per_child['<root>'] = root_count
    return per_child

def human_readable(n: int) -> str:
    for unit in ['','K','M','B']:
        if abs(n) < 1000:
            return f"{n:.0f}{unit}"
        n /= 1000.0
    return f"{n:.0f}T"

def summarize_model_params(model: torch.nn.Module, trainable_only: bool = True) -> Tuple[int, Dict[str, int]]:
    total = count_parameters(model, trainable_only=trainable_only)
    by_child = count_parameters_by_child(model, trainable_only=trainable_only)
    # pretty print
    mode = "trainable" if trainable_only else "all"
    print(f"Parameter summary ({mode}):")
    width = max((len(k) for k in by_child.keys()), default=5)
    for name, cnt in sorted(by_child.items(), key=lambda x: -x[1]):
        print(f"  {name:<{width}} : {cnt:,}  ({human_readable(cnt)})")
    print(f"  {'TOTAL':<{width}} : {total:,}  ({human_readable(total)})")
    return total, by_child
