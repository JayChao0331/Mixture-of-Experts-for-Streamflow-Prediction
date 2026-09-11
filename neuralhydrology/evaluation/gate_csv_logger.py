import os, csv, math
from collections import deque
import torch
import numpy as np




class GateCSVLogger:
    """
    Logs per-sample gate probabilities to a CSV as the model runs.
    Columns: batch_idx, sample_idx, p_e1, p_e2, ..., p_eE
    """
    def __init__(self, model, csv_path: str):
        self.model = model
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

        # open file (append) and detect if header exists
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        self.f = open(csv_path, "a", newline="")
        self.w = csv.writer(self.f)

        self.num_experts = None
        self.batch_idx = 0

        # register forward hook on gating_net (expects output shape (B, E))
        self.handle = self.model.gating_net.register_forward_hook(self._hook)

        # remember whether header was present
        self._has_header = file_exists

    @torch.no_grad()
    def _hook(self, module, inputs, output):
        """
        Called every time gating_net runs during a forward pass.
        Writes one CSV row per sample in the batch.
        """
        probs = output.detach()
        if probs.dim() != 2:
            raise RuntimeError(f"Expected gate probs of shape (B, E), got {tuple(probs.shape)}")

        B, E = probs.shape
        if self.num_experts is None:
            self.num_experts = E
            if not self._has_header:
                header = ["batch_idx", "sample_idx"] + [f"p_e{i+1}" for i in range(E)]
                self.w.writerow(header)
                self._has_header = True

        probs_cpu = probs.cpu().numpy()  # (B, E)
        for i in range(B):
            row = [self.batch_idx, i] + probs_cpu[i].tolist()
            self.w.writerow(row)

        self.batch_idx += 1  # next batch

    def close(self):
        if getattr(self, "handle", None) is not None:
            self.handle.remove()
            self.handle = None
        if getattr(self, "f", None):
            self.f.flush()
            self.f.close()
