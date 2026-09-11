from typing import Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config



class LSTMGateAttn(nn.Module):
    def __init__(
        self,
        d_in: int,
        num_experts: int,
        hidden: int = 64,
        num_layers: int = 1,
        temperature: float = 0.7,   # init τ0
        s_min: float = 1.0,         # clamp range for s = 1/τ
        s_max: float = 8.0,
    ):
        super().__init__()
        # ---- learnable global logit scale s = 1 / τ ----
        # map desired init τ0 -> s0, then to raw parameter via inverse-sigmoid
        s0 = 1.0 / max(temperature, 1e-6)
        s0 = float(min(max(s0, s_min), s_max))
        # inverse of sigmoid within [s_min, s_max]
        def inv_sigmoid_from_clamped(val, lo, hi):
            x = (val - lo) / (hi - lo)
            x = min(max(x, 1e-6), 1 - 1e-6)
            return math.log(x) - math.log(1 - x)
        self.s_min, self.s_max = s_min, s_max
        self.raw_scale = nn.Parameter(torch.tensor(inv_sigmoid_from_clamped(s0, s_min, s_max), dtype=torch.float))

        # encoder + attention
        self.lstm = nn.LSTM(input_size=d_in, hidden_size=hidden, num_layers=num_layers, bidirectional=False)
        self.attn_w = nn.Linear(hidden, hidden, bias=True)
        self.attn_v = nn.Linear(hidden, 1, bias=False)
        self.proj = nn.Linear(hidden, num_experts)

    @property
    def current_tau(self) -> float:
        # for logging: τ = 1 / s
        with torch.no_grad():
            s = self.s_min + (self.s_max - self.s_min) * torch.sigmoid(self.raw_scale)
            return float(1.0 / max(s.item(), 1e-6))

    def forward(self, x_d: torch.Tensor) -> torch.Tensor:
        """
        x_d: (T, B, D) — use ALL timesteps
        returns: probs (B, E)
        """
        h_seq, _ = self.lstm(x_d)                 # (T, B, H)
        H = h_seq.permute(1, 0, 2).contiguous()   # (B, T, H)

        scores = self.attn_v(torch.tanh(self.attn_w(H))).squeeze(-1)  # (B, T)
        alpha  = torch.softmax(scores, dim=1)                          # (B, T)

        context = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)          # (B, H)
        logits  = self.proj(context)                                    # (B, E)

        # learnable, clamped scale s in [s_min, s_max]
        s = self.s_min + (self.s_max - self.s_min) * torch.sigmoid(self.raw_scale)
        probs = F.softmax(s * logits, dim=-1)                           # (B, E)
        return probs



class MoETau(BaseModel):
    def __init__(self, cfg: Config, num_experts: int = 8):
        super(MoETau, self).__init__(cfg=cfg)

        self.num_experts = num_experts
        self.embedding_net = InputLayer(cfg)

        # Create multiple LSTM experts
        self.experts = nn.ModuleList([
            nn.LSTM(input_size=self.embedding_net.output_size, hidden_size=cfg.hidden_size)
            for _ in range(num_experts)
        ])

        self.gating_net = LSTMGateAttn(d_in=self.embedding_net.output_size, num_experts=num_experts, hidden=32, temperature=0.3)

        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size, activation="linear")
        self._reset_parameters()

    def _reset_parameters(self):
        if self.cfg.initial_forget_bias is not None:
            for expert in self.experts:
                expert.bias_hh_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_d = self.embedding_net(data)

        # Get expert weights from the gating network
        expert_weights = self.gating_net(x_d)

        expert_outputs = []
        for expert in self.experts:
            lstm_output, (_, _) = expert(x_d)
            expert_outputs.append(lstm_output)

        expert_outputs = torch.stack(expert_outputs, dim=2)

        expert_outputs = expert_outputs.permute(1, 0, 2, 3).contiguous()
        mixture_output = torch.sum(expert_weights.unsqueeze(1).unsqueeze(-1) * expert_outputs, dim=2)

        pred = {'mixture_output': mixture_output}
        pred.update(self.head(self.dropout(mixture_output)))

        return pred
