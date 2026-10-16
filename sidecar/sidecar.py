import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

MIN_DECAY = 0.82
DECAY_SPAN = 0.18


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def bounded_decay_log(unbounded_decay):
    return torch.log(MIN_DECAY + DECAY_SPAN * torch.sigmoid(unbounded_decay))


def gated_recurrence(decay_log_per_token, write_gated_value, chunk=256):
    B, T, D = write_gated_value.shape
    carried_state = write_gated_value.new_zeros(B, D)
    states = []
    for s in range(0, T, chunk):
        decay_cumsum = decay_log_per_token[:, s:s + chunk].cumsum(1)
        decayed_writes = torch.exp(-decay_cumsum) * write_gated_value[:, s:s + chunk]
        state = torch.exp(decay_cumsum) * (carried_state.unsqueeze(1) + torch.cumsum(decayed_writes, 1))
        states.append(state)
        carried_state = state[:, -1]
    return torch.cat(states, 1)


class RecurrentBlock(nn.Module):
    def __init__(self, d, kernel=4, mult=4):
        super().__init__()
        mlp_width = mult * d
        self.norm1 = RMSNorm(d)
        self.conv = nn.Conv1d(d, d, kernel, groups=d)
        self.kernel = kernel
        self.decay = nn.Linear(d, d)
        self.write = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.norm2 = RMSNorm(d)
        self.mlp_up = nn.Linear(d, mlp_width * 2)
        self.mlp_down = nn.Linear(mlp_width, d)

    def forward(self, x):
        y = self.norm1(x)
        y = F.pad(y.transpose(1, 2), (self.kernel - 1, 0))
        y = self.conv(y).transpose(1, 2)
        decay_log_per_token = bounded_decay_log(self.decay(y))
        write_gated_value = torch.sigmoid(self.write(y)) * y
        x = x + self.out(gated_recurrence(decay_log_per_token, write_gated_value))
        swiglu_gate, swiglu_value = self.mlp_up(self.norm2(x)).chunk(2, -1)
        return x + self.mlp_down(F.silu(swiglu_gate) * swiglu_value)


class Sidecar(nn.Module):
    def __init__(self, d_model=1024, d=512, layers=4, rank=256, kernel=4, mult=4):
        super().__init__()
        self.inp = nn.Linear(d_model, d)
        self.blocks = nn.ModuleList(RecurrentBlock(d, kernel, mult) for _ in range(layers))
        self.norm = RMSNorm(d)
        self.down = nn.Linear(d, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, h):
        x = self.inp(h)
        for blk in self.blocks:
            x = blk(x)
        return self.up(self.down(self.norm(x)))


def sidecar_loss(sidecar, hidden, lm_head, labels, weights=None, chunk_tokens=2048):
    # hidden [B,T,D]; labels [B,T] so position t predicts labels[t]; weights [B,T] where 0 masks a token, None for all ones; sidecar None gives the frozen-backbone baseline
    hidden = hidden.float()
    corrected = hidden if sidecar is None else hidden + sidecar(hidden)
    flat = corrected.reshape(-1, corrected.shape[-1])
    tgt = labels.reshape(-1)
    w = torch.ones_like(tgt, dtype=torch.float32) if weights is None else weights.reshape(-1).float()
    total = flat.new_zeros(())
    for s in range(0, flat.shape[0], chunk_tokens):
        def piece(x, y, ww):
            ce = F.cross_entropy(lm_head(x.to(lm_head.weight.dtype)).float(), y, reduction="none")
            return (ce * ww).sum()
        total = total + checkpoint(piece, flat[s:s + chunk_tokens], tgt[s:s + chunk_tokens], w[s:s + chunk_tokens], use_reentrant=False)
    return total / w.sum().clamp_min(1.0)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
