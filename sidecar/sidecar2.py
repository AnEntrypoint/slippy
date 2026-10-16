import torch
import torch.nn as nn
import torch.nn.functional as F

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


def chunked_linear_recurrence(decay_log_per_token, write_gated_value, chunk):
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


class RecBlock(nn.Module):
    def __init__(self, d, kernel=4, mult=4):
        super().__init__()
        mlp_width = mult * d
        self.pre_mix_norm = RMSNorm(d)
        self.conv = nn.Conv1d(d, d, kernel, groups=d, bias=False)
        self.kernel = kernel
        self.decay = nn.Linear(d, d)
        self.write = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.mlp_norm = RMSNorm(d)
        self.mlp_up = nn.Linear(d, mlp_width * 2)
        self.mlp_down = nn.Linear(mlp_width, d)

    def forward(self, x, chunk):
        y = self.pre_mix_norm(x)
        y = F.pad(y.transpose(1, 2), (self.kernel - 1, 0))
        y = self.conv(y).transpose(1, 2)
        decay_log_per_token = bounded_decay_log(self.decay(y))
        write_gated_value = torch.sigmoid(self.write(y)) * y
        output_gate = torch.sigmoid(self.gate(y))
        x = x + output_gate * self.out(chunked_linear_recurrence(decay_log_per_token, write_gated_value, chunk))
        swiglu_gate, swiglu_value = self.mlp_up(self.mlp_norm(x)).chunk(2, -1)
        return x + self.mlp_down(F.silu(swiglu_gate) * swiglu_value)


def cap_relative_rms(correction, hidden, max_rel):
    hidden_rms = hidden.float().pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    correction_rms = correction.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    return correction * torch.clamp(max_rel * hidden_rms / correction_rms, max=1.0)


class Sidecar(nn.Module):
    def __init__(self, d_model=1024, d=512, layers=4, rank=256, kernel=4, mult=4,
                 n_read=1, use_err=False, chunk=128, corr_max_rel=0.0):
        super().__init__()
        self.chunk = chunk
        self.use_err = use_err
        self.corr_max_rel = corr_max_rel
        self.read_proj = nn.Linear(d_model * n_read, d, bias=False)
        self.err_in = nn.Linear(d_model, d, bias=False) if use_err else None
        self.blocks = nn.ModuleList(RecBlock(d, kernel, mult) for _ in range(layers))
        self.norm = RMSNorm(d)
        if rank and rank < d_model:
            self.down = nn.Linear(d, rank, bias=False)
            self.up = nn.Linear(rank, d_model, bias=False)
        else:
            self.down = None
            self.up = nn.Linear(d, d_model, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hs, err=None):
        # hs: list of [B,T,D] with the LM-head layer last; err: [B,T,D] or None
        x = self.read_proj(torch.cat([h.float() for h in hs], -1))
        if self.use_err and err is not None:
            x = x + self.err_in(err.float())
        for blk in self.blocks:
            x = blk(x, self.chunk)
        y = self.norm(x)
        if self.down is not None:
            y = self.down(y)
        correction = self.up(y)
        if self.corr_max_rel > 0:
            correction = cap_relative_rms(correction, hs[-1], self.corr_max_rel)
        return correction


def sidecar_loss(sidecar, hs, lm_head, labels, weights, err=None, chunk_tokens=4096):
    # hs: list of [1,T,D] with the LM-head layer last; labels [1,T] so position t predicts labels[t]; weights [1,T] where 0 masks a token; err [1,T,D] or None; sidecar None gives the frozen-backbone baseline
    final_hidden = hs[-1]
    corrected = final_hidden.float() if sidecar is None else final_hidden.float() + sidecar(hs, err)
    flat = corrected.reshape(-1, corrected.shape[-1])
    tgt = labels.reshape(-1).long()
    w = weights.reshape(-1).float()
    total = flat.new_zeros(())
    for s in range(0, flat.shape[0], chunk_tokens):
        lg = lm_head(flat[s:s + chunk_tokens].to(lm_head.weight.dtype)).float()
        ce = F.cross_entropy(lg, tgt[s:s + chunk_tokens], reduction="none")
        total = total + (ce * w[s:s + chunk_tokens]).sum()
    return total / w.sum().clamp_min(1.0)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
