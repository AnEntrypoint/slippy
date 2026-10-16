import torch
import torch.nn as nn
import torch.nn.functional as F

from sidecar2 import (Sidecar, bounded_decay_log, cap_relative_rms, chunked_linear_recurrence,
                      count_params, sidecar_loss)

D_MODEL = 64
VOCAB = 97


def naive_recurrence(decay_log_per_token, write_gated_value):
    B, T, D = write_gated_value.shape
    state = write_gated_value.new_zeros(B, D)
    out = []
    for t in range(T):
        state = torch.exp(decay_log_per_token[:, t]) * state + write_gated_value[:, t]
        out.append(state)
    return torch.stack(out, 1)


def test_recurrence_matches_sequential():
    torch.manual_seed(0)
    for chunk in (1, 7, 128, 256):
        T = chunk * 2 + 3
        decay_log = torch.log(torch.empty(2, T, 16).uniform_(0.82, 1.0))
        write = torch.randn(2, T, 16)
        got = chunked_linear_recurrence(decay_log, write, chunk)
        want = naive_recurrence(decay_log, write)
        err = (got - want).abs().max().item()
        scale = want.abs().max().item()
        assert err / scale < 1e-5, (chunk, err, scale)
    print("recurrence matches sequential: chunk 1/7/128/256, rel err < 1e-5")


def test_decay_bounds():
    u = torch.linspace(-40, 40, 10001)
    decay = torch.exp(bounded_decay_log(u))
    lo, hi = decay.min().item(), decay.max().item()
    assert lo >= 0.82 - 1e-6 and hi <= 1.0 + 1e-6, (lo, hi)
    print(f"decay bounded in [{lo:.4f}, {hi:.4f}] over unbounded input [-40, 40]")


def test_zero_init_identity():
    torch.manual_seed(1)
    for rank in (0, 16):
        m = Sidecar(d_model=D_MODEL, d=32, layers=2, rank=rank, n_read=2, use_err=True)
        hs = [torch.randn(1, 40, D_MODEL) for _ in range(2)]
        err = torch.randn(1, 40, D_MODEL)
        corr = m(hs, err)
        assert corr.shape == (1, 40, D_MODEL)
        assert corr.abs().max().item() == 0.0, corr.abs().max().item()
    print("fresh sidecar emits exact zero correction: rank 0 and 16, use_err 1")


def test_loss_matches_reference():
    torch.manual_seed(2)
    torch.set_default_dtype(torch.float32)
    head = nn.Linear(D_MODEL, VOCAB, bias=False)
    h = torch.randn(1, 200, D_MODEL)
    labels = torch.randint(0, VOCAB, (1, 200))
    weights = torch.ones(1, 200)
    weights[:, :50] = 0
    base = sidecar_loss(None, [h], head, labels, weights)
    logits = head(h.reshape(-1, D_MODEL)).float()
    ce = F.cross_entropy(logits, labels.reshape(-1).long(), reduction="none")
    want = (ce * weights.reshape(-1)).sum() / weights.sum()
    assert torch.allclose(base, want, atol=1e-6), (base.item(), want.item())
    int32 = sidecar_loss(None, [h], head, labels.int(), weights)
    assert torch.allclose(int32, base, atol=1e-6)
    print(f"sidecar_loss baseline equals weighted CE ({base.item():.4f}), int32 labels accepted")


def test_correction_cap():
    torch.manual_seed(5)
    head = nn.Linear(D_MODEL, VOCAB, bias=False)
    h = torch.randn(1, 256, D_MODEL)
    labels = torch.randint(0, VOCAB, (1, 256))
    weights = torch.ones(1, 256)

    def rms_ratio(correction, hidden):
        return (correction.pow(2).mean(-1).sqrt() / hidden.pow(2).mean(-1).sqrt())

    huge = cap_relative_rms(100.0 * h, h, 0.25)
    assert torch.allclose(rms_ratio(huge, h), torch.full((1, 256), 0.25), atol=1e-5)
    small = 1e-3 * torch.randn(1, 256, D_MODEL)
    assert torch.allclose(cap_relative_rms(small, h, 0.1), small, atol=1e-9)

    for max_rel in (0.1, 0.3):
        m = Sidecar(d_model=D_MODEL, d=32, layers=2, rank=16, n_read=1, corr_max_rel=max_rel)
        assert m([h]).abs().max().item() == 0.0
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        for _ in range(50):
            opt.zero_grad()
            sidecar_loss(m, [h], head, labels, weights).backward()
            opt.step()
        worst = rms_ratio(m([h]), h).max().item()
        assert worst <= max_rel + 1e-6, (max_rel, worst)
        print(f"   corr_max_rel {max_rel}: 50 steps, worst per-token rms ratio {worst:.4f}")

    free = Sidecar(d_model=D_MODEL, d=32, layers=2, rank=16, n_read=1)
    opt = torch.optim.Adam(free.parameters(), lr=1e-2)
    for _ in range(50):
        opt.zero_grad()
        sidecar_loss(free, [h], head, labels, weights).backward()
        opt.step()
    print(f"   uncapped after the same 50 steps: worst ratio {rms_ratio(free([h]), h).max().item():.4f}")
    print("correction cap: zero-init preserved, per-token rms bounded, inactive while small")


def trainable_step_loss():
    torch.manual_seed(3)
    m = Sidecar(d_model=D_MODEL, d=32, layers=2, rank=16, n_read=1)
    head = nn.Linear(D_MODEL, VOCAB, bias=False)
    h = torch.randn(1, 256, D_MODEL)
    labels = torch.randint(0, VOCAB, (1, 256))
    weights = torch.ones(1, 256)
    before = sidecar_loss(m, [h], head, labels, weights).item()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(20):
        opt.zero_grad()
        loss = sidecar_loss(m, [h], head, labels, weights)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1e9).item()
        opt.step()
    after = sidecar_loss(m, [h], head, labels, weights).item()
    assert after < before
    assert torch.isfinite(torch.tensor(grad_norm))
    return before, after, count_params(m)


def test_fp16_headroom():
    torch.manual_seed(4)
    worst_decay_log = torch.full((1, 512, 8), torch.log(torch.tensor(0.82)))
    write = torch.ones(1, 512, 8)
    for chunk in (128, 256, 446):
        decay_cumsum = worst_decay_log[:, :chunk].cumsum(1)
        biggest = torch.exp(-decay_cumsum).max().item()
        fp16_ok = biggest < 65504
        print(f"   chunk {chunk}: worst-case exp(-cumsum) {biggest:.3e} "
              f"fp16 {'fits' if fp16_ok else 'OVERFLOWS'} (max 65504)")
        if chunk <= 256:
            got = chunked_linear_recurrence(worst_decay_log[:, :chunk], write[:, :chunk], chunk)
            assert torch.isfinite(got).all()
    print("fp32 scan survives the worst-case decay; fp16 would not")


if __name__ == "__main__":
    test_recurrence_matches_sequential()
    test_decay_bounds()
    test_zero_init_identity()
    test_loss_matches_reference()
    test_correction_cap()
    before, after, params = trainable_step_loss()
    print(f"20 adam steps: loss {before:.4f} -> {after:.4f}, params {params}")
    test_fp16_headroom()
