import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sidecar2 import Sidecar, sidecar_loss

V, D, T, STEPS, N_WIN = 64, 32, 128, 300, 4


def make_windows(n, seed, label_from=None):
    g = torch.Generator().manual_seed(seed)
    hs, ys = [], []
    for _ in range(n):
        h = torch.randn(1, T, D, generator=g)
        hs.append(h)
        if label_from is None:
            ys.append(torch.randint(0, V, (1, T), generator=g))
        else:
            ys.append((h @ label_from.T).argmax(-1))
    return hs, ys


def reuse_run(cap, label_from=None, lr=3e-3):
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(7)
    head = nn.Linear(D, V, bias=False)
    with torch.no_grad():
        head.weight.normal_(0, 1.0, generator=g)
    for p in head.parameters():
        p.requires_grad_(False)
    train_hs, train_y = make_windows(N_WIN, 1, label_from)
    val_hs, val_y = make_windows(N_WIN, 2, label_from)
    w = torch.ones(1, T)
    sc = Sidecar(d_model=D, d=32, layers=2, rank=0, n_read=1, chunk=32, corr_max_rel=cap)
    opt = torch.optim.Adam(sc.parameters(), lr=lr)
    base_val = float(sidecar_loss(None, val_hs[0:1], head, val_y[0], w))
    peak_ratio = 0.0
    for step in range(STEPS):
        i = step % N_WIN
        opt.zero_grad()
        sidecar_loss(sc, train_hs[i:i + 1], head, train_y[i], w).backward()
        opt.step()
        with torch.no_grad():
            corr = sc(train_hs[i:i + 1])
            ratio = corr.pow(2).mean(-1).sqrt() / train_hs[i].pow(2).mean(-1).sqrt()
            peak_ratio = max(peak_ratio, float(ratio.max()))
    val_ce = float(sidecar_loss(sc, val_hs[0:1], head, val_y[0], w))
    return val_ce - base_val, peak_ratio


def test_uncapped_memorises_the_cache_and_hurts_val():
    delta, ratio = reuse_run(0.0)
    print("   uncapped: val delta %+.4f, correction reaches %.2fx hidden rms" % (delta, ratio))
    assert delta > 1.0, delta
    assert ratio > 1.5, ratio


def test_cap_bounds_the_correction():
    for cap in (0.1, 0.3):
        _, ratio = reuse_run(cap)
        print("   cap %.1f: correction peaks at %.3fx hidden rms" % (cap, ratio))
        assert ratio <= cap + 1e-3, (cap, ratio)


def test_cap_helps_more_as_it_tightens():
    deltas = [reuse_run(cap, label_from=head_labels())[0] for cap in (0.0, 0.3, 0.1)]
    print("   learnable labels: delta uncapped %+.4f, cap 0.3 %+.4f, cap 0.1 %+.4f" % tuple(deltas))
    assert deltas[0] > deltas[1] > deltas[2], deltas
    assert deltas[2] < 0.0, deltas


def head_labels():
    g = torch.Generator().manual_seed(7)
    head_w = torch.randn(V, D, generator=g)
    return head_w + 1.0 * torch.randn(V, D, generator=g)


if __name__ == "__main__":
    test_uncapped_memorises_the_cache_and_hurts_val()
    test_cap_bounds_the_correction()
    test_cap_helps_more_as_it_tightens()
    print("cache reuse: the cap is what keeps repeated windows from diverging")
