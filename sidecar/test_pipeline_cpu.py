import hashlib
import random
import types

import torch
import torch.nn as nn

import accel
import train2
from sidecar2 import Sidecar, sidecar_loss

D_MODEL = 32
VOCAB = 53
LAYERS = 4
LENGTH = 64


class StubBackbone:
    def __init__(self):
        self.config = types.SimpleNamespace(hidden_size=D_MODEL, vocab_size=VOCAB)
        self.lm_head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.model = self

    def get_input_embeddings(self):
        return self.emb

    def __call__(self, x, output_hidden_states=False):
        base = self.emb(x)
        states = [base + torch.randn_like(base) * 0.1 * (i + 1) for i in range(LAYERS + 1)]
        return types.SimpleNamespace(hidden_states=states)


def fake_seqs(n_rows, rng):
    seqs = []
    for _ in range(n_rows):
        n = rng.randrange(5, 220)
        ids = [rng.randrange(VOCAB) for _ in range(n)]
        assistant = [rng.random() < 0.8 for _ in range(n)]
        seqs.append((ids, assistant))
    return seqs


def one_window_check(backbone, seqs, mask, pack):
    cache = train2.alloc_cache(1, LENGTH, D_MODEL, 1, torch.float32)
    train2.fill_cache(cache, backbone, seqs, range(1), LENGTH, random.Random(7), mask, pack,
                      "cpu", [LAYERS])
    replicated_ids, replicated_weights = train2.sample_window(
        seqs, LENGTH, random.Random(7), mask, pack)
    assert cache["targets"][0].tolist() == replicated_ids[1:], (mask, pack)
    assert cache["weights"][0].tolist() == replicated_weights, (mask, pack)
    assert cache["argmax"][0].shape == (LENGTH,)
    assert cache["base_ce"][0] > 0 and torch.isfinite(torch.tensor(cache["base_ce"][0]))
    assert torch.isfinite(cache["hidden"][0][0]).all()
    return sum(replicated_weights), cache["base_ce"][0]


def test_windows(backbone, seqs):
    real_all, ce_all = one_window_check(backbone, seqs, "all", True)
    assert real_all == LENGTH, real_all
    real_asst, _ = one_window_check(backbone, seqs, "assistant", True)
    assert 0 < real_asst < LENGTH, real_asst
    padded, _ = one_window_check(backbone, seqs, "all", False)
    rng = random.Random(1)
    mean_unpacked = sum(sum(train2.sample_window(seqs, LENGTH, rng, "all", False)[1])
                        for _ in range(200)) / 200
    mean_packed = sum(sum(train2.sample_window(seqs, LENGTH, rng, "all", True)[1])
                      for _ in range(200)) / 200
    assert mean_packed == LENGTH and mean_unpacked < LENGTH
    print(f"packed window: {real_all}/{LENGTH} tokens weighted at mask=all, "
          f"{real_asst}/{LENGTH} at mask=assistant, base_ce {ce_all:.3f}")
    print(f"200 windows: mean weighted tokens {mean_unpacked:.1f} unpacked vs "
          f"{mean_packed:.1f} packed ({LENGTH / mean_unpacked:.2f}x)")


def test_err_channel(backbone, seqs):
    cache = train2.alloc_cache(1, LENGTH, D_MODEL, 1, torch.float32)
    train2.fill_cache(cache, backbone, seqs, range(1), LENGTH, random.Random(11), "all", True,
                      "cpu", [LAYERS])
    emb = backbone.emb.weight
    err = train2.make_err(emb, cache["targets"][0], cache["argmax"][0])
    assert err.shape == (1, LENGTH, D_MODEL)
    assert err[0, 0].abs().max().item() == 0.0
    for j in (1, 5, LENGTH - 1):
        want = emb[cache["targets"][0][j - 1].long()] - emb[cache["argmax"][0][j - 1].long()]
        assert torch.allclose(err[0, j], want, atol=1e-6), j
    print("make_err is the previous-step embedding error at every position, zero at 0")


def test_training_step(backbone, seqs):
    train2.LM_HEAD = backbone.lm_head
    cache = train2.alloc_cache(3, LENGTH, D_MODEL, 1, torch.float32)
    train2.fill_cache(cache, backbone, seqs, range(3), LENGTH, random.Random(3), "all", True,
                      "cpu", [LAYERS])
    for use_err in (0, 1):
        sidecar = Sidecar(d_model=D_MODEL, d=32, layers=2, rank=16, n_read=1, use_err=bool(use_err))
        base = train2.mean(train2.evaluate_windows(None, cache, "cpu", backbone.emb.weight, use_err, 1, 4096))
        start = train2.mean(train2.evaluate_windows(sidecar, cache, "cpu", backbone.emb.weight, use_err, 1, 4096))
        assert abs(start - base) < 1e-5, (use_err, start, base)
        per_window = train2.evaluate_windows(sidecar, cache, "cpu", backbone.emb.weight, use_err, 1, 4096)
        assert len(per_window) == cache["windows"] == len(cache["base_ce"]), len(per_window)
        assert abs(train2.mean(per_window) - start) < 1e-6, (per_window, start)
        opt = torch.optim.Adam(sidecar.parameters(), lr=3e-3)
        for _ in range(30):
            total = 0.0
            for i in range(cache["windows"]):
                hs, tgt, w, err = train2.window_tensors(cache, i, "cpu", backbone.emb.weight, use_err, 1)
                loss = sidecar_loss(sidecar, hs, backbone.lm_head, tgt, w, err, 4096)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
        end = train2.mean(train2.evaluate_windows(sidecar, cache, "cpu", backbone.emb.weight, use_err, 1, 4096))
        assert end < start, (use_err, end, start)
        print(f"use_err {use_err}: step-0 ce {start:.4f} equals base {base:.4f}, "
              f"30 steps -> {end:.4f}")


def test_evaluate_windows_is_inference_only(backbone, seqs, monkeypatch):
    train2.LM_HEAD = backbone.lm_head
    cache = train2.alloc_cache(2, LENGTH, D_MODEL, 1, torch.float32)
    train2.fill_cache(cache, backbone, seqs, range(2), LENGTH, random.Random(5), "all", True,
                      "cpu", [LAYERS])
    sidecar = Sidecar(d_model=D_MODEL, d=32, layers=2, rank=16, n_read=1)
    real = train2.sidecar_loss
    grad_enabled = []

    def spy(*a, **k):
        grad_enabled.append(torch.is_grad_enabled())
        return real(*a, **k)

    monkeypatch.setattr(train2, "sidecar_loss", spy)
    train2.evaluate_windows(sidecar, cache, "cpu", backbone.emb.weight, 0, 1, 4096)
    assert grad_enabled == [False] * cache["windows"], grad_enabled
    assert all(p.grad is None for p in sidecar.parameters())
    print(f"evaluate_windows: {len(grad_enabled)} windows scored with grad disabled, "
          f"no backward graph retained")


class RotaryLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, hidden_states, position_embeddings, position_ids=None, cache_position=None):
        cos, sin = position_embeddings
        return hidden_states + self.proj(hidden_states) * cos + sin


class RotaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, D_MODEL)
        self.layers = nn.ModuleList([RotaryLayer() for _ in range(LAYERS)])
        self.embedding_norm = nn.LayerNorm(D_MODEL)

    def pos_emb(self, hidden, position_ids):
        angle = position_ids.float().unsqueeze(-1) * torch.ones(1, 1, D_MODEL)
        return angle.cos(), angle.sin()


def test_manual_hidden_states_walks_rotary_layers():
    torch.manual_seed(0)
    model = RotaryModel()
    x = torch.randint(0, VOCAB, (1, 9))
    states = accel.manual_hidden_states(model, x)
    assert len(states) == LAYERS + 1, len(states)
    hidden = model.embed_tokens(x)
    positions = torch.arange(9).unsqueeze(0)
    cos, sin = model.pos_emb(hidden, positions)
    for layer in model.layers:
        hidden = layer(hidden, (cos, sin))
    want = model.embedding_norm(hidden)
    assert torch.allclose(states[-1], want, atol=1e-6), float((states[-1] - want).abs().max())
    print(f"manual hidden-state walk: {len(states)} states, rotary embeddings passed to every layer")


def test_is_oom_matches_xla_shapes():
    assert accel.is_oom(RuntimeError("RESOURCE_EXHAUSTED: Out of memory allocating 4294967296 bytes"))
    assert accel.is_oom(RuntimeError("Resource exhausted: attempting to allocate 4.00GiB"))
    assert accel.is_oom(torch.OutOfMemoryError("CUDA out of memory"))
    assert not accel.is_oom(ValueError("tokenizer broke"))
    print("is_oom: RESOURCE_EXHAUSTED, Resource exhausted, and CUDA OOM are recognised")


def fingerprint(cache):
    return hashlib.sha1(cache["targets"].cpu().numpy().tobytes()).hexdigest()


def test_hard_token_mask(backbone, seqs):
    cache = train2.alloc_cache(1, LENGTH, D_MODEL, 1, torch.float32)
    train2.fill_cache(cache, backbone, seqs, range(1), LENGTH, random.Random(3), "all", True,
                      "cpu", [LAYERS])
    emb = backbone.emb.weight
    full = train2.window_tensors(cache, 0, "cpu", emb, 0, 1)[2]
    assert float(full.min()) == 1.0 and float(full.max()) == 1.0
    hard = train2.window_tensors(cache, 0, "cpu", emb, 0, 1, 0.25)[2]
    kept = hard > 0
    want = max(1, int(0.25 * LENGTH))
    assert abs(int(kept.sum()) - want) <= 2, (int(kept.sum()), want)
    base_ce = cache["base_token_ce"][0].float()
    assert float(base_ce[kept].min()) >= float(base_ce[~kept].max()) - 1e-3
    assert torch.allclose(base_ce.mean(), torch.tensor(cache["base_ce"][0]), atol=1e-2)
    print(f"hard_frac 0.25 keeps {int(kept.sum())}/{LENGTH} tokens, all of them above "
          f"ce {float(base_ce[kept].min()):.3f} vs {float(base_ce[~kept].max()):.3f} for the rest")


def test_val_fingerprint(backbone, seqs):
    train2.LM_HEAD = backbone.lm_head
    cache = train2.alloc_cache(3, LENGTH, D_MODEL, 1, torch.float32)
    train2.fill_cache(cache, backbone, seqs, range(3), LENGTH, random.Random(3), "all", True,
                      "cpu", [LAYERS])
    first = fingerprint(cache)
    train2.fill_cache(cache, backbone, seqs, range(3), LENGTH, random.Random(7), "all", True,
                      "cpu", [LAYERS])
    assert fingerprint(cache) != first, (fingerprint(cache), first)
    train2.fill_cache(cache, backbone, seqs, range(3), LENGTH, random.Random(3), "all", True,
                      "cpu", [LAYERS])
    assert fingerprint(cache) == first
    print("val fingerprint: one draw hashes stable, a different draw of the windows hashes different")


if __name__ == "__main__":
    torch.manual_seed(0)
    backbone = StubBackbone()
    seqs = fake_seqs(40, random.Random(0))
    test_windows(backbone, seqs)
    test_err_channel(backbone, seqs)
    test_training_step(backbone, seqs)
    test_hard_token_mask(backbone, seqs)
    test_val_fingerprint(backbone, seqs)
