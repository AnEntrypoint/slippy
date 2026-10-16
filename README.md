# slippy

An autoregressive *sidecar* for a frozen [LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M)
backbone. The sidecar is a small causal recurrent network that reads the backbone's hidden
states and emits a correction to the final hidden state. The correction is zero-initialised,
so at step 0 the combined model is bit-identical to the frozen backbone and can only improve
from there — it can never make the base model worse at initialisation.

Two signals are predicted at the same time:

1. **the effect of the context** — the usual next-token objective, conditioned on the
   backbone's own hidden trajectory;
2. **the effect of training** — the previous-step prediction error
   `emb[y_{t-1}] - emb[argmax(logits_{t-1})]` is fed back into the recurrence. That error is
   (minus) the cross-entropy gradient w.r.t. the output embedding, so the recurrent state is
   performing online gradient descent on the context as it reads it: a model that predicts
   what training on this context would do.

## Layout

| path | what |
| --- | --- |
| `sidecar/sidecar.py`, `sidecar/train_kaggle.py` | v1: correction trained with the backbone in the loop |
| `sidecar/sidecar2.py`, `sidecar/train2.py` | v2: cached hidden states, multi-layer readout, error feedback |
| `sidecar/make_kernel2.py` | merges model + trainer into one file and writes `kernel-metadata.json` |
| `sidecar/datascan.py` → `scan.json` | dataset statistics; `dupes.json` records a one-off duplicate-detection pass |
| `sidecar/checkfmt.py` | local check that the encoding matches `apply_chat_template` |

## Why v2 caches hidden states

With the backbone in the training loop, one 8192-token step costs roughly 10.9 TFLOP:
5.7 for the 350M backbone, 0.8 for the sidecar, and 4.4 for the `lm_head` because gradient
checkpointing re-runs it. The backbone's hidden states are *inputs* to the sidecar and are
frozen, so they can be computed once per window and stored. After caching, a step is only
the sidecar plus the `lm_head` — about 4 TFLOP — and chunking the `[T, V]` logits costs no
FLOPs at all, unlike checkpointing. That is the difference between ~0.22 and roughly 1-2
steps/s on a Kaggle T4, and throughput, not convergence, was what limited v1: it reached
only ~2000 of 4000 planned steps inside its 2.5 h budget. v2 measures **1.48 steps/s** at
8192-token windows (`AGENTS.md` has the log line), a 6.7x gain.

The cache stores, per window: the hidden states of each read layer (fp16/bf16), the target
ids, the loss weights, and the backbone's own argmax `P`. `P` is stored rather than an error
vector because it is one int32 per token instead of `d_model` floats, and the error is a
lookup away. A small fraction of the
cache is regenerated every epoch so the model does not overfit a fixed ~2M-token buffer — at
the default 0.05 that proved far too little, and Results says what that cost.

fp16 vs bf16 is chosen by benchmarking a matmul on the actual device rather than by
capability flags: `torch.cuda.is_bf16_supported()` reports `True` on a T4, but Turing has no
bf16 tensor cores, so bf16 there is emulated. The benchmark agrees, and not marginally — on
the Kaggle T4 a 4096x4096 matmul measures **23.0 TFLOPS in fp16 against 2.14 in bf16**, a
10.7x gap. v1 picked bf16 on that flag and paid for it on every step. The sidecar itself runs in fp32 because its
recurrence exponentiates a cumsum whose dynamic range overflows fp16.

## Kaggle workflow

Training runs on Kaggle GPUs only. From `sidecar/`:

```sh
python make_kernel2.py v2base '{"use_err":0,"wall_budget_s":18000}'
cd kaggle_pkg/kernel_v2base && kaggle kernels push
kaggle kernels status heclgang/lfm-sidecar-v2base
kaggle kernels output heclgang/lfm-sidecar-v2base -p .
```

`make_kernel2.py` takes a variant name plus JSON overrides for any argparse default in
`train2.py`, so each A/B is one command. The data is attached as the Kaggle dataset
`heclgang/lfm-sidecar-data`. `HF_TOKEN` is read from Kaggle secrets, not from disk.

## Results

v1 (`loss_mask=all`) reached **-0.521** cross-entropy on validation (2.659 -> 2.137) in ~2000
steps with 16.7M sidecar parameters. Those numbers were produced with a tokenisation bug —
the trainer wrote `<|user|>`, which is not a token in this vocabulary — so v2 re-measures
everything with the encoding verified equal to `apply_chat_template`.

v2 at full size diverges. Two 5 h runs at 8192-token windows (`v2base`, `use_err=0`, and
`v2err`, `use_err=1`) both start bit-identical to the backbone — `eval step 0: base_ce 2.6788
sidecar_ce 2.6788`, so the zero-init correction really is a no-op at initialisation — and then
get monotonically worse: validation cross-entropy climbs 2.6788 -> 15.1943 over 21825 steps
(+12.52), and the error-feedback variant lands at +12.62. The per-window training loss never
trends down; it swings between 0.28 and 20.8, which is the signature of a correction that has
grown to dominate the hidden state instead of nudging it. The CPU test in `sidecar/` puts a
number on that: an uncapped correction reaches 15x the RMS of the hidden state after 50 steps.

The cause is the cache. Caching hidden states buys 6.7x throughput, but it makes training
repeat: 21825 steps over a 256-window, 2.1M-token buffer is 85 passes, and only 5% of the
buffer — 12 windows — is regenerated per pass. v1 recomputed the backbone every step, so every
step was a window the sidecar had never seen. Nothing bounds the size of the correction, so it
memorises the buffer and the damage on unseen windows is unbounded.

The fix now being measured is a per-token trust region, `--corr_max_rel`, which caps the
correction's RMS at a fraction of the hidden state's RMS. It is a `min(1, ...)` clamp, so it is
inactive while the correction is small and step 0 still matches the backbone exactly. Wave 2
measures four configs per backbone on both LFM2-350M and LFM2.5-350M; wave 3 repeats that
factorial with the cap on at 0.1 and 0.3, crossed with two cache-refill rates, so the cap and
the data-reuse half of the diagnosis are measured separately. Fresh numbers, the config lists
and the log lines live in `AGENTS.md`.

## Not in this repo

The training corpus (`trainingdata/`) and all Kaggle run outputs are excluded on purpose.
Nothing here needs them: `train2.py` reads the corpus from `/kaggle/input` at run time.
