# AGENTS.md — trainlfm (slippy)

Facts only; every number is citable from a repo file or a `kaggle_out/` log, and the source is named. Supersedes `README.md`.

## 1. What this is

Autoregressive *sidecar* over a frozen `LiquidAI/LFM2-350M` backbone (`BACKBONE`, `train2.py:20`): a small causal recurrent net reads backbone hidden states and adds a correction to the final hidden state before the LM head. Zero-init (`up.weight` zeros) => at step 0 the combined model is bit-identical to the backbone - hence the `must match base` eval.

| | files | shape |
| --- | --- | --- | --- |
| v1 | `sidecar.py`, `train_kaggle.py` | backbone in the loop; no cache; one read layer |
| v2 | `sidecar2.py`, `train2.py` | hidden states precomputed into a window cache; multi-layer readout (`--read_layers`); optional previous-step error feedback (`--use_err`) |

`make_err`: `emb[y_{t-1}] - emb[argmax(logits_{t-1})]` = minus the CE gradient w.r.t. the output embedding at the previous step - online gradient descent on the context as it is read. `sidecar_loss(..., sidecar=None)` is the frozen-backbone baseline: same weighted CE, correction omitted.

Why v2 caches: backbone in the loop costs ~10.9 TFLOP per 8192-token step (5.7 / 0.8 / 4.4 `lm_head` re-run) vs ~4 cached; v1 reached ~2000 of 4000 steps in 2.5 h, v2 1.48 steps/s (§7).

## 2. Layout (paths under `sidecar/`)

`accel.py` (device abstraction, §6); `conftest.py` (`backbone`/`seqs` fixtures); `datascan.py`/`tokstats.py` -> `scan.json` (corpus stats), `tokstats.json` (per-row token stats); `ceprofile.py` (frozen-backbone per-token CE, cpu fp32, §3); `real_cpu_check.py` (real LFM2-350M hidden states, cpu, §7); `compare.py` (rank by `best_delta`), `report.py` (log -> startup lines, result rows, `step:delta`), `factorial.py` (per-config `delta`, effect vs `base`, paired per-window `+- SE`), `join_waves.py` (merge runs for cross-wave pairing, §12); tests `test_sidecar_math.py` (scan, decay, zero-init, weighted CE, fp16 headroom, cap), `test_pipeline_cpu.py` (packed windows, `make_err`, step-0 ce == base ce, hard mask, fingerprint), `test_sweep_cpu.py` (`build_configs`, `config_args`, `lr_at`, budget split, crash rescue), `test_cache_reuse_cpu.py` (reuse arm, §7); `*.sh` - `push_when_free.sh` (poll to terminal, push), `wave2_push.sh` (both wave-2 sweeps, retry rejects), `pull.sh <variants>` (wait terminal, archive `train.log` -> `train-vN.log`, pull, report); `verify_kernels.py`/`check_kernels.py` (merged `train_all.py` -> backbone, budget split, config list), `checkfmt.py`/`checkfmt2.py` (v2 encoding vs `apply_chat_template`).

Also: `kaggle_pkg/kernel_<variant>/`, `kaggle_pkg/data/` (corpus mirror = dataset `<owner>/lfm-sidecar-data`), `kaggle_out/<variant>/`. Do not edit by hand: `.env` (live HF_TOKEN - never print it), `trainingdata/`, `kaggle_out/`, `.gm/`, `.claude/`, `kaggle_pkg/` (only `make_kernel*.py` rewrites it).

## 3. Data

`trainingdata/train_messages.jsonl` 153,710,261 B, `val_messages.jsonl` 3,126,213 B; one `{"messages":[{"role":...,"content":...},...]}` per line. Mirrored byte-identically into `kaggle_pkg/data/` as `<owner>/lfm-sidecar-data` (private, CC0). The trainer reads `/kaggle/input/lfm-sidecar-data/*.jsonl`; `resolve_input` falls back to walking `/kaggle/input` for the basename.

`tokstats.json` (7017 train rows, seed 1234, 182.5 s): mean 6656.1 tok/row, p50 7739, p90 9724, p95 10284, p99 11471, max 27465, min 47, chars_per_token 3.137, total_tokens_estimate 46,705,663, assistant_token_frac 0.8398, result_token_frac_of_assistant 0.7921, row_tokens_gt_8192_frac 0.4357, row_tokens_gt_16384_frac 0.001.

`scan.json`: 7017 rows / 19678 messages, dup_rows 2506, bad_json 0, prefix_collision_rows 2870, mean_chars 20878.1, max_chars 27992, asst_char_frac 0.8251, result_frac_of_all 0.6559, distinct_tool_calls 204; contaminated_val_prefixes 52, contaminated_val_rows 63.

`dup_rows 2506` is a prefix artifact - `datascan.py` hashes `first_user[:200]`: distinct assistant contents 6989/7017, distinct rows 7017/7017, token-weighted duplicate fraction 0.00118 (`dupes.json`, blake2b, 2026-10-10T16:08Z). Dedup costs 28 rows (0.40%) for 0.12% more distinct tokens - not worth it against ~2.76 M distinct windows (`sum of max(1, n_i - 8192)`) vs a 768-window cache (0.03%). Real redundancy is structural: tool results are 79% of assistant tokens.

The loss is a hard tail over near-perfect tokens (`ceprofile.py`, fp32 cpu, 6 packed windows x 2048 tok from rows 0-59, 2026-10-11): mean 2.5107 (cf. v2sweep train `base_ce` 2.5339), p10 0.001 / p25 0.008 / p50 0.339 / p75 4.223 / p90 8.417 / p99 14.285; hardest 25% carry **0.811** of CE, hardest 50% 0.991, easiest 50% 0.009; 0.369 of tokens below 0.05, 0.217 above 5. Buckets (share, ce share, mean): assistant 0.989 / 0.972 / 2.467, user 0.009 / 0.023 / 6.280; each `<|im_start|>` header bucket <=0.1%. Only `user` and `assistant` roles exist (60 sampled rows) => tool-result text sits inside assistant content, no role mask separates it.

Caveat: `tokstats.py` tokenises with `LFM2.5-350M`'s tokenizer and the v1 rendering, so its numbers approximate the trainer's.

## 4. Tokenisation (v1 != v2)

v2 `encode_rows`: `<|im_start|>{role}\n{content}<|im_end|>\n` per message, prefixed with `bos`. v1: `<|{role}|>\n{content}\n`, no `bos`; `<|user|>` is not a token of this vocabulary, so **every v1 result was measured under a broken encoding** - v2 re-measures with a verified encoding.

## 5. Kaggle workflow

`<owner>` is `$KAGGLE_OWNER` (default `admincoas`, the account the kaggle CLI is logged into); the builders, `pull.sh` and `push_when_free.sh` read it.

From `sidecar/`: `python make_kernel2.py v2base '{"use_err":0,"wall_budget_s":18000}'`, then `cd kaggle_pkg/kernel_v2base && kaggle kernels push`, `kaggle kernels status <owner>/lfm-sidecar-v2base`, `kaggle kernels output <owner>/lfm-sidecar-v2base -p .`

`make_kernel2.py <variant> '<json>' [--out <dir>]` (and `make_kernel.py` for v1) concatenates model + trainer into `<dir>/kernel_<variant>/train_all.py` (`kaggle_pkg` by default) and writes `<owner>/lfm-sidecar-<variant>`, `code_file train_all.py`, private, GPU on, internet on, `dataset_sources` = the corpus. Each JSON key overrides the `default=` of the matching `add_argument("--key" ...)`; `--tpu` also sets `--accelerator tpu`, `enable_tpu true / enable_gpu false` (§6).

`train2.py` defaults: steps 1000000, peak_lr 3e-4, floor_lr 1e-5, warmup 300, d 512, layers 4, rank 256, cache_len 8192, cache_tokens 2097152, cache_ram_frac 0.6, cache_reserve_gb 4.0, refill_frac 0.05, corr_max_rel 0.0, accum 1, hard_frac 1.0, pack 1, val_windows 16, loss_chunk 4096, eval_every 1000, wall_budget_s 18000, save_margin_s 900, accelerator gpu, backbone `LiquidAI/LFM2-350M`; AdamW betas (0.9, 0.95), weight_decay 0.05, clip 1.0. v1 defaults: steps 4000, wall_budget_s 9000, save_margin_s 600.

Outputs in `/kaggle/working`: `train.log` (appended), `result.json`, checkpoints `{step, args, state_dict}` - `sidecar.pt` (best val), `sidecar_final.pt`, `sidecar_crash.pt`; under `--sweep` named per config and `result.json` is a list. One result dict: `name, loss_mask, use_err, read_layers, base_ce, sidecar_ce, delta, best_val_ce, best_delta, steps, params_M, refill_s, cache{n,len,k}, args`, plus `val_window_ce`/`base_window_ce` (per-window pair behind `delta`) and `val_fingerprint` (sha1 of the val targets). `HF_TOKEN` comes from `kaggle_secrets.UserSecretsClient`; wave 1's `result.json` predates `best_delta`, `best_val_ce`, `val_window_ce`/`base_window_ce`, `val_fingerprint`, `refill_s`, so `compare.py` prints `+nan` - a missing key, not a NaN.

Kernel ids: `lfm-sidecar-<variant>` - v1 {train, all2, asst, asst2, stream}, v2 {smoke, v2base, v2err, v2ml, v2data, v25base, v25err, v2sweep, v25sweep, v3cap, v35cap, v4cap, v45cap, tpusmoke}; `v4fresh`/`v45fresh` exist under `kaggle_pkg/` but were never pushed (§12).

## 6. Devices and TPU

`accel.py` is the only place device-specific calls live; `train2.py` uses `pick_device`, `device_dtype` (bf16 tpu / fp16 cuda / fp32 cpu), `attn_implementation` (sdpa off cuda), `device_label`, `mark_step`, `sync`, `empty_cache`, `reset_peak`, `memory_stats` (`xm.get_memory_info` on xla), `is_oom`, `optimizer_step` (`xm.optimizer_step`), `to_device`, `model_hidden_states`. `--accelerator {gpu,tpu,cpu,auto}` defaults gpu, in `SWEEP_FIXED`.

On tpu: `--tpu` on the builder; the matmul benchmark is skipped; `generate` is skipped (one token per XLA recompile). `model_hidden_states` falls back to `manual_hidden_states` - walking `model.layers`, applying `norm`/`embedding_norm`, raising if its argmax disagrees with the model forward on >1e-3 of tokens - when a transformers build ignores `output_hidden_states` (§9).

**Kaggle TPU hours are a separate pool from GPU hours.** `kaggle quota` 2026-10-11 ~10:10Z: GPU 30.40 h used / 0.00 h left / 30.00 h; TPU 0.00 h used / 20.00 h left; both refresh 2026-10-17T00:00:00. `tpusmoke` v1 pushed 09:50:48Z, stayed `QUEUED` for ~10 h; v2 (fixed code, `9f6917a`) pushed 20:01:35Z, `COMPLETE` 22:42Z. Smoke (`kaggle_out/tpusmoke/train.log`, 20 steps, cache 32x1024): `xla TPU devices 8 torch_xla 2.8.0`, compute bf16, `base_ce` 4.0007 train / 4.9714 val (fp16 GPU smoke 4.0213 / 4.9245), step-0 `sidecar_ce` == `base_ce` 4.9714, `delta` -0.0393 @10 / -0.0559 @20, peak 1.35 GB of 16.91 GB, no OOM, 20 steps in 19 s incl. two evals (no per-step recompile visible). Quota gate: `v5tpu` is pushed only after this smoke. The `kaggle` CLI prints `charmap` errors on Windows unless `PYTHONIOENCODING=utf-8`; files still download.

TPU facts (v5e-8 pod, one chip used): 16 GB HBM per chip; torch 2.8.0 + torch_xla 2.8.0 preinstalled; transformers unpinned on the image (the manual-walk fallback matters); bf16 only, no grad scaling; multi-process / SPMD crashes (SIGSEGV/SIGABRT), so single chip; TPU concurrency 1; 9 h session cap.

Wave 5 = `v5tpu`, 3 cells (`cap02` anchor at 0.2, `cap03`, `cap05`; all refill 0.25), 3300 s each, cache 768x8192, `eval_every` 400, `val_windows` 64. `cap02` replicates wave 3's best cell on-device because TPU runs bf16 where GPU ran fp16, so a cross-device `delta` is dtype-confounded. Pushed after the smoke passed its gates (torch_xla up, bf16 `base_ce`, step-0 base match, 2026-10-13 ~09:14Z via `.tmp_w3/tpu_wave5.sh`). **Outcome: failed, no result.** Queued ~11.5 h, then ran ~9 h (TPU used 9.06 h) and was `CANCEL_ACKNOWLEDGED` against a planned ~9900 s; output had no `train.log` or `result.json`, only a 0-byte `cap02.pt` (`kaggle_out/v5tpu`). Cause unknown: `train_one` checks the deadline only between steps, so a stall inside a step, eval or refill is unbounded; the 8192-token shape was never timed on TPU (the smoke used 1024). Diagnostic `v5diag` (300 steps, 32x8192, refill 0, eval every 100, budget 2100 s) pushed 2026-10-14 to measure steady steps/s and eval time at the real shape before any sweep is re-pushed. `fill_windows_with_oom_fallback` halves the initial cache fill on OOM (the one unguarded fatal path); `evaluate_windows` runs under `no_grad` with one XLA graph per window (3b03e95).

## 7. Measured results (only what repo files state)

v1 `kaggle_out/all2/result.json` (`loss_mask=all`, 4000 steps, d 512 / layers 4 / rank 256, params_M 16.68608, `last_eval_step` 2000): `base_ce` 2.6586900129914284 -> `sidecar_ce` 2.137289546430111, `delta` **-0.5214004665613174**; `asst2` (`loss_mask=assistant`, same shape, `last_eval_step` 1500): 2.151584140956402 -> 1.704471431672573, **-0.44711270928382874**. Both under the v1 encoding bug (§4).

v2 smoke (`kaggle_out/smoke/train.log`, 20 steps, cache 32x1024, 900 s): `encoded train 7017 rows 46707712 tokens val 143 rows in 117s`, `base_ce` 4.0213 train / 4.9245 val, `sidecar params 17.73M use_err 0 read [16]`, then `FATAL` (§9); Full-size v2 v1 (`v2base/train-v1.log`, killed at 16 min by the refill OOM): `cache train 256x8192 val 16x8192 hidden_bytes 4.29GB`, `train cache done 573s base_ce 2.6934`, `steps/s 1.48`.

Wave 1 = v2base/v2err v2 (OOM fix), pushed 2026-10-10 ~21:58Z, COMPLETE: `base_ce` 2.6788 -> `delta` **+12.5155** (21825 steps, 256 windows, refill 0.05, val 16); v2err **+12.6189**. Val CE ~linear 3.3394 @500 -> 15.1545 @21500. Cause: 85 passes over ~1276 distinct windows ~= 10.5 M of 46.7 M tokens with nothing bounding the correction - 50 identical Adam steps give worst per-token correction RMS **15.0199x** the hidden RMS. `test_cache_reuse_cpu.py`: uncapped val `delta` **+4.6347** (correction peaks 1.81x hidden RMS); labels linear in the hidden state: +2.4746 uncapped, +0.0656 @0.3, **-0.1252** @0.1.

Wave 2 = `v2sweep`/`v25sweep`, pushed 01:09:58Z/01:10:52Z, ERROR 02:42Z after only their first config (§9). `v2sweep/train.log`: `host ram 33.7GB frac 0.6 reserve 4.0GB -> 768 of 768 windows (12.94GB)`; `train cache done 1689s base_ce 2.5339`; `val cache done base_ce 2.7297 fingerprint e5d63fc9b5a0`. Cell `base` (uncapped, refill 0.05, 768 windows, 4330 steps): `delta` **+2.0250**, `best_delta` **+0.6589** at the step-400 eval; `v25sweep` **+2.8088** (4235 steps). At 5.6 passes it still runs away, train loss 1.45...>4 vs train `base_ce` 2.53 - instability the cap bounds.

Wave 3 = `v3cap` (LFM2-350M) / `v35cap` (LFM2.5-350M), COMPLETE (`result.json`, 4 cells, 4400 / 4304-4308 steps, 768x8192, val 64, `val_fingerprint` e5d63fc9) - `corr_max_rel` {0.1, 0.2} x `refill_frac` {0.05, 0.25}. Val `delta` (`base_ce` 2.7297 LFM2 / 4.1389 LFM2.5):

| cell | cap | refill | LFM2-350M | LFM2.5-350M |
| --- | --- | --- | --- | --- |
| `base` (wave 2) | 0 | 0.05 | +2.0250 | +2.8088 |
| `cap10`/`cap01` | 0.1 | 0.05 | **-0.3534** | **-1.0262** |
| `cap20`/`cap02` | 0.2 | 0.05 | **-0.4452** | **-1.3317** |
| `cap10data`/`cap01data` | 0.1 | 0.25 | **-0.3820** | **-1.0374** |
| `cap20data`/`cap02data` | 0.2 | 0.25 | **-0.4706** | **-1.3423** |

The cap is the result: every capped cell beats the backbone, no curve diverges, val `delta` still descending at 4400 steps (so depth pays). cap 0.1->0.2 gains 0.09 / 0.31 vs refill 0.05->0.25's 0.027 / 0.011 - **cap dominates freshness** ~10:1. `factorial.py` (`join_lfm2`/`join_lfm25`): paired per-window `cap - base` +2.3784 +- 0.1071 (cap10) ... +2.4956 +- 0.1119 (cap20data), t ~= 22, n 64; the backbone effect does not replicate in magnitude (+0.4605 mean, worst deviates 1.2443; LFM2.5 base CE 4.1389 under the fallback tokenizer, §8) but the ranking does (`best_delta` +0.7120 mean, worst deviates 0.2257).

Settled: **the smoke's 4.02/4.92 `base_ce` was context length, not an encoding fault** (at 8192 tokens the frozen backbone scores 2.6934 train / 2.6788 val, bracketing v1's 2.6587); **caching buys 6.7x**.

Real-data cpu check (`real_cpu_check.py`, real LFM2-350M fp32 hidden states, 512-token windows), val `delta` at **2.5 exposures**: **-0.3707** uncapped accum 1, **-0.2726** cap 0.1, **-0.2658** accum 4, **-0.2107** both (val drawn from the *same* 60 rows as train, so only the ranking carries). Reuse arm (32 windows x 512 tok, cap 0, accum 1): -0.2053 @2 exposures, -0.2402 @5, -0.0987 @8, **+0.0149 @10**, +0.3394 @12, **+1.2914 @18**, **+2.1347 @20** - uncapped, the sidecar turns harmful between 8 and 10 exposures of one buffer.

## 8. Design decisions the code cannot state

- **Correction capped relative to the hidden (2026-10-11).** RMS(correction) <= `max_rel` x RMS(`hs[-1]`) per token (`cap_relative_rms`, `min(1, ...)`, a no-op on zero). `--corr_max_rel` default 0.0 (off), not in `SWEEP_FIXED`. Worst per-token ratio after 50 identical Adam steps: 0.1000 @0.1, 0.3000 @0.3, 15.0199 uncapped. CPU toy best cap 0.20 at 75 and at 18.8 exposures, 0.50 at 4.7.
- **`--accum` N** averages the gradient over N windows per optimizer step (2026-10-11); one 8192-token window is one micro-batch, so accum 1 updates on a single-window gradient. `steps_done`, `lr_at`, warmup and `--steps` count optimizer steps => accum 4 buys 1/4 the updates per wall clock and warmup 300 covers 4x more data.
- **`--hard_frac`** trains only the top-CE tokens (2026-10-11): `fill_cache` stores per-token fp16 `base_token_ce` (+2 B/token => `* 11` in `clamp_cache_windows`), `window_tensors` zeroes weights outside the top `hard_frac`, val keeps 1.0 so `delta` stays comparable. Unused.
- **Decay**: `bounded_decay_log = log(0.82 + 0.18*sigmoid(u))` => decay in [0.82, 1.0] (MIN_DECAY 0.82, DECAY_SPAN 0.18), not a free sigmoid/exp.
- **The recurrence runs in fp32.** decay in [0.82,1.0] => the 256-token cumsum spans ~ -51...0, so the other branch evaluates `exp(+51) ~ 1e22`, past fp16's 65504; chunk 128 (v2) / 256 (v1). Hence `.float()` in `Sidecar.forward`/`sidecar_loss`.
- **Partial refill is not charged to the config's clock.** Fires once per pass when `ctx["order_pos"]` wraps (`train2.py:437-446`), replacing `max(1, int(n_train*refill_frac))` windows (12 of 256 at 0.05); fill ~2.24 s/window vs 0.68 s/step, so a refill-heavy cell measures fewer steps, not more data. `train_one` credits `fill_overhead` to `deadline` and to LR `elapsed` (`train2.py:466`) => wall <= ~2 x budget.
- **Windows are packed, not padded** (`--pack 1`, default): pre-2026-10-11 `sample_window` right-padded the 3193/7017 short rows with the repeated last token - **23.2% of every window's tokens were padding**.
- **The cache is host RAM, so more windows are free per step.** `alloc_cache` uses `torch.empty`, `fill_cache` writes `.cpu()` => host RAM limits, not the 14.56 GiB T4. 0.05 on 256 windows => ~500 windows ~= 4 M tokens vs 17.73 M params. `clamp_cache_windows` = 0.6xRAM - 4.0 GB - val windows, floor 8.
- **`result.json` is rewritten after every config**; `FATAL` rewrites it from `CRASH_STATE` (`results`, `args`, `sidecar`, `step`, `name`, `prefix`).
- **`--steps` 1000000 is intentional**: wall clock governs (`deadline = start + (wall_budget_s - save_margin_s)/n_configs`); `lr_at` anneals on `max((step+1)/steps, elapsed/budget)`.
- **fp16 vs bf16 is benchmarked** (`pick_compute_dtype` times 4096x4096 matmuls): **fp16 23.00 vs bf16 2.14 TFLOPS** (README), 21.01/2.21 in the v2base logs. `is_bf16_supported()` is True on a T4 with no bf16 tensor cores (v1 picked bf16 on that flag).
- **LFM2.5-350M is an architecture-identical drop-in** (hidden 1024 / 16 layers / vocab 65536 / `tie_embedding`; full attention at 0-based 2, 5, 8, 10, 12, 14). Its tokenizer names `TokenizersBackend`, absent from transformers 4.57.3 => `train2.py` falls back to the LFM2 tokenizer.
- **Sweep mode packs several questions into one slot.** `--sweep` is a JSON list of `{"name":..., <overrides>}`; `build_configs` validates, rejecting unknown keys and `SWEEP_FIXED` ones (data paths, cache shape, `loss_mask`, `read_layers`, budget, seed, backbone, `pack`, `accelerator`). `train_one` shares backbone, cache, val and window order, splits the budget evenly; one `random.Random(args.seed)` per config drives order and refills (`ctx["rng"]/["order"]/["order_pos"]`); shared val => paired comparison.
- **Loss masking.** `assistant` weights the body and the `<|im_end|>\n` tail, **not** the `<|im_start|>assistant\n` header.

## 9. Known issues

- **int32 targets crash (fixed in v2, keep the fix).** No int32 CUDA kernel for `F.cross_entropy` - the smoke died with `NotImplementedError: nll_loss_forward_no_reduce_cuda_kernel_index not implemented for 'Int'`. Fix: `tgt = labels.reshape(-1).long()`.
- **Cache refill OOM (fixed, keep the fix).** Both version-1 v2 variants died at step ~255 with `torch.OutOfMemoryError: ... Tried to allocate 4.00 GiB` in `fill_cache` on the first refill - the `[1,H,8192,8192]` SDPA matrix of one 8192-token forward. Load-bearing: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` at module scope, `empty_cache` after the initial fill and around each refill, the refill wrapped in `except torch.OutOfMemoryError`, `fill_cache` writing `hidden` last so an abort cannot mix new `hidden` with stale fields (fill peak 12.57 GB on a 14.56 GiB T4).
- **Startup depends on the Kaggle image's transformers build**: 4.57.3's LFM2 has no `output_hidden_states`/`all_hidden_states`, so the hidden-state probe used to raise before training; `model_hidden_states` now falls back to a manual layer walk (§6) - re-check after any image rebuild.
- **v1 encoding bug** (§4) taints the only two v1 deltas; 63 val rows share a prefix with a train row; `tokstats.py` used the wrong tokenizer.
- **A sweep config overriding an existing key crashed the run (fixed 2026-10-11).** `Namespace(**vars(base_args), **overrides)` => `TypeError: multiple values for keyword argument 'use_err'` after config 1; wave 2 lost `err`, `small`, `data`. `config_args()` merges dicts; pinned by `test_sweep_cpu.py`.
- **Kaggle allows only 2 concurrent batch GPU sessions**; a third push is rejected `Maximum batch GPU session count of 2 reached.` and does not queue => wait for a *terminal* status, not a file. No `kaggle kernels cancel`: wave-3 v1 (pre-fix, pushed 02:45:53Z) was left to crash at config 1 (ERROR 03:20-03:22Z) with v2 armed behind it. Wave 2 ran four configs per kernel (`base`, `err`, `data`, `small`), 12600 / 600 = 3000 s per config ~= 4400 steps.
- **Weekly GPU quota 30.00 h** (`Maximum weekly GPU quota of 30.00 hours reached.`), reached 08:47Z 2026-10-11 after ~23.7 h of waves 1-3 plus rebuilds; every GPU push fails until it resets and nothing reports the reset time. TPU hours are separate (§6).

## 10. Working rules

- **Never edit a shell script that is currently running.** One was rewritten by a comment sweep mid-run and exited 0 without pushing; keep long-running scripts in a scratch dir (`C:/dev/trainlfm/.tmp_w3/`).
- **Take timestamps from `date -u`.** git-bash `ls -l --time-style=+%H:%M:%SZ` prints local time (UTC+2) with a literal `Z`; it mis-dated wave 3's rebuild by 2 h.
- **Detach background work with the harness's own background tasks**, not `( nohup cmd >log 2>&1 & )`: on 2026-10-11 that idiom died with its parent (gm's `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`) - it killed `wave4_run.sh` and `cpu5.sh`, 0-byte logs, no surviving python. Fix in flight in `c:\dev\gm`; until it lands use `run_in_background` and verify the log grows.
- **Standing constraints:** training runs on **Kaggle only** (the local box authors, packages, verifies, pushes, analyses; cpu-only tests are fine); **rebuilds are pre-approved** (rewrite a trainer, swap the backbone, re-cut the cache, then realign AGENTS.md and `kaggle_pkg/kernel_*`); public mirror https://github.com/AnEntrypoint/slippy, lanmower-only attribution, never staging `trainingdata/`, `kaggle_out/`, `.env`; local python is `python3` (no `-I`) (torch 2.6.0+cu124, transformers 4.57.3 in *user* site-packages); in gm `bash` prefix `cd <repo>/sidecar &&`, since gm's `cwd` selects the daemon.
- Comments: a function may keep **one** line stating a shape/unit contract the code cannot express; rationale, measurements and history go here. Five remain: `sidecar2.py` 82, 95; `sidecar.py` 82; `sample_window` docstrings at `train2.py:119`, `train_kaggle.py:64`. Prefer self-explanatory names (`decay_log_per_token`, `write_gated_value`).
- Public names the kernel builders depend on: `Sidecar`, `count_params`, `sidecar_loss`, `RMSNorm`, and the source-file replacements (`import math` in `sidecar.py`, `from sidecar2 import ...`, `if __name__ == "__main__":`).
- After any edit: `ast.parse` the edited files, rebuild one variant, parse the merged file, run `verify_kernels.py`.

## 11. Audits (2026-10-10 sidecar2.py + train2.py; 2026-10-11 sweep mode)

Verified correct - line by line, not defects, do not fix:
- **Target/weight alignment**: `weights[1:]`, `y = ids[1:]` => `w[j]` describes target `y[j]`; `sidecar_loss` uses the same `w`.
- **`make_err` is the previous step's error** (§1): `err[0] = 0`, same shift as `generate`'s `shifted = cat([p[:1], p[:-1]])`.
- **The chunk-parallel scan is the sequential recurrence**: `h_t = A_t * (h_{s-1} + cumsum(exp(-c_u) * b_u))` == `decay[t]*h_{t-1} + write_gated_value[t]`; largest intermediate at `chunk=128` is `exp(128*0.1985) ~ 1.1e11` (fp32 overflows above `chunk ~ 445`, fp16 above 4); v1's `gated_recurrence` byte-identical.
- **`hs[-1]` is the post-norm final hidden** (`embedding_norm` precedes `last_hidden_state`; `parse_read_layers` appends `n_layers`).

Open findings:
- **`--use_err` regime gap**: training feeds the ground-truth previous token, `generate` its own output, so past the prompt `err[j]` is `emb[argmax_corrected_{j-1}] - emb[argmax_base_{j-1}]` - inherent to teacher forcing.
- **Val noise**: `val_windows` (16; wave 2+: 64) x `cache_len` drawn with replacement from 143 rows, so `delta` and checkpoint selection carry sampling noise; `factorial.py`'s paired per-window section removes it, cross-backbone pairing gated on `val_fingerprint`.
- **Wave-2 configs share one train cache**: each config's RNG refills the same 768-window buffer, so config 4 trains on a buffer partly regenerated by configs 1-3 - `delta` gaps of a few 1e-2 are confounded with cache age.

Fixed during those audits, do not reintroduce: trailing `save(...)` discarding the best-val checkpoint; a late OOM losing earlier configs; refill time charged to the budget; 23% padding; and three `test_sweep_cpu.py`-pinned fixes - `steps` as the loop index, FATAL rescue reading cleared `results`, sweepable `val_windows`.

## 12. Wave plan

Two GPU slots, one wave at a time (§9). `.tmp_w3/wave4_run.sh` runs the loop: push (retrying rejects every 300 s) -> poll for terminal -> `kaggle kernels output` -> `join_waves.py --out join_lfm2 v2sweep:base=base v3cap:cap01=cap10,cap02=cap20,cap01data=cap10data,cap02data=cap20data v4cap:cap03=cap30,cap05=cap50` (and the `v25sweep`/`v35cap`/`v45cap` triple into `join_lfm25`) -> `factorial.py join_lfm2 join_lfm25` -> `compare.py`.

- **Waves 1-3 COMPLETE** (§7): uncapped +12.5155 / +12.6189, then +2.0250 / +2.8088; every wave-3 cap cell beats the backbone, best **-0.4706** LFM2 / **-1.3423** LFM2.5 at cap 0.2 + refill 0.25; every curve still descending when cut at 4400 steps.
- **Wave 4 (`v4cap`/`v45cap`, cap 0.3 / 0.5) is superseded (2026-10-16)**: the cap sweep tunes one constant of the offline fit, not the stage-1 proof. Its kernels were deleted; the builders regenerate them.
- **Proof run = `v6proof` (`admincoas/lfm-sidecar-v6proof`), pushed 2026-10-16 ~13:27Z**: one cell `proof` at cap 0.2, refill 0.25, 256x8192 cache, val 16, budget 3300 s. It exercises the block-id export (`val_block_id` beside `val_window_ce` and `base_window_ce`), which stage 1 needs as its (block, effect) pairs.
- **Next:** read `val_block_id`/`val_window_ce`/`base_window_ce` from its `result.json`; fit the stage-1 sidecar on those pairs; compare against the mean-effect and block-length baselines (§12 stage-1 gates).
