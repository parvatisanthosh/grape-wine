# ORBIT-LLM — Findings Log

Running record of experimental results. Each section states what was measured, how,
what was found, and what is still open. Raw data lives in `results/`.

**Hardware:** Intel Core i5-1335U (2 P-cores + 8 E-cores, 12 threads), Intel Iris Xe iGPU,
15.7 GB RAM, Windows 11. **Software:** OpenVINO 2026.3, OpenVINO GenAI 2026.3 (built from
source at `bd8d6542`). **Model:** TinyLlama-1.1B-Chat-v1.0, exported with `optimum-cli` to
FP16, INT8 and INT4 IR.

| Precision | Size on disk | Folder |
|---|---|---|
| FP16 | 2098.2 MB | `tinyllama-ov` |
| INT8 | 1050.7 MB | `tinyllama-int8` |
| INT4 | 623.6 MB | `tinyllama-int4` (≈12% of layers kept at INT8) |

---

## Phase 2 — Benchmarking harness

### Method

Every configuration runs in a **fresh process** (`run_matrix.py` → `benchmark_matrix_worker.py`),
so one model's memory and caches cannot leak into the next measurement.

Per configuration: load the pipeline, **1 unmeasured warm-up**, then **3 measured runs**.
Generation is made deterministic and length-exact:

- `do_sample = False` (greedy), `ignore_eos = True`, `max_new_tokens = N` → exactly N output tokens.
- Prompts are built by tokenizing repeated text with `max_length` + `truncation`, so the
  model sees exactly the requested number of input tokens. Both counts are verified against
  OpenVINO's own `perf_metrics` and the run is marked failed if they differ.
- `apply_chat_template = False`, so the template cannot add hidden tokens.

Metrics come from OpenVINO `perf_metrics` (TTFT, TPOT, throughput, generate duration).
Peak memory is the process working set, sampled every 10 ms by a background thread during
generation. Note that OpenVINO's reported throughput is decode throughput (≈ 1000 / TPOT);
it excludes TTFT.

### Baseline (27 input tokens → 50 output tokens, `benchmark_one.py`)

| Mean of 3 runs | FP16 | INT8 | INT4 |
|---|---|---|---|
| TTFT | 527 ms | 469 ms | 364 ms |
| TPOT | 115.4 ms | 80.5 ms | 46.3 ms |
| Throughput | 8.83 tok/s | 12.53 tok/s | 21.66 tok/s |
| Peak RSS | 3972 MiB | 2126 MiB | 1338 MiB |
| vs FP16 | 1.00× | 1.42× throughput, −46.5% RSS | 2.45× throughput, −66.3% RSS |

FP16 throughput varied 7.5–10.4 tok/s across just 3 runs, which was the first sign of the
noise problem described in Finding 4.

---

## Phase 3 — Quantization workload matrix

3 precisions × input {32, 128, 512} × output {32, 128} = 18 configurations × 3 runs.

### Finding 1 — The first matrix measured the prefix cache, not prefill

In the first matrix (`results/matrix_summary.csv`), the same prompt was used for warm-up
and every measured run. TTFT came out **flat at ~45–105 ms regardless of prompt length**,
for example FP16 at 512 input tokens: 106 ms. A 512-token prefill cannot be as fast as a
32-token one, so the prompt was not being processed again.

`test_prefix_cache.py` confirmed it: a repeated 512-token prompt was fast, and a new prompt
of the same length was slow.

**Mechanism (confirmed in the GenAI source).** On CPU, `LLMPipeline` defaults to the
paged-attention / continuous-batching backend and builds it with
`get_latency_oriented_scheduler_config()` (`src/cpp/src/utils.cpp`). That configuration sets
`enable_prefix_caching = true` and an unlimited `max_num_batched_tokens`. KV-cache blocks
from earlier requests are reused when a new prompt shares the same prefix, so repeated
prompts skip almost all of prefill.

**Consequence.** That matrix is a valid measurement of *cache-hit* TTFT and decode speed,
but not of prefill cost. It is kept as `matrix_*` (cached) data.

**Pitfall for later phases.** Passing a plain `SchedulerConfig{}` to the pipeline does
*not* keep these defaults. It sets `enable_prefix_caching = false` and
`max_num_batched_tokens = 256`, which splits a long prefill into 256-token chunks. Any
KV-cache experiment must start from the latency-oriented values. `orbit_run` does this.

### Finding 2 — Cold-prompt matrix (`results/matrix_cold_summary.csv`)

`benchmark_cold_worker.py` gives warm-up and each measured run a **different** prompt
(different topic, same exact length), so every measured run pays full prefill.

Median results:

| Model | In | Out | TTFT ms | TPOT ms | Tok/s | Peak MiB | Speedup vs FP16 |
|---|---|---|---|---|---|---|---|
| FP16 | 32 | 32 | 801.6 | 118.1 | 8.47 | 4147 | 1.00× |
| FP16 | 128 | 32 | 2708.6 | 124.9 | 8.01 | 4112 | 1.00× |
| FP16 | 512 | 32 | 10506.1 | 115.6 | 8.65 | 4165 | 1.00× |
| INT8 | 32 | 32 | 404.5 | 63.5 | 15.75 | 2124 | 1.86× |
| INT8 | 128 | 32 | 1110.6 | 61.2 | 16.35 | 2138 | 2.04× |
| INT8 | 512 | 32 | 3719.7 | 57.9 | 17.27 | 2191 | 2.00× |
| INT4 | 32 | 32 | 316.6 | 45.9 | 21.81 | 1337 | 2.58× |
| INT4 | 128 | 32 | 1185.0 | 45.1 | 22.17 | 1347 | 2.77× |
| INT4 | 512 | 32 | 5689.1 | 52.9 | 18.89 | 1401 | 2.18× |

The full 18-row table, including output length 128, is in `matrix_cold_summary.csv`.

- **TTFT now scales with prompt length**, roughly linearly. For FP16, 16× more input tokens
  (32 → 512) gives 13× higher TTFT.
- **Memory savings are stable:** INT8 saves 47–49% and INT4 saves 66–68%, at every
  workload size.
- **Decode speedup:** INT8 is 1.8–2.0× faster than FP16, INT4 2.2–3.1× faster.

### Finding 3 — Prefill and decode prefer different precisions

Prefill rate at 512 input tokens is input tokens / TTFT. Decode rate is 1000 / TPOT.

| | FP16 | INT8 | INT4 |
|---|---|---|---|
| Prefill (512 tokens) | 49 tok/s | **138 tok/s** | 90 tok/s |
| Decode | 8.6 tok/s | 17.3 tok/s | **18.9 tok/s** |

**INT4 decodes fastest but INT8 prefills fastest.** At 512 input tokens, INT4's TTFT is
53% higher than INT8's (5.7 s vs 3.7 s).

Likely explanation (hypothesis, not yet verified):
- Decode is **memory-bandwidth-bound**. Each token streams all weights once, so smaller
  weights win, and INT4 is fastest.
- Prefill processes many tokens per weight load, so it is **compute-bound**. INT4 has to
  unpack and dequantize 4-bit groups, which adds compute exactly where compute is the
  bottleneck.

If this holds, it is direct evidence for the plan's phase-aware research question: the best
configuration would depend on whether a request is prefill-heavy (long prompt, short
answer) or decode-heavy (short prompt, long answer).

> **Status: not reproduced.** In the C++ re-run (below), INT4's TTFT at 512 input tokens
> was 3.9 s / 3.1 s (32 / 128 output tokens) against INT8's 3.2 s / 3.6 s. That is no
> consistent penalty. The original 5.7 s was most likely inflated by the same noise as
> Finding 4. Treat this as an open hypothesis. Test it with longer prompts (1k–2k tokens),
> more runs and interleaved ordering before claiming it.

### Finding 4 — Run-to-run and day-to-day noise is large

- The original INT4 512→128 runs (`matrix_cold_original.csv`) reported TPOT of 116–179
  ms/token, against ~49 ms for every other INT4 configuration. That configuration was rerun
  at 13:11 (`int4_512_128_retest.csv`), and the rerun rows replace the originals in
  `matrix_cold_clean.csv`. **This is the only difference between the original and clean
  files.**
- Decode speed for identical work differed between days. FP16 TPOT was 87–94 ms in the
  cached matrix (Oct 4) and 109–125 ms in the cold matrix (Oct 5). Decode work is the same
  in both, so the two matrices can be compared on **TTFT only**, not on decode speed.
- Likely causes: background load, OneDrive sync (the project folder is on OneDrive) and
  thermal state of a 15 W laptop CPU.

**Action taken.** `orbit_run` records available RAM and system CPU-busy % immediately
before every measured run, so noisy runs can be identified instead of guessed at.

### Open questions

1. **RSS is about 2× the on-disk size for every precision:** FP16 4.15 GB vs 2.10 GB,
   INT8 2.13 GB vs 1.05 GB, INT4 1.34 GB vs 0.62 GB.
   - **FP16 — explained.** The CPU plugin on this machine reports
     `INFERENCE_PRECISION_HINT = f32` and capabilities `FP32, INT8`: there is no native
     FP16 or BF16 compute on the i5-1335U. FP16 weights are therefore converted to f32 at
     compile time. 1.1B parameters × 4 bytes ≈ 4.1 GiB, which matches the measured RSS. On
     this CPU, the "FP16" baseline is really *f32 compute*.
   - **INT8 and INT4 — explained in Phase 4 (Finding 9).** Committed private memory
     after warm-up is only 850 MiB for INT4, while the working set is 1350 MiB. The
     ~500 MiB difference is the memory-mapped IR file, which the working set counts and
     the commit charge does not. The weights are resident twice: once as the mapped file,
     once as the compiled model's private copy.
2. **Prefill is slow even for INT8** (138 tok/s). Does it improve with P-cores only, or a
   different thread count? Early `orbit_run` checks suggest P-core-only may beat the
   12-thread default on this hybrid CPU.
3. **INT4's prefill penalty:** does it grow with prompt length (1k, 2k tokens)? That decides
   where the INT8/INT4 crossover is.
4. **Lazy weight loading:** RSS stays at ~125–275 MB after pipeline construction and jumps
   only on the first generate call (see `../LOG.md`). Weights appear to be paged in on
   first use, which is why warm-up matters.

---

## C++ executor parity (`orbit_run`)

From here on, experiments run through `cpp/orbit_run.exe`, which uses the OpenVINO GenAI
**C++ API**. The Python layer only orchestrates and analyses. Before switching, the C++
executor was checked against the Python harness.

### Full matrix re-run (`results/cpp_matrix_cold_*`)

The same 18 cold-prompt configurations were run through `run_cpp_matrix.py`:
54/54 runs succeeded in 10.4 minutes.

- **Functionally identical:** the same exact input and output token counts on every run.
- **Memory:** peak memory is 45–55 MiB lower in every configuration. That is the Python
  interpreter: RSS before loading is 54 MiB under Python and 8 MiB under C++.
- **Timing was mostly lower:** TTFT was 13–51% lower in most configurations, and FP16 TPOT
  9–23% lower. Some INT4 configurations were slower, with TPOT up to 20% higher.
- **Conditions were not the same:** the Python matrix ran in the morning with applications
  open; the C++ matrix ran in the evening after closing them. Background CPU during the
  C++ run: median 28.5%, range 17–57%. Available RAM was only 1.0–1.5 GB during the FP16
  configurations.

### Interleaved A/B (same session, alternating executors)

To separate "C++ vs Python" from "morning vs evening", two configurations were run
alternately: Python, C++, Python, C++, … for 3 rounds × 3 measured runs each (n = 9 per
executor). Raw data is in `results/ab_python_vs_cpp_*`. Medians:

| Configuration | Executor | TTFT | TPOT | Peak RSS |
|---|---|---|---|---|
| FP16, 128 in → 32 out | Python | 1612 ms | 126.4 ms | 4113 MiB |
| | C++ | 1496 ms | 133.7 ms | 4064 MiB |
| INT8, 128 in → 128 out | Python | 955 ms | 62.5 ms | 2138 MiB |
| | C++ | 761 ms | 54.2 ms | 2090 MiB |

- **FP16:** no meaningful difference. TTFT is 7% lower and TPOT 6% higher, and the run
  ranges overlap heavily.
- **INT8:** C++ is 20% lower on TTFT and 13% lower on TPOT, and much less variable. C++
  TTFT range was 707–860 ms; Python's was 591–1144 ms. A possible cause is the Python
  memory-sampling thread competing for the GIL and a core. Not verified.
- **Conclusion:** the C++ executor is a faithful replacement. Its numbers are equal to or
  slightly better than Python's. The large gaps in the full-matrix comparison came mostly
  from different machine conditions, not from the executor.

### Finding 5 — Absolute numbers drift by up to ~40% between sessions

FP16 TPOT for comparable work was 92 ms in the C++ matrix and 126–134 ms in the A/B test an
hour later, on the same evening, with the same binary. FP16 is the configuration closest to
this machine's free-RAM limit, so it is the most sensitive.

**Rules adopted for every later experiment:**

1. Compare configurations only **within the same session**. Interleave or randomise their
   order, rather than running all of A and then all of B.
2. Use **5 measured runs** per configuration instead of 3, and report the median with its
   range.
3. Keep the per-run machine state (`sys_available_ram_mib`, `sys_cpu_busy_percent`) in the
   dataset. For the Performance Twin (Layer 1) these are **input features**, not noise to
   discard. Predicting performance under a given machine state is exactly the twin's job.
4. Pause OneDrive sync before benchmarking. The project folder is synced, and the sync
   client used ~16% CPU while results were being written.

---

## Phase 4 — KV cache and memory (`plans/kv_precision.json`, `prefix_cache.json`, `cache_size.json`)

Method: `run_experiment.py`, all through `orbit_run`. Each plan runs 2 rounds in shuffled
order, with 3 measured runs per round, so n = 6 per configuration. 252 measured runs, 0
failures, 69 minutes in total. Raw rows are in `results/experiments/*.jsonl`; summaries
are in the matching `*_summary.csv`. Background CPU before runs: median 25%, range 9–76%.

**Defaults verified on this machine** (`Core.get_property("CPU", …)`):
`KV_CACHE_PRECISION = u8`, `INFERENCE_PRECISION_HINT = f32`, `INFERENCE_NUM_THREADS = 0`
(auto), `SCHEDULING_CORE_TYPE = ANY_CORE`. OpenVINO already quantizes the KV cache to
8 bits by default on this CPU.

**Expected KV size** for TinyLlama (22 layers, 4 KV heads, head dim 64):
2 × 22 × 4 × 64 = 11,264 values per token. That is 11.0 KiB at u8, 22.0 KiB at f16 and
44.0 KiB at f32. At 1920 tokens: 21 / 41 / 83 MiB.

### Finding 6 — KV precision behaves exactly as theory predicts, but barely matters at 1.1B

Peak RSS growth from 256 to 1920 prompt tokens (+1664 tokens), median of 6 runs:

| KV type | INT4 | INT8 | Extra vs u8 (measured) | Extra vs u8 (theory) |
|---|---|---|---|---|
| u8 (default) | +222.6 MiB | +223.3 MiB | — | — |
| f16 | +237.9 MiB | +236.5 MiB | +14 MiB | +17.9 MiB |
| f32 | +272.8 MiB | +273.0 MiB | +50 MiB | +53.6 MiB |

- The *differences* between KV types match the theoretical KV sizes within a few MiB,
  for both weight precisions.
- **Speed:** TTFT shows no consistent effect. At 1920 tokens, INT4 TPOT was 56.5 ms with
  u8, 50.0 ms with f16 and 46.5 ms with f32. That suggests dequantizing the KV cache costs
  some decode time at long context, but the ranges overlap and INT8 shows no such trend.
  Weak signal.
- **Conclusion for a 1.1B model:** at its full 2k context, the KV cache is 21–83 MiB,
  against ~1.3–2.1 GB of weights. KV precision is not a meaningful lever here. It becomes
  one for larger models and longer contexts. For example, a 3B model with 8 KV heads ×
  head dim 128 needs ≈ 112 KiB per token at f16, so 8k tokens is about 0.9 GiB. **Phase 5
  should add a 3B model.**

### Finding 7 — Memory growth with prompt length is ~12× larger than the KV cache

Total growth is **0.134 MiB per prompt token**, while the u8 KV cache accounts for only
0.011 MiB per token.

Hypothesis: the paged-attention path computes **full-vocabulary logits for every prompt
token** during prefill. 32,000 vocab × 4 bytes = 0.122 MiB per token; adding the
0.011 MiB of KV gives 0.133 MiB per token, against 0.134 measured. The arithmetic fits
closely, but it is not yet confirmed from the source or a profiler.

The SDPA backend grows even more: +462 MiB RSS from 256 to 1920 tokens, against +222 MiB
for PA (n = 1 each). Its TTFT is also 69% higher at 1920 tokens (24.4 s vs 14.4 s).
32 heads × 1920² × 4 bytes = 450 MiB, which suggests SDPA materializes the full
attention-score matrix while PA does not. Also a hypothesis.

> **Status: logits hypothesis refuted (Finding 16).** It predicted ~0.60 MiB/token for
> Qwen2.5-3B, with its 151,936-token vocabulary. The measured growth was 0.23 MiB/token.
> The growth tracks MLP activation size instead. See Finding 16.

**Why it matters:** for memory prediction, prompt length has a much larger effect than the
KV cache does on this model. A memory model for the Twin should be
`weights + a × prompt_tokens + kv_bytes × (prompt + output tokens)`, with `a` measured
per backend.

### Finding 8 — Prefix caching: ~177× faster TTFT on repeated prompts, no cost otherwise

| Model | Prompt | TTFT, repeated, caching **on** | TTFT, repeated, caching off | TTFT, unique, on | TTFT, unique, off |
|---|---|---|---|---|---|
| INT4 | 512 | **69 ms** | 4939 ms | 3407 ms | 3507 ms |
| INT4 | 1920 | **77 ms** | 12949 ms | 13750 ms | 13585 ms |
| INT8 | 512 | **60 ms** | 2643 ms | 4529 ms | 3386 ms |
| INT8 | 1920 | **72 ms** | 12737 ms | 12732 ms | 18955 ms |

- With caching on, TTFT on a repeated prompt is almost independent of length: 60–77 ms
  for both 512 and 1920 tokens. Only the newly generated tokens need computing.
- For unique prompts, on and off are within noise of each other, and peak memory is
  identical. Leaving prefix caching on, which is the `LLMPipeline` default, costs nothing
  measurable for a single user.
- Noisy cells: INT8 unique/off at 1920 tokens ranged 11.4–25.4 s, and INT8 unique/on at
  512 tokens ranged 2.6–6.6 s. Both are session noise, not configuration effects.
- **For ORBIT-LLM:** "does this request share a prefix with a recent one?" is the single
  strongest TTFT predictor found so far. Chat follow-ups and repeated system prompts get
  this effect for free. The Workload Profiler should detect it.

### Finding 9 — Working-set memory hides real memory commitments

`cache_size_gb` pre-allocates the KV cache. In the main experiment it changed peak
**working set** by only +20–56 MiB, even at 4 GB. After adding committed private memory
(`PrivateUsage`) to `orbit_run`, a follow-up check (INT4, 512 tokens, n = 1 each) showed:

| `cache_size_gb` | Working set after warm-up | Committed after warm-up | Peak committed |
|---|---|---|---|
| 0 (dynamic) | 1350 MiB | 850 MiB | 851 MiB |
| 2 | 1352 MiB | 2892 MiB (+2042) | **4323 MiB** |
| 4 | 1355 MiB | 4948 MiB (+4098) | **8431 MiB** |

- The pre-allocated cache *is* committed in full (+2.0 / +4.0 GiB), but its pages are not
  touched, so the working set never shows it.
- **Peak** committed memory reaches ≈ 2× the cache size during initialization. A 4 GB
  cache transiently commits about 7.6 GB extra.
- The same measurement explains the INT8/INT4 "2× disk size" question (Open question 1):
  working set = private memory + the memory-mapped IR file.

**Consequence for the Performance Twin:** OOM risk must be predicted from **committed
memory against the system commit limit**, not from working set. All new `orbit_run`
rows record `commit_after_load_mib`, `commit_after_warmup_mib`,
`commit_after_generation_mib` and `lifetime_peak_commit_mib`. The Phase 4 experiment rows
predate this field.

### Finding 10 — How much of the noise does machine state explain?

Each run's deviation from its configuration's median was correlated with the machine
state recorded just before it (all 252 Phase 4 runs):

| Metric | Typical spread (p10–p90 of run / median) | r with background CPU % | r with free RAM |
|---|---|---|---|
| TTFT | 0.78 – 1.23 | 0.25 | −0.05 |
| TPOT | 0.89 – 1.12 | 0.13 | 0.01 |

Background CPU load explains only a small part of the variance. Free RAM explains none of
it on these sizes, which never came near the limit. The remaining noise is probably
thermal and power-limit behaviour of a 15 W CPU, plus thread placement on hybrid
P/E-cores. Neither is measured yet.

**Implications:**
- (a) Expect a TTFT prediction-error floor of roughly ±20% on this laptop. Evaluate the
  Twin against that floor, not against zero.
- (b) Record CPU frequency or temperature, and CPU load *during* the run, before training
  the Twin.
- (c) The P-core experiment (week 5) may reduce the variance as well as the mean.

---

## Phase 5 — Device and CPU core placement (`plans/device_cores.json`, `gpu_fp16.json`, `load_time.json`)

Method: same as Phase 4 (2 shuffled rounds × 3 runs, n = 6 per configuration), 64 output
tokens, unique prompts. `orbit_run` now also records CPU behaviour *during* each run:
effective MHz, own CPU share and background CPU share. 216 measured runs plus 12
load-time invocations, 0 failures, 49 minutes.

Variants:

| Variant | Settings |
|---|---|
| `cpu_auto` | OpenVINO default (all 12 threads, any core) |
| `cpu_pcore` | `SCHEDULING_CORE_TYPE = PCORE_ONLY` |
| `cpu_ecore` | `SCHEDULING_CORE_TYPE = ECORE_ONLY` |
| `cpu_4t`, `cpu_8t` | `INFERENCE_NUM_THREADS` = 4 or 8 |
| `gpu` | Intel Iris Xe iGPU |

### Finding 11 — The iGPU wins every per-request metric, by a wide margin

Medians, n = 6:

| Model | Prompt | Best CPU TTFT | GPU TTFT | Best CPU TPOT | GPU TPOT | GPU decode tok/s |
|---|---|---|---|---|---|---|
| INT4 | 128 | 969 ms (8t) | **315 ms** | 38.2 ms (P-core) | **26.6 ms** | 37.5 |
| INT4 | 1024 | 9011 ms (E-core) | **1740 ms** | 46.9 ms (P-core) | **28.2 ms** | 35.5 |
| INT8 | 128 | 1013 ms (P-core) | **291 ms** | 58.6 ms (P-core) | **40.5 ms** | 24.7 |
| INT8 | 1024 | 7999 ms (8t) | **1804 ms** | 52.1 ms (P-core) | **37.7 ms** | 26.6 |
| FP16 | 128 | 3420 ms (auto) | **225 ms** | 123.1 ms (auto) | **48.4 ms** | 20.7 |
| FP16 | 1024 | 27285 ms (auto) | **2358 ms** | 171.1 ms (auto) | **49.7 ms** | 20.1 |

- **Prefill:** the GPU is 3–5× faster than the *best* CPU configuration for INT4/INT8, and
  12× faster than the CPU for FP16 at 1024 tokens.
- **Decode:** the GPU is 1.4× faster than the *best* CPU configuration (P-cores).
- **FP16 benefits most.** The iGPU has native FP16; the CPU has to compute in f32
  (Open question 1). FP16 on the GPU decodes at 20 tok/s, against 6–8 tok/s on the CPU,
  and uses 2.5 GB RSS instead of 4.1 GB.
- **The GPU is much more stable.** INT4 at 1024 tokens: GPU TTFT ranged 1470–1772 ms
  (±10%), CPU default 7171–14398 ms (2×). Moving work off the contended CPU removes most of
  the session noise from Findings 5 and 10.
- **Caveat on GPU memory:** process RSS is *lower* on the GPU (INT4 1080 MiB vs 1307 MiB),
  but iGPU buffers live in shared system memory and may not all be attributed to the
  process working set. GPU memory needs a separate measurement before any memory claim is
  made.

### Finding 12 — On the hybrid CPU, the OpenVINO default is never the best configuration

Two P-cores only (`cpu_pcore`) beat the 12-thread default on decode in all four cells:

| Model | Prompt | TPOT, default | TPOT, P-core only | Change |
|---|---|---|---|---|
| INT4 | 128 | 63.4 ms | 38.2 ms | −40% |
| INT4 | 1024 | 60.5 ms | 46.9 ms | −22% |
| INT8 | 128 | 69.4 ms | 58.6 ms | −16% |
| INT8 | 1024 | 84.3 ms | 52.1 ms | −38% |

- `cpu_pcore` used only ~15% of total CPU time (≈ 2 of 12 logical processors) and was
  still the fastest decoder.
- Likely mechanism (hypothesis): decode is memory-bandwidth-bound, and each layer's work
  is split across threads. With E-cores in the pool, every step waits for the slowest
  (E-core) thread to finish its share — a straggler effect. More threads add
  synchronization cost without adding bandwidth.
- **Prefill is different.** More threads help this compute-bound phase: 8 threads gave the
  best INT8 TTFT at 1024 tokens and the best INT4 TTFT at 128 tokens. **The best CPU
  thread placement differs between prefill and decode.** This is concrete evidence for the
  plan's phase-aware research question. Within one request, OpenVINO compiles one thread
  configuration, so exploiting it would need two compiled models or a per-phase switch.
  That is a design question for the controller.

### Finding 13 — The model cache helps the GPU and hurts the CPU

Pipeline construction time, INT4, 3 rounds. Round 1 of a cached variant populates the cache.

| Variant | Round 1 | Round 2 | Round 3 |
|---|---|---|---|
| CPU, no cache | 2.3 s | 2.0 s | 2.4 s |
| CPU, `cache_dir` | 10.8 s | 13.2 s | 10.5 s |
| GPU, no cache | 7.6 s | 8.0 s | 9.2 s |
| GPU, `cache_dir` | 20.8 s *(cold)* | **2.1 s** | **2.8 s** |

- **GPU:** without a cache, every load recompiles kernels, which took 6–11 s in
  `device_cores` and 14–19 s for FP16. With a warm cache, load drops to 2–3 s, about the
  same as the CPU. Populating the cache costs one slow load of ~21 s.
- **CPU:** the cache makes every load ~5× *slower*, even when warm (10–13 s vs 2–2.4 s).
  The likely reason is that the cached blob is read in full instead of memory-mapping the
  IR. Not verified.
- **Rule:** enable `cache_dir` for GPU, never for CPU.

### Finding 14 — What this means for ORBIT-LLM's decision problem

On this laptop, for steady-state generation, **GPU + INT4 dominates**: fastest TTFT,
fastest decode, lowest variance. A controller that always picked it would be
near-optimal per request. The real decisions are therefore around it:

1. **Cold start vs request size.** Example: one request with a 128-token prompt and
   64 output tokens. Single-run numbers, so approximate.

   | Path | Load | + TTFT | + decode | ≈ Total |
   |---|---|---|---|---|
   | CPU (P-cores) | ~1.4–3.7 s | 1.0 s | 64 × 38 ms = 2.4 s | 5–7 s |
   | GPU, no cache | 6–11 s | 0.3 s | 1.7 s | 8–13 s |
   | GPU, warm cache | 2–3 s | 0.3 s | 1.7 s | 4–5 s |

   Whether the GPU is worth it depends on cache state and how much work follows. That is
   exactly a predict-then-choose problem.
2. **Phase-aware CPU threads** (Finding 12), when the GPU is unavailable or busy.
3. **Prefix reuse** (Finding 8), the largest single TTFT effect: ~177×.
4. **Memory and OOM limits** (Finding 9), which will start to bind with a 3B model.

**Next:** a 3B model is now the most important addition. It makes KV-cache and memory
decisions real, and tests whether "GPU always wins" survives a model 3× larger in a
GPU that shares system memory.

---

## Layer 1 — Performance Twin v0 (`build_dataset.py`, `twin_v0.py`)

**Dataset.** `build_dataset.py` merges every successful C++-executor run into
`results/dataset.csv`: 504 runs over 85 distinct configuration+workload combinations.

Features use only what a controller knows *before* running a request:

| Feature group | Columns |
|---|---|
| Configuration | precision, device, backend, cores, threads, KV precision, prefix caching, cache size |
| Workload | prompt tokens, output tokens, prefix-cache hit |
| Machine state | background CPU %, available RAM — measured just before the run |

Targets: TTFT, TPOT and peak RSS.

**Predictors.**

- **analytical** — four physical laws from the findings, fitted per hardware placement by
  least squares:
  - TTFT ∝ prompt tokens
  - cache-hit TTFT ≈ constant
  - TPOT ≈ linear in context
  - memory ≈ linear in tokens

  Groups fall back to coarser ones when a configuration is unseen.
- **gbm** — gradient-boosted trees (`HistGradientBoostingRegressor`) on log targets.

**Evaluation.** 5-fold `GroupKFold` holding out *whole configurations*, so every
prediction is for a configuration+workload the model never saw. The **noise floor**
predicts each run from the median of the *other* runs of the same configuration — the
best any predictor could do without knowing the run's own noise.

### Finding 15 — Simple physical laws predict as well as ML, close to the noise floor

| Target | Predictor | Mean APE | Median APE | Within ±20% |
|---|---|---|---|---|
| TTFT | noise floor | 19.3% | 11.7% | 65.9% |
| | **analytical** | 23.5% | 15.7% | 57.7% |
| | gbm | 29.0% | 15.5% | 59.9% |
| | gbm, no machine state | 33.1% | 21.3% | 47.0% |
| TPOT | noise floor | 13.7% | 8.4% | 76.2% |
| | analytical | 15.4% | 11.2% | 71.0% |
| | **gbm** | 15.0% | 10.3% | 75.8% |
| Peak RSS | noise floor | 0.4% | 0.0% | 100% |
| | **analytical** | 1.1% | 0.3% | 100% |
| | gbm | 3.3% | 0.5% | 94.0% |

- **Both predictors are within ~4 percentage points of the noise floor** on TTFT and TPOT
  median error, and memory is predicted almost exactly. Most of the remaining error is
  run-to-run noise that no pre-run predictor can see.
- **The analytical model matches the ML model.** On this data, what matters is getting
  the physics right: proportional prefill, constant cache-hit TTFT. Model capacity adds
  little. That is a useful, honest result: the Twin is explainable.
- **Machine state helps the ML model's TTFT predictions:** within ±20% rises from 47% to
  60% of runs when background CPU and free RAM are included.
- **A bug found by evaluation.** The first analytical version predicted a flat TTFT when a
  held-out configuration's group had been seen at only one prompt length, giving 10×
  errors (mean APE 115%). Assuming TTFT ∝ prompt length in that case fixed it.
- **Weak spots:** GPU FP16 (median APE 40–47%, little GPU data) and very short prompts
  (`load_time`: 32 tokens, 61–68%). More GPU and short-prompt data are needed.

---

## Phase 6 — 3B model (Qwen2.5-Coder-3B-Instruct, INT4)

Pre-converted IR from `huggingface.co/OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov`
(1.8 GB, Apache-2.0); there was not enough RAM or disk to export it locally. Architecture:
36 layers, 16 attention heads, 2 KV heads, head dim 128, hidden 2048, MLP 11008, vocab
151,936, 32k context.

**Tooling fix:** Qwen's exported tokenizer ignores `max_length`/`truncation`. `orbit_run`
now encodes the full text and slices the token tensor to the exact length, which gives
identical results for TinyLlama.

### Finding 16 — First 3B results, and a refuted hypothesis

Single runs, INT4, 32 output tokens, about 1.3–2.4 GB of RAM free:

| | CPU, 256 tokens | CPU, 1920 tokens | GPU, 256 tokens | GPU, 1920 tokens |
|---|---|---|---|---|
| TTFT | 6.0 s | **50.4 s** | 0.97 s | 9.6 s |
| TPOT | 127 ms (7.9 tok/s) | 159 ms | 58 ms (17 tok/s) | 106 ms |
| Peak RSS | 3841 MiB | 4222 MiB | 2545 MiB | 2676 MiB |
| Committed after warm-up | 2235 MiB | 2616 MiB | 2461 MiB | 2595 MiB |

- **On CPU, a 3B model is borderline-unusable for long prompts.** A 1920-token prompt
  takes 50 s to first token. The GPU is 5–6× faster on TTFT and 1.5–2.2× faster on decode.
- **Memory is tight.** CPU peak RSS reached 4.2 GB while the system had only 1.3 GB free.
  This is the first model where OOM risk is a live constraint on this laptop.
- **Finding 7's logits hypothesis is refuted** by a prediction made before running.
  Full-vocabulary logits per prompt token would cost 151,936 × 4 B = 0.58 MiB per token,
  for ~990 MiB growth from 256 to 1920 tokens. The measurement was **382 MiB**
  (0.23 MiB per token).
- **Better hypothesis: MLP activation buffers.** Remove the KV share (u8: 0.011 MiB per
  token for TinyLlama, 0.018 for Qwen) and compare the rest with
  `(hidden + 2 × MLP) × 4 B`:

  | Model | Non-KV growth | `(hidden + 2 × MLP) × 4 B` | Ratio |
  |---|---|---|---|
  | TinyLlama (MLP 5632) | 0.123 MiB | 0.051 MiB | 2.4× |
  | Qwen2.5-3B (MLP 11008) | 0.21 MiB | 0.092 MiB | 2.3× |

  The constant ratio across two architectures suggests prefill memory is dominated by a
  few live intermediate buffers of size `prompt_tokens × (hidden + 2 × MLP)` in f32. Still
  a hypothesis, but it now predicts both models, and it gives the Twin a memory feature
  derived from architecture.

**Next for Phase 6:** a proper 3B plan (n = 6, shuffled), run with as much free RAM as
possible. Planned dimensions: GPU vs CPU P-cores, KV precision at long context, and
prefix caching. GPU memory needs its own measurement, because process RSS under-counts
iGPU buffers.

### Finding 17 — 3B model: the GPU is what makes it usable (`plans/qwen3b.json`)

n = 6 per configuration, 2 shuffled rounds, 64 output tokens, unique prompts.

| Qwen2.5-Coder-3B INT4 | CPU P-cores, 256 tok | GPU, 256 tok | CPU P-cores, 1024 tok | GPU, 1024 tok |
|---|---|---|---|---|
| TTFT | 5.8 s | **0.96 s** | 29.0 s | **4.3 s** |
| Decode | 8.7 tok/s | **18.9 tok/s** | 7.0 tok/s | **18.4 tok/s** |
| Peak RSS | 3838 MiB | 2524 MiB | 4014 MiB | 2624 MiB |

- **Speed:** the GPU is 6–7× faster on TTFT and 2.2–2.6× faster on decode. At 1.1B
  (Finding 11), the GPU was the better choice; at 3B, CPU-only gives 29 s to first token
  for a one-page prompt.
- **KV precision on CPU (u8 default vs f16):**
  - Memory: f16 costs +18 MiB at 1024+64 tokens, against 19.1 MiB from theory (Qwen:
    18 KiB per token at u8, 36 KiB at f16).
  - Speed: f16 was *faster* at 1024 tokens, with TPOT 106 vs 144 ms and TTFT 25.4 vs 29.0 s.
    That suggests u8 dequantization costs decode time at longer contexts, matching the
    weak signal in Finding 6. But the u8 TPOT range was wide (95–217 ms), so it is not
    conclusive.
- **Explicit KV precision crashes on the GPU.** Both u8 and f16 fail on the first
  `generate()`:

  ```
  [GPU] Incorrect block size for Paged Attention operation for key cache quant mode
  BY_CHANNEL. Expected 20, but got 12
  ```

  Only the plugin default works; it reports `KV_CACHE_PRECISION = dynamic`. The failing
  rows are recorded in `results/gpu_u8_kv_failure.jsonl` and `results/experiments/qwen3b.jsonl`.
  An upstream issue is drafted in `notes/upstream_issue_gpu_kv_precision.md` (not filed).
  For ORBIT this is a **hard constraint**: KV precision is a CPU-only setting on this
  stack.

### Finding 18 — Long context on the GPU, and a machine stall caught by telemetry (`plans/qwen3b_long_context.json`)

| Prompt | TTFT | Prefill rate | TPOT | Peak RSS |
|---|---|---|---|---|
| 2048 | 8.7 s | 235 tok/s | 56 ms | 2739 MiB |
| 4096 | 19.9 s | 206 tok/s | 53 ms | 3011 MiB |
| 8192 | 53.2 s | 154 tok/s | 61 ms | 3330 MiB |

- **Prefill is superlinear at long context.** Doubling the prompt raises TTFT 2.3× (from
  2k to 4k) and then 2.7× (from 4k to 8k), because attention cost grows with the square of
  the length. Decode slows only slightly.
- **One run stalled for 9.8 minutes** (4096 tokens, round 2): TPOT was 8425 ms/token
  against ~53 ms normally, and wall time was 588 s.
  - The during-run telemetry shows the whole machine was idle: own CPU 0%, background
    CPU 2%, against 11–14% in the neighbouring runs.
  - This points to the iGPU or system being put into a low-power state mid-run (for
    example a display timeout), not to anything in the model. It happened despite the
    runner holding a Windows "system required" request.
  - **Rule adopted:** `build_dataset.py` drops runs whose TTFT or TPOT exceeds 5× their
    own configuration's median. It drops exactly this one run. Ordinary noise stays
    within 2×.

### Finding 19 — On new data, the physical model generalizes and the trees do not

The Twin was re-run on the extended dataset: 557 runs, 94 configurations, now including
3B.

| Target | Noise floor | Analytical | GBM |
|---|---|---|---|
| TTFT, median APE | 11.4% | **15.7%** | 18.0% |
| TTFT, mean APE | 18.8% | **24.5%** | 393.7% |
| TPOT, median APE | 8.1% | 11.2% | **9.3%** |
| Peak RSS, median APE | 0.1% | **0.3%** | 0.4% |

- **The GBM's TTFT collapses on prefix-cache hits.** For held-out *repeated* 1920-token
  prompts it predicted ~13 s against an actual 58–77 ms (errors of 17,000–22,000%).
  Cache hits at 512 tokens were in its training folds, but the trees learned
  "long prompt → slow" without learning *why* hits are fast. The analytical model encodes
  the mechanism (a hit has near-constant TTFT) and predicted 60–69 ms.
- **On the 3B model, the analytical TTFT was 14–24% median error**, against 54–69% for
  the GBM.
- **Conclusion:** for a controller that has to handle configurations and workloads not in
  its training data, the mechanism-based Twin is more trustworthy, and it is explainable.
  `orbit.py` uses the analytical Twin. The GBM remains slightly better on TPOT within the
  training distribution.
