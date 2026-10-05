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
   INT8 2.13 GB vs 1.05 GB, INT4 1.34 GB vs 0.62 GB. Since this holds even for INT8 and
   INT4, it is probably not FP16→FP32 up-conversion. More likely the IR is resident twice:
   the memory-mapped model file plus the compiled model's own weight copy. To test: load
   with `ov::enable_mmap(false)` and compare.
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
