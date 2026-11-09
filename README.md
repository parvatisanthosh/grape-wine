# ORBIT-LLM

**Online Resource-Budgeted Intelligent Tuning for local LLM inference** — a controller on
top of the OpenVINO GenAI C++ API. For each request it predicts how every available
inference configuration would perform on *this* machine, picks the best one that meets
the user's constraints, runs it, and corrects its predictions from what actually
happened.

Built from task **SWFW_5S_13 — Local LLM on OpenVINO**: run a 1B–3B LLM locally with the
OpenVINO GenAI C++ API, and understand inference, quantization, the KV cache and hardware
utilization. Every claim here comes from experiments on one laptop: Intel Core i5-1335U
(2 P-cores + 8 E-cores), Iris Xe iGPU, 16 GB RAM, OpenVINO 2026.3. Full results and
methodology are in **[FINDINGS.md](FINDINGS.md)**.

## Architecture

```
 request (prompt length, output length, constraints, objective)
   │
   ▼
 orbit.py ─────────────────────────────────────────────────────────────┐
   │  Workload + machine state: prompt/output tokens, prefix reuse,    │
   │  free RAM, CPU load                                               │
   │                                                                   │
   │  Layer 1  Performance Twin (twin_v0.py)                           │
   │           predicts load time, TTFT, TPOT, peak memory for each    │
   │           measured (model, device/placement) candidate            │
   │                                                                   │
   │  Layer 2  Optimizer: drop candidates that break constraints or    │
   │           risk OOM, optionally even in a calibrated bad case      │
   │           (safety margins), pick by objective                     │
   │           (latency/memory/quality), fall back honestly            │
   │           (robust → point-only → best-effort) when nothing fits   │
   │                                                                   │
   │  Layer 3  Online correction: run, compare actual vs predicted,    │
   │           update correction factors: TTFT split by prefix-cache   │
   │           hit/miss, TPOT, memory, and model load time             │
   ▼                                                                   │
 orbit_exec.py ──► cpp/orbit_run.exe  (OpenVINO GenAI C++ API)         │
                     LLMPipeline + SchedulerConfig, KV-cache and       │
                     threading properties, exact-length prompts,       │
                     JSON metrics + memory/CPU telemetry ──────────────┘
                                         │
 run_experiment.py + plans/*.json ───────┘  (systematic experiments → dataset)
```

## Components

| File | Role |
|---|---|
| `cpp/orbit_run.cpp` | C++ executor on the GenAI C++ API. One configuration per process: device, attention backend, prefix caching, KV-cache precision and size, threads, core type, model cache. Writes one JSON line per run with TTFT, TPOT, throughput, working set, committed memory, and CPU state before and during the run. See [cpp/README.md](cpp/README.md). |
| `orbit_exec.py` | Calls `orbit_run.exe` from Python and records crashes as rows. |
| `run_experiment.py`, `plans/` | Plan-driven experiments: a grid plus named variants, shuffled per round, resumable, keeps the machine awake. |
| `analyze_experiment.py` | Median [min–max] per configuration for any experiment. |
| `build_dataset.py` | Merges all executor runs into `results/dataset.csv`, keeping only pre-run features. |
| `twin_v0.py` | Layer 1: analytical and gradient-boosted predictors, evaluated against the noise floor on held-out configurations. |
| `orbit.py` | Layers 2 and 3: chooses a configuration for a request, optionally runs it and learns from the result. |
| `evaluate_controller.py` | Evaluation: OpenVINO defaults vs a fixed INT4-iGPU configuration vs ORBIT (with and without safety margins). Interleaved and paired: every request is run by all systems back-to-back in a seeded random order. Calm phase plus a busy phase with background CPU load. |
| `replay_correction.py` | Replays logged runs through the online correction with a simulated slowdown, to show how fast predictions recover (no hardware needed). |
| `noise_gpu.py` | Keeps the iGPU busy by running another LLM on it in a loop, for the GPU-contention experiment. |
| `quality_eval.py` | Quality proxy: bits per byte on a fixed local corpus, comparable across tokenizers. |
| `models.json` | Model registry: TinyLlama-1.1B FP16/INT8/INT4, Qwen2.5-Coder-3B INT4. |

## Quick start

```powershell
# 1. Build the executor (see cpp/README.md for details)
cmake -S cpp -B cpp/build -G "Visual Studio 17 2022" -A x64 `
      -DOpenVINO_DIR="C:/Users/parva/venv/Lib/site-packages/openvino/cmake"
cmake --build cpp/build --config Release

# 2. Ask the controller for a configuration
python orbit.py --prompt-tokens 1024 --output-tokens 128 --max-ttft-ms 3000

# 3. ... and run it, learning from the result
python orbit.py --prompt-tokens 1024 --output-tokens 128 --max-ttft-ms 3000 --run

# Run the controller evaluation (takes a few hours; leave the laptop alone)
python evaluate_controller.py --systems baseline static orbit orbit_robust --repeats 5 --busy-repeats 5

# GPU-contention experiment: calm, then busy (noise_gpu.py running in a second terminal)
python run_experiment.py plans/gpu_contention_calm.json
python run_experiment.py plans/gpu_contention_busy.json

# Reproduce an experiment, then rebuild the dataset and re-evaluate the Twin
python run_experiment.py plans/device_cores.json
python build_dataset.py
python twin_v0.py
```

## Key results

Full methodology and per-experiment numbers are in [FINDINGS.md](FINDINGS.md).

### Hardware, quantization and the KV cache

- **Quantization (TinyLlama-1.1B, CPU):** INT8 decodes 1.8–2.0× faster than FP16 and INT4
  2.2–3.1× faster, with 47–49% and 66–68% less memory. On this CPU, "FP16" actually runs
  in f32, because the CPU has no native FP16.
- **Quality cost of quantization** (bits per byte on a fixed local corpus, lower is
  better): FP16 0.692, INT8 0.692, INT4 0.702 — INT4 costs only ~0.01 BPB. Moving to
  Qwen2.5-Coder-3B (INT4) improves it to 0.552, so model size matters far more for quality
  than precision does.
- **Prefix caching:** a 1920-token prompt repeated within the same process reaches its
  first token in ~75 ms instead of ~13 s (≈ 177×). For unique prompts it has no measurable
  cost.
- **KV cache:** u8 (the CPU default), f16 and f32 cost exactly their theoretical sizes.
  At 1.1B the KV cache is small next to the weights, and prompt-length activations grow
  ~12× faster than it does.
- **Memory measurement:** a pre-allocated KV cache is fully committed but invisible in
  the working set; peak commit reaches ~2× the cache size. OOM risk must be predicted
  from committed memory.
- **Hardware:** the Iris Xe iGPU beats the best CPU configuration by 3–5× on TTFT and
  1.4× on decode, with far less variance. On the hybrid CPU, two P-cores beat the
  12-thread default on decode by 16–40%. The model cache cuts GPU load time from ~8 s to
  ~2.5 s but slows CPU load ~5×.
- **3B model:** on CPU, a 1920-token prompt takes 50 s to first token; the iGPU is
  5–6× faster.

### Performance Twin and online correction

- **Twin:** simple physical laws predict TTFT and TPOT within ~4 percentage points of the
  run-to-run noise floor on unseen configurations, and peak memory to ~0.3% — as
  accurately as gradient-boosted trees. TTFT uses a power law in prompt length (prefill
  grows faster than linearly), which removed impossible negative predictions for the 3B
  model.
- **Online correction:** after a simulated 1.25× memory slowdown, memory-prediction error
  falls from ~20% to under 1% within about 4 requests (`replay_correction.py`). TTFT
  correction helps after a real change but chases noise when nothing changes, so a
  lower learning rate is safer.
- **Cache-aware correction:** prefix-cache hits (~20 ms TTFT) were dragging a shared
  correction factor down and making long cache-miss prompts under-predicted by 15–40%.
  TTFT is now corrected separately for hits and misses; decode keeps one shared factor.

### Controller evaluation

40 requests per system (5 scenario types × 5 repeats on a calm machine, 3 × 5 under
background CPU load), run **interleaved and paired**: each request is run by every system
back-to-back in a seeded random order, so all systems face the same machine conditions.

| System | SLA violations | Mean total time | Quality (BPB) | Model switches |
|---|---|---|---|---|
| OpenVINO defaults (INT8, CPU) | 60% | 28.7 s | 0.69 | 1 |
| Fixed INT4 on iGPU | 22.5% | 4.7 s | 0.70 | 1 |
| ORBIT | 27.5% | 8.0 s | 0.66 | 13 |
| ORBIT + safety margins | 27.5% | 9.5 s | 0.69 | 13 |

Paired comparison — requests where exactly one of the two systems kept its promise:

| A vs B | Only A kept | Only B kept | Exact two-sided p (McNemar) |
|---|---|---|---|
| ORBIT vs OpenVINO defaults | 13 | 0 | ≈ 0.0002 |
| ORBIT vs fixed INT4-iGPU | 0 | 2 | 0.5 (no significant difference) |

- **ORBIT beats the OpenVINO defaults decisively, and ties a hand-tuned fixed
  configuration** on SLA attainment. It delivers better output quality when quality is
  requested (0.66 vs 0.70 BPB), at the cost of more model switches.
- **The safety label is informative:** in an earlier run, choices ORBIT marked "robust"
  (safe even in the calibrated bad case) failed 5% of the time (1/20), while "point-only"
  choices failed 40% (8/20).
- **A timing confound was found and fixed:** in the first evaluation each system ran its
  whole trace after the previous one, and identical configurations got different results
  hours apart. Interleaving removed that bias.

### Why a fixed configuration ties ORBIT: the GPU-contention test

An adaptive controller only pays off if some condition changes which configuration is
best. The most plausible one here is another application using the shared iGPU. The same
12 configurations were measured on a calm machine and while `noise_gpu.py` kept the iGPU
busy with a second LLM (median of 6 runs, total time = TTFT + 64 output tokens):

| Model | Prompt | iGPU calm → busy | CPU P-cores calm → busy | Best when busy |
|---|---|---|---|---|
| INT4 | 128 | 2.1 → 3.4 s | 3.2 → 6.1 s | iGPU |
| INT4 | 1024 | 3.4 → 5.5 s | 13.8 → 24.4 s | iGPU |
| INT8 | 128 | 2.4 → 4.8 s | 4.0 → 7.9 s | iGPU |
| INT8 | 1024 | 3.8 → 7.8 s | 12.9 → 20.1 s | iGPU |

**No crossover.** Contention slowed the iGPU 1.6–2.0×, but the CPU slowed just as much
(1.4–2.0×), so the iGPU stayed best in every case. The iGPU has no memory of its own: it
shares RAM and memory bandwidth with the CPU, and decode is memory-bandwidth-bound, so
the CPU is not an independent fallback on this laptop. GPU model loading suffered most
(~6 s → 12–36 s).

**Conclusion:** on this laptop and these workloads, a fixed INT4-on-iGPU configuration is
sufficient, and that is why ORBIT ties it. ORBIT's measurable value is against default
settings and in quality-aware model selection.

## Limitations

- **Simulated residency.** Every run is a fresh process that reloads the model; the
  evaluator charges load time only when a system switches configuration. A persistent
  worker that keeps models loaded would measure this directly.
- **One laptop, small samples.** 40 requests per system; small differences between
  systems are not statistically meaningful.
- **Quality proxy.** Bits per byte on a small corpus, not task accuracy.
- **iGPU memory** is only partly visible in process memory counters.
- **The contention test** loads the iGPU with another LLM, which also uses some CPU and
  memory bandwidth; it is realistic ("another AI app is running"), not pure GPU-only load.

## Status

| Phase | Status |
|---|---|
| 1. C++ baseline / executor | done |
| 2. Benchmarking framework | done |
| 3. Quantization study | done |
| 4. KV-cache study | done |
| 5. Device and core placement | done |
| 5. Performance Twin v0 | done |
| 6. Optimizer and online correction | done (`orbit.py`) |
| 6. 3B model | done |
| 7. Quality metric | done (`quality_eval.py`) |
| 8. Evaluation (defaults vs fixed vs ORBIT) | done, interleaved and paired |
| 9. GPU-contention crossover test | done: no crossover |
| Possible next | persistent C++ worker for measured residency |
