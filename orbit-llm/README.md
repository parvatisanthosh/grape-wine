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
   │           risk OOM, pick by objective (latency/memory/quality),   │
   │           report the Pareto front                                 │
   │                                                                   │
   │  Layer 3  Online correction: run, compare actual vs predicted,    │
   │           update per-candidate correction factors                 │
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

# Reproduce an experiment, then rebuild the dataset and re-evaluate the Twin
python run_experiment.py plans/device_cores.json
python build_dataset.py
python twin_v0.py
```

## Key results so far

Full results are in [FINDINGS.md](FINDINGS.md).

- **Quantization (TinyLlama-1.1B, CPU):** INT8 decodes 1.8–2.0× faster than FP16 and INT4
  2.2–3.1× faster, with 47–49% and 66–68% less memory. On this CPU, "FP16" actually runs
  in f32, because the CPU has no native FP16.
- **Prefix caching:** a repeated 1920-token prompt reaches its first token in ~75 ms
  instead of ~13 s (≈ 177×). For unique prompts it has no measurable cost.
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
- **Performance Twin:** simple physical laws predict TTFT and TPOT within ~4 percentage
  points of the run-to-run noise floor on unseen configurations, and peak memory to ~0.3%
  — as accurately as gradient-boosted trees.
- **3B model:** on CPU, a 1920-token prompt takes 50 s to first token; the iGPU is
  5–6× faster.

## Status

| Phase | Status |
|---|---|
| 1. C++ baseline / executor | done |
| 2. Benchmarking framework | done |
| 3. Quantization study | done |
| 4. KV-cache study | done |
| 5. Device and core placement | done |
| 5. Performance Twin v0 | done |
| 6. Optimizer and online correction | first version (`orbit.py`) |
| 6. 3B model | in progress |
| 7. Quality metric | written (`quality_eval.py`), not yet run |
| 8. Evaluation (baseline vs static vs ORBIT) | next |
