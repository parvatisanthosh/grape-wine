# Local LLM on OpenVINO — with ORBIT-LLM

Task **SWFW_5S_13 — Local LLM on OpenVINO**: run a 1B–3B parameter LLM entirely locally
using the **OpenVINO GenAI C++ API**, and understand inference, quantization, the KV
cache, hardware utilization and KV-cache management in OpenVINO.

- **Deliverable:** [`cpp/orbit_chat`](cpp/orbit_chat.cpp) — a local multi-turn chat
  application on the GenAI C++ API. It streams answers, lets you choose model and device
  (CPU or the Iris Xe iGPU), exposes OpenVINO's KV-cache management (prefix caching, KV
  precision, cache eviction), and reports TTFT, tokens/s and memory every turn.
- **Report:** [REPORT.md](REPORT.md) — results organized by the task's four learning
  outcomes.
- **Lab log:** [FINDINGS.md](FINDINGS.md) — every experiment, its method, raw-data path
  and conclusion.
- **Extension:** **ORBIT-LLM** (*Online Resource-Budgeted Intelligent Tuning*) — a
  controller that predicts how each configuration would perform on this machine, picks
  the best one that meets the user's constraints, and corrects itself from measured
  results.

All numbers come from one laptop: Intel Core i5-1335U (2 P-cores + 8 E-cores), Iris Xe
iGPU, 16 GB RAM, OpenVINO 2026.3. Models: TinyLlama-1.1B-Chat (FP16/INT8/INT4) and
Qwen2.5-Coder-3B-Instruct (INT4).

## Demo

```powershell
# Build once (details in cpp/README.md)
cmake -S cpp -B cpp/build -G "Visual Studio 17 2022" -A x64 `
      -DOpenVINO_DIR="C:/Users/parva/venv/Lib/site-packages/openvino/cmake"
cmake --build cpp/build --config Release

$env:Path = "C:\Users\parva\venv\Lib\site-packages\openvino\libs;$env:Path"

# Chat with TinyLlama on the CPU
.\cpp\build\Release\orbit_chat.exe --model C:\Users\parva\tinyllama-int4

# Chat with the 3B model on the iGPU
.\cpp\build\Release\orbit_chat.exe --model C:\Users\parva\qwen2.5-coder-3b-int4 --device GPU --cache-dir ov_cache\gpu

# Scripted conversation with KV-cache eviction and per-turn metrics
.\cpp\build\Release\orbit_chat.exe --model C:\Users\parva\tinyllama-int4 `
    --script conversations\long_recall.txt --evict 32:128:512
```

## ORBIT-LLM architecture

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
| `cpp/orbit_chat.cpp` | **The local chat application** (GenAI C++ API): multi-turn chat with streaming and per-turn metrics, plus KV-cache management options. |
| `cpp/orbit_worker.cpp` | Long-lived process that keeps models loaded and serves requests over stdin/stdout (`orbit_worker_client.py`). |
| `run_chat_study.py`, `conversations/` | KV-cache studies through the chat app, e.g. cache eviction on a long conversation with a recall probe. |
| `cpp/orbit_run.cpp` | C++ executor on the GenAI C++ API. One configuration per process: device, attention backend, prefix caching, KV-cache precision and size, threads, core type, model cache. Writes one JSON line per run with TTFT, TPOT, throughput, working set, committed memory, and CPU state before and during the run. See [cpp/README.md](cpp/README.md). |
| `orbit_exec.py` | Calls `orbit_run.exe` from Python and records crashes as rows. |
| `run_experiment.py`, `plans/` | Plan-driven experiments: a grid plus named variants, shuffled per round, resumable, keeps the machine awake. |
| `analyze_experiment.py` | Median [min–max] per configuration for any experiment. |
| `build_dataset.py` | Merges all executor runs into `results/dataset.csv`, keeping only pre-run features. |
| `twin_v0.py` | Layer 1: analytical and gradient-boosted predictors, evaluated against the noise floor on held-out configurations. |
| `orbit.py` | Layers 2 and 3: chooses a configuration for a request, optionally runs it and learns from the result. |
| `quality_eval.py` | Quality proxy: bits per byte on a fixed local corpus, comparable across tokenizers. |
| `colab/`, `summarize_humaneval.py` | Task-level quality: HumanEval on the 3B model at FP16, INT8 and INT4, run on Colab and summarized locally. |
| `prefix_overwrite_test.py` | Tests whether a prompt survives in the prefix cache when other prompts run in between. |
| `prefix_pool_study.py` | Finds how many conversations a fixed KV pool keeps, against the prediction from the OpenVINO source (Finding 26). |
| `data/` | Real datasets for evaluation; see [data/README.md](data/README.md). |
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
- **Quantization (Qwen2.5-Coder-3B, HumanEval):** FP16 and INT8 each pass 137 of 164
  problems and INT4 passes 135, with no detectable difference. INT4 weights are 71%
  smaller than FP16.
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

| Part | Status |
|---|---|
| Local chat application on the GenAI C++ API (`orbit_chat`) | done |
| Measurement executor and worker (`orbit_run`, `orbit_worker`) | done |
| Quantization study (FP16 / INT8 / INT4) | done |
| KV-cache study: precision, size, prefix caching, attention backend | done |
| KV-cache eviction study | done (Finding 21) |
| Hardware: CPU vs iGPU, core placement, model cache, residency, contention, battery | done |
| 3B model (Qwen2.5-Coder-3B INT4) | done |
| Performance Twin, optimizer, online correction | done |
| ORBIT evaluation vs static configuration | preliminary (REPORT.md §5) |
| Quality metric (bits per byte) | done |
| Task-level quality of the 3B model (HumanEval) | done (Finding 25) |
