# Draft issue (not filed): DO NOT FILE, no longer reproducible

**Status, 2026-11-09:** the crash below happened on 2026-10-06 but **cannot be reproduced
now** with the same OpenVINO 2026.3.0 pip install, the same GenAI build (`bd8d6542`), the
same Iris Xe driver (32.0.101.5542) and the same `orbit_run` code. Rerun of the exact
original settings, all successful:

- Qwen2.5-Coder-3B INT4 on GPU, `--kv-precision` u8 and f16, 256 and 1024 prompt tokens,
  64 output tokens, shared `ov_cache/gpu`;
- `cpp/repro/repro_gpu_kv_precision.cpp`, a minimal standalone program, with u8 and with
  the default precision;
- TinyLlama INT4 with u8 and f16, with and without the model cache, and with a fresh cache.

What changed is unknown. On 2026-10-06 the u8 run seems to have loaded a model that was
already in the compiled-model cache, but this could not be confirmed. Today's u8 runs
compiled new cache entries. **Do not file this unless it reproduces again.**

Related upstream report: openvinotoolkit/openvino#39615 (opened 2026-11-08). It hits the
same `Incorrect block size ... BY_CHANNEL` check, on Arc B60 with f32 inference precision
and u4 KV cache. In OpenVINO 2026.3, `transformations_pipeline.cpp` sizes the BY_CHANNEL
key block from the inference precision, while `paged_attention.cpp` expects a fixed
fp16 layout. If the crash returns, add the details as a comment there instead of opening
a new issue: "Expected 20, but got 12" is the u8 expectation meeting a u4-sized block.

---

The original draft follows. One correction: the crash was recorded only with the Qwen 3B
model, not with TinyLlama.

**Title:** [GPU] Setting `KV_CACHE_PRECISION` (u8 or f16) with the paged-attention LLMPipeline fails: "Incorrect block size for Paged Attention operation for key cache quant mode BY_CHANNEL. Expected 20, but got 12"

## Environment

- OpenVINO 2026.3.0 (pip), OpenVINO GenAI 2026.3.0.0 built from source at `bd8d6542`
- Windows 11, Intel Core i5-1335U, Intel Iris Xe Graphics (integrated)
- GPU plugin reports `KV_CACHE_PRECISION = dynamic` and `INFERENCE_PRECISION_HINT = f16` by default
- Model: `OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov` from Hugging Face.

## Steps to reproduce (C++ API)

Construct `ov::genai::LLMPipeline(models_path, "GPU", properties)` with:

- `ov::genai::scheduler_config(...)`, using the paged-attention / continuous-batching
  backend. The scheduler values are the latency-oriented defaults: unlimited
  `max_num_batched_tokens`, prefix caching on.
- `ov::hint::kv_cache_precision(ov::element::u8)`, **or** `ov::element::f16`.

Then call `generate()` on any prompt.

## Actual result

Pipeline construction succeeds. The first `generate()` throws:

```
Exception from src\inference\src\cpp\infer_request.cpp:224:
Check 'valid_block_size' failed at src\plugins\intel_gpu\src\graph\paged_attention.cpp:73:
[GPU] Incorrect block size for Paged Attention operation for key cache quant mode BY_CHANNEL. Expected 20, but got 12
```

## Expected result

Either generation works with the requested KV-cache precision, or construction rejects the
property with a clear message.

## Notes

- Without `kv_cache_precision` the same GPU pipeline works correctly. That is the
  plugin's dynamic default.
- On CPU, the same properties work for u8, f16 and f32.
- The message suggests the block size GenAI computes for the KV cache does not match the
  layout the GPU plugin expects for BY_CHANNEL key-cache quantization. It fails even when
  f16 is requested, so the quant mode seems to be chosen independently of the requested
  precision.
- Failing rows with full error text: `results/gpu_u8_kv_failure.jsonl` and
  `results/experiments/qwen3b.jsonl` (`status = warmup_failed`).

## Before filing

- Check for an existing issue: search "BY_CHANNEL" and "Incorrect block size".
- Reproduce with a minimal standalone program, not `orbit_run`, and attach it.
- Try the latest nightly to see if it is already fixed.
