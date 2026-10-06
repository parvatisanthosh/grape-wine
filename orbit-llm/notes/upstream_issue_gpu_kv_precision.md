# Draft issue for openvinotoolkit/openvino.genai (not filed)

**Title:** [GPU] Setting `KV_CACHE_PRECISION` (u8 or f16) with the paged-attention LLMPipeline fails: "Incorrect block size for Paged Attention operation for key cache quant mode BY_CHANNEL. Expected 20, but got 12"

## Environment

- OpenVINO 2026.3.0 (pip), OpenVINO GenAI 2026.3.0.0 built from source at `bd8d6542`
- Windows 11, Intel Core i5-1335U, Intel Iris Xe Graphics (integrated)
- GPU plugin reports `KV_CACHE_PRECISION = dynamic` and `INFERENCE_PRECISION_HINT = f16` by default
- Model: `OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov` from Hugging Face. Also seen with TinyLlama-1.1B IR.

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
