# To file at github.com/openvinotoolkit/openvino (the user files it, not the assistant)

Searched 2026-11-11 with the GitHub issue search across openvinotoolkit. No existing
report found. Related but different: openvinotoolkit/openvino#39615 (the GPU
"Incorrect block size … BY_CHANNEL" check).

---

**Title:** [GPU] Model cache (CACHE_DIR) ignores KV_CACHE_PRECISION: a cache hit returns the KV layout of the first compile

**Describe the bug**

On GPU, compiling a paged-attention model with `CACHE_DIR` set returns the cached blob
even when `KV_CACHE_PRECISION` differs from the compile that created it.
- The returned model keeps the first compile's KV cache inputs: element type and block
  shape.
- `get_property(KV_CACHE_PRECISION)` reports the newly requested value, as a plain
  string rather than an element type.

So the requested KV precision is silently not applied. LLMPipeline users are affected,
because GenAI sizes and allocates the KV cache from those compiled inputs.

**Environment**
- OpenVINO 2026.3.0-22451-8a17657b995 (pip) and 2026.4.1-22982-e213a147257 (pip): both
  reproduce
- Intel Iris Xe Graphics (i5-1335U iGPU), driver 32.0.101.5542, Windows 11
- Model: TinyLlama-1.1B-Chat INT4 OpenVINO IR. The same happens with
  Qwen2.5-Coder-3B-Instruct INT4.

**To reproduce** (OpenVINO only, no GenAI):

```python
import tempfile, openvino as ov
from openvino._offline_transformations import paged_attention_transformation

def layout(core, path, precision, cache_dir=None):
    m = core.read_model(path + "/openvino_model.xml")
    paged_attention_transformation(m, False, False, True, False, False, False)  # as GenAI LLMPipeline
    props = {"KV_CACHE_PRECISION": precision}
    if cache_dir:
        props["CACHE_DIR"] = cache_dir
    c = core.compile_model(m, "GPU", props)
    k = next(p for p in c.inputs if any(n.startswith("key_cache.") for n in p.get_names()))
    return c.get_property("KV_CACHE_PRECISION"), k.get_element_type(), k.get_partial_shape()

core = ov.Core()
path = "TinyLlama-1.1B-Chat-v1.0-int4-ov"
print(layout(core, path, "f16"))            # f16, f16 [?,4,64,16]
print(layout(core, path, "u8"))             # i8,  i8  [?,4,64,20]
with tempfile.TemporaryDirectory() as d:
    print(layout(core, path, "f16", d))     # f16, f16 [?,4,64,16]
    print(layout(core, path, "u8", d))      # 'u8', f16 [?,4,64,16]   <-- f16 layout returned
```

Output on 2026.4.1:

```
no cache    f16: reported <Type: 'float16'>, key_cache.0 <Type: 'float16'> [4, 64, 16]
no cache    u8: reported <Type: 'int8_t'>, key_cache.0 <Type: 'int8_t'> [4, 64, 20]
CACHE_DIR   f16: reported <Type: 'float16'>, key_cache.0 <Type: 'float16'> [4, 64, 16]
CACHE_DIR   u8: reported u8, key_cache.0 <Type: 'float16'> [4, 64, 16]
```

The opposite order (u8 first, then f16) returns the int8 layout for f16.

**Expected behavior**

Either `KV_CACHE_PRECISION` is part of the model-cache key, so a different precision
compiles and caches a new blob, or the property is rejected or warned about on a cache
hit. In either case, `get_property` should report the precision that is actually in use.

**Possibly related**

On 2026-10-06, with a shared `CACHE_DIR`, explicit `u8` and `f16` KV precisions on this
GPU failed at the first `generate()` with:

```
Incorrect block size for Paged Attention operation for key cache quant mode BY_CHANNEL. Expected 20, but got 12
```

A key block of 12 is the 4-bit layout, which is the GPU default here (`dynamic` resolves
to u4, with key [?,4,64,12] and value [?,4,16,36]). That error could be a cached 4-bit
blob meeting an 8-bit request. We could not reproduce it again, so this link is only a
guess.

Repro script: `orbit-llm/cpp/repro/repro_gpu_cache_kv_precision.py` (exit code 1 = reproduced).
