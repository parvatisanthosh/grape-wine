"""Minimal repro: on GPU, a model-cache hit ignores KV_CACHE_PRECISION.

Compiles the same paged-attention model twice into one fresh CACHE_DIR, first
with precision A, then with precision B, and prints the KV cache inputs of each
compiled model. Without the cache, A and B give different cache layouts. With
it, the second compile returns the first blob: the cache layout is still A's,
while get_property(KV_CACHE_PRECISION) reports B.

Usage (OpenVINO only, no GenAI needed):
    python repro_gpu_cache_kv_precision.py MODEL_DIR [A] [B]
    defaults: A = f16, B = u8
Exit code 1 if the problem reproduces.
"""

import sys
import tempfile
from pathlib import Path

import openvino as ov
from openvino._offline_transformations import paged_attention_transformation


def compile_layout(core, model_dir, precision, cache_dir=None):
    model = core.read_model(Path(model_dir) / "openvino_model.xml")
    # The transformation GenAI's LLMPipeline applies before compiling.
    paged_attention_transformation(model, False, False, True, False, False, False)
    properties = {"KV_CACHE_PRECISION": precision}
    if cache_dir:
        properties["CACHE_DIR"] = cache_dir
    compiled = core.compile_model(model, "GPU", properties)
    key = next(p for p in compiled.inputs if any(n.startswith("key_cache.") for n in p.get_names()))
    shape = key.get_partial_shape()
    return (str(compiled.get_property("KV_CACHE_PRECISION")), str(key.get_element_type()),
            [shape[i].get_length() for i in range(1, 4)])


def main():
    model_dir = sys.argv[1]
    first, second = (sys.argv[2:4] + ["f16", "u8"][len(sys.argv[2:4]):])[:2]
    core = ov.Core()
    print("OpenVINO", ov.get_version(), "|", core.get_property("GPU", "FULL_DEVICE_NAME"))

    no_cache = {p: compile_layout(core, model_dir, p) for p in (first, second)}
    for precision, (reported, element, shape) in no_cache.items():
        print(f"no cache    {precision}: reported {reported}, key_cache.0 {element} {shape}")

    with tempfile.TemporaryDirectory() as cache_dir:
        cached = {p: compile_layout(core, model_dir, p, cache_dir) for p in (first, second)}
    for precision, (reported, element, shape) in cached.items():
        print(f"CACHE_DIR   {precision}: reported {reported}, key_cache.0 {element} {shape}")

    wrong = cached[second][1:] != no_cache[second][1:]
    print("REPRODUCED: the cached compile kept the first precision's KV layout"
          if wrong else "Not reproduced: the cache respected KV_CACHE_PRECISION")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
