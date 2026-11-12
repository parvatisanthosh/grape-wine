"""GPU memory of the KV cache per precision, measured (Finding 30).

For each KV precision and prompt length, a fresh orbit_generate process loads
Qwen2.5-Coder-3B INT4 on the iGPU and answers one long-context prompt from
data/needle_prompts.jsonl. The process's shared GPU memory (gpu_memory.py) is
read 1 s after loading and then sampled for its peak while it generates.

The first sample already includes the KV cache: GenAI allocates the KV blocks
at the start of generate(), and prefill then takes 5-20 s. So the two values
are nearly equal, and the measurement is the comparison ACROSS precisions at
the same prompt length: weights and activations are the same, so the
difference is KV memory, to set against the exact layout from kv_layout.py.

    python kv_memory.py [--kv default u8 f16] [--repeats 2]
Output: results/kv_memory.jsonl
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from gpu_memory import GpuMemoryReader, GpuMemorySampler
from kv_quality import GENERATE, MODELS, NEEDLE_PROMPTS, load_jsonl
from orbit_exec import OPENVINO_LIBS

OUT = Path("results/kv_memory.jsonl")
LENGTHS = [1000, 2000, 3500]


def measure(kv, prompt, env):
    with tempfile.TemporaryDirectory() as folder:
        prompts = Path(folder) / "prompt.jsonl"
        answers = Path(folder) / "answer.jsonl"
        prompts.write_text(json.dumps({"id": prompt["id"], "content": prompt["content"]}) + "\n",
                           encoding="utf-8")
        command = [str(GENERATE), "--model", MODELS["Q3B-INT4"], "--device", "GPU",
                   "--kv-precision", kv, "--prompts", str(prompts), "--out", str(answers),
                   "--max-new-tokens", "16"]
        process = subprocess.Popen(command, env=env, stderr=subprocess.PIPE, text=True,
                                   encoding="utf-8", errors="replace")
        reader = None
        while reader is None and process.poll() is None:
            try:
                reader = GpuMemoryReader(process.pid)
            except OSError:
                time.sleep(0.2)
        after_load = None
        sampler = None
        for line in process.stderr:
            if line.startswith("Loaded in"):
                time.sleep(1.0)  # let the driver settle
                reader.read_mib()
                after_load = reader.read_mib()
                sampler = GpuMemorySampler(process.pid, interval_s=0.25)
                sampler.start()
        process.wait()
        peak = sampler.stop() if sampler else None
        reader.close()
        row = load_jsonl(answers)[0] if answers.exists() else {}
    return {"kv": kv, "id": prompt["id"], "prompt_tokens": row.get("input_tokens"),
            "first_sample_mib": after_load, "peak_mib": peak,
            "ttft_ms": row.get("ttft_ms"), "exit_code": process.returncode}


def main():
    from run_experiment import prevent_sleep, require_ac_power

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kv", nargs="+", default=["default", "u8", "f16"])
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    require_ac_power(False)
    prevent_sleep()

    env = dict(os.environ)
    env["PATH"] = f"{OPENVINO_LIBS}{os.pathsep}{env['PATH']}"
    prompts = {row["length"]: row for row in load_jsonl(NEEDLE_PROMPTS) if row["depth"] == 0.5}
    with OUT.open("a", encoding="utf-8") as out:
        for repeat in range(args.repeats):
            for length in LENGTHS:
                for kv in args.kv:
                    row = {"repeat": repeat, "length": length, **measure(kv, prompts[length], env)}
                    print(json.dumps(row), flush=True)
                    out.write(json.dumps(row) + "\n")
                    out.flush()


if __name__ == "__main__":
    main()
