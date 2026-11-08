"""How long does a prompt stay in the prefix cache while other requests run?

The worker evaluation thought static got 5 prefix-cache hits on follow_up, but
every one took ~3.8 s, the same as a miss. The residency experiment had the
same hidden pattern: a repeated prompt hit (~40 ms) when repeated right away,
but missed (~1.6-2.1 s) after three other 1024-token prompts in between.

Hypothesis: with cache_size_gb = 0 the KV cache is allocated dynamically and
only about as big as the work in flight, so other requests reuse the blocks
that held the old prompt. A fixed cache_size_gb keeps a pool big enough for
the old prefix to survive.

Test, for each cache size: run a conversation prompt, run F tokens of other
prompts ("filler"), then repeat the conversation and time its TTFT. A hit
means the prefix survived F tokens of other traffic.

    python cache_retention_experiment.py
    python cache_retention_experiment.py --cache-sizes 0 1 2 --repeats 3

Results go to results/cache_retention.jsonl. Not results/experiments/:
build_dataset.py reads every file there into the twin's training data.
"""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

from orbit_worker_client import OrbitWorker

MODELS_JSON = Path("models.json")
OUTPUT = Path("results/cache_retention.jsonl")

LABEL = "INT4"
SLOT = "INT4@gpu"
GPU_CACHE_DIR = "ov_cache/gpu"

CONVERSATION_ID = "run_1#conversation"
CONVERSATION_TOKENS = 1920  # the follow_up scenario's prompt
FILLER_TOKENS = 1024  # each filler request
OUTPUT_TOKENS = 16  # short outputs: the cache is what is being measured

# Tokens of other traffic between the conversation and its repeat.
FILLER_AMOUNTS = [0, 1024, 2048, 4096, 8192]
FILLER_VARIANTS = ["run_2", "run_3", "run_4", "run_5"]

# A repeat counts as a hit if its TTFT is under this share of a miss.
HIT_FRACTION = 0.3


def get_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cache-sizes",
        type=int,
        nargs="+",
        default=[0, 1],
        help="KV cache sizes in GB; 0 = dynamic (what every run used so far).",
    )
    parser.add_argument("--repeats", type=int, default=2)
    return parser.parse_args()


def save(row):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row) + "\n")


def run_cache_size(model_path, cache_size_gb, repeats):
    rows = []
    filler_number = 0

    def generate(worker, step, prompt_id, tokens, **extra):
        reply = worker.generate(SLOT, tokens, OUTPUT_TOKENS, prompt_id=prompt_id)
        row = {
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "cache_size_gb": cache_size_gb,
            "step": step,
            **extra,
            **reply,
        }
        save(row)
        rows.append(row)
        if reply.get("status") != "success":
            raise RuntimeError(f"{step} failed: {reply.get('error', reply.get('status'))}")
        return reply

    print(f"\n=== cache_size_gb = {cache_size_gb} ===", flush=True)
    with OrbitWorker() as worker:
        options = {"label": LABEL, "device": "GPU", "cache_dir": GPU_CACHE_DIR}
        if cache_size_gb:
            options["cache_size_gb"] = cache_size_gb
        load = worker.load(SLOT, model_path, **options)
        save({"cache_size_gb": cache_size_gb, "step": "load", **load})
        if load["status"] != "success":
            raise RuntimeError(f"load failed: {load.get('error')}")

        # The first request after a load pays the GPU first-inference cost.
        generate(worker, "first_after_load", "warmup#first", FILLER_TOKENS)

        for repeat in range(1, repeats + 1):
            for filler in FILLER_AMOUNTS:
                # (Re)run the conversation so it is the most recent prompt.
                generate(worker, "prime", CONVERSATION_ID, CONVERSATION_TOKENS,
                         repeat=repeat, filler_tokens=filler)
                for _ in range(filler // FILLER_TOKENS):
                    filler_number += 1
                    variant = FILLER_VARIANTS[filler_number % len(FILLER_VARIANTS)]
                    generate(worker, "filler", f"{variant}#filler{filler_number}",
                             FILLER_TOKENS, repeat=repeat, filler_tokens=filler)
                generate(worker, "repeat", CONVERSATION_ID, CONVERSATION_TOKENS,
                         repeat=repeat, filler_tokens=filler)

        worker.unload(SLOT)

    return rows


def summarize(rows):
    # A miss for reference: the first prime of each cache size is always a miss.
    print("\n" + "=" * 66)
    print("PREFIX-CACHE RETENTION (conversation repeat TTFT, ms, median)")
    print("=" * 66)
    for cache_size in sorted({row["cache_size_gb"] for row in rows}):
        mine = [row for row in rows if row["cache_size_gb"] == cache_size]
        primes = [row for row in mine if row["step"] == "prime"]
        miss_ms = primes[0]["ttft_ms"]
        label = "dynamic" if cache_size == 0 else f"{cache_size} GB"
        print(f"\nKV cache {label}   (miss reference: {miss_ms:,.0f} ms)")
        print(f"  {'other tokens between':>22}{'TTFT':>10}{'hits':>8}")
        for filler in FILLER_AMOUNTS:
            repeats = [row for row in mine if row["step"] == "repeat" and row["filler_tokens"] == filler]
            if not repeats:
                continue
            ttfts = [row["ttft_ms"] for row in repeats]
            hits = sum(ttft < HIT_FRACTION * miss_ms for ttft in ttfts)
            print(f"  {filler:>22,}{statistics.median(ttfts):>10,.0f}{hits:>5}/{len(ttfts)}")
        peak = max(row.get("sampled_peak_rss_mib") or 0 for row in mine)
        print(f"  peak process memory: {peak:,.0f} MiB")
    print(f"\nRows: {OUTPUT.resolve()}")


def main():
    args = get_arguments()
    models = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    rows = []
    for cache_size in args.cache_sizes:
        rows += run_cache_size(models[LABEL]["path"], cache_size, args.repeats)
    summarize(rows)


if __name__ == "__main__":
    main()