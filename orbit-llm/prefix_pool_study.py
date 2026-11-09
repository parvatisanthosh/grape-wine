"""How many conversations can a fixed KV pool keep in the prefix cache?

K conversations, each a distinct long prompt, take turns on one pipeline:

    warm pass:     c1, c2, ..., cK      (cold prefill, fills the cache)
    measured pass: c1, c2, ..., cK      (a hit if the prompt is still cached)

The pool is either the default dynamic one (0) or a fixed number of KV blocks.

Prediction from the OpenVINO source (Finding 26): a conversation of P prompt
tokens plus G generated tokens needs ceil((P + G) / block_size) blocks, with a
block size of 32 tokens on CPU and 16 on GPU. Freed blocks are reused least
recently used first, and a prefix matches only from its first block. Taking
turns is the worst case for that policy: all hits while the pool holds every
conversation, and none once it holds less.

By default each K is tested at its boundary, K * need + offset blocks for each
--offsets value, plus the dynamic pool. With --pools, every K runs at every pool.
Whether a prompt is still cached does not depend on power, but its timing does.

    python prefix_pool_study.py --device GPU --out results/experiments/prefix_pool.jsonl
"""

import argparse
import json
import math
from pathlib import Path

from orbit_worker_client import OrbitWorker
from run_experiment import prevent_sleep, require_ac_power


MODEL = "C:/Users/parva/tinyllama-int4"
BLOCK_SIZE = {"CPU": 32, "GPU": 16}
PROMPT_VARIANT = "run_1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--prompt-tokens", type=int, default=1920)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--conversations", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--pools",
        type=int,
        nargs="+",
        default=None,
        help="Pool sizes in KV blocks, 0 = dynamic; every K runs at every pool.",
    )
    parser.add_argument(
        "--offsets",
        type=int,
        nargs="+",
        default=[-1, 0, 1, 2],
        help="Without --pools: test each K at K * need + offset blocks.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--allow-battery", action="store_true")
    return parser.parse_args()


def blocks_per_conversation(prompt_tokens, output_tokens, device):
    return math.ceil((prompt_tokens + output_tokens) / BLOCK_SIZE[device])


def plan_runs(conversations, need, pools=None, offsets=(-1, 0, 1, 2)):
    """(pool, K) pairs: a full grid if pools are given, else each K's boundary."""
    if pools is not None:
        return [(pool, count) for pool in pools for count in conversations]
    runs = []
    for count in conversations:
        runs.append((0, count))
        runs += [
            (count * need + offset, count)
            for offset in offsets
            if count * need + offset > 0
        ]
    return runs


def run_one(worker, args, repeat, pool, count, need, results):
    options = {"label": "INT4", "device": args.device}
    if args.device == "GPU":
        options["cache_dir"] = "ov_cache/gpu"
    if pool:
        options["kv_blocks"] = pool

    slot = f"pool{pool}"
    reply = worker.load(slot, args.model, **options)
    if reply["status"] != "success":
        print(f"pool {pool}: load failed: {reply.get('error')}", flush=True)
        return

    ttfts = []
    for phase in ("warm", "measured"):
        for number in range(1, count + 1):
            row = worker.generate(
                slot, args.prompt_tokens, args.output_tokens, prompt_id=f"{PROMPT_VARIANT}#c{number}"
            )
            record = {
                "repeat": repeat,
                "device": args.device,
                "pool_blocks": pool,
                "conversations": count,
                "blocks_per_conversation": need,
                "phase": phase,
                "conversation": number,
                "prompt_tokens": args.prompt_tokens,
                "output_tokens": args.output_tokens,
                "status": row.get("status"),
                "error": row.get("error"),
                "ttft_ms": row.get("ttft_ms"),
                "tpot_ms": row.get("tpot_ms_per_token"),
                "rss_mib": row.get("rss_after_generation_mib"),
                "commit_mib": row.get("commit_after_generation_mib"),
                "on_ac_power": row.get("on_ac_power"),
                "battery_percent": row.get("battery_percent"),
            }
            results.write(json.dumps(record) + "\n")
            results.flush()
            if phase == "measured":
                ttfts.append(record["ttft_ms"])
    worker.unload(slot)
    shown = ", ".join(f"{value:.0f}" if value is not None else "failed" for value in ttfts)
    print(f"repeat {repeat} pool {pool:>5} K={count}: measured TTFT ms [{shown}]", flush=True)


def main():
    args = parse_args()
    require_ac_power(args.allow_battery)
    prevent_sleep()

    need = blocks_per_conversation(args.prompt_tokens, args.output_tokens, args.device)
    runs = plan_runs(args.conversations, need, args.pools, args.offsets)
    print(f"{args.device}: {need} blocks per conversation; (pool, K) runs {runs}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as results, OrbitWorker() as worker:
        for repeat in range(1, args.repeats + 1):
            for pool, count in runs:
                run_one(worker, args, repeat, pool, count, need, results)


if __name__ == "__main__":
    main()
