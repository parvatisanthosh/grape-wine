"""Does a long unrelated prompt push an earlier prompt out of the prefix cache?

Sends prompts through one long-lived orbit_worker pipeline, in the order that
evaluate_controller.py produced for its follow-up requests:

    A, A            a repeat with nothing in between: should be a real hit
    B, A            a long unique prompt in between: the case under test
    C, A            a short unique prompt in between: the control

The same sequence runs with a dynamic KV pool and with a fixed one larger than
every prompt together. If a bigger pool brings the hit back, the blocks really
were being overwritten; otherwise something else is losing the prefix.

A hit is judged from the measured TTFT, because the evaluation's cache_hit
field only records that the same prompt had been sent before.

    python prefix_overwrite_test.py --out results/prefix_overwrite.jsonl
"""

import argparse
import ctypes
import json
from pathlib import Path

from orbit_worker_client import OrbitWorker
from run_experiment import prevent_sleep, require_ac_power


MODEL = "C:/Users/parva/tinyllama-int4"
PROMPT_A = "run_1#conversation"
PROMPT_B = "run_3#long"
PROMPT_C = "run_2#short"
LONG_TOKENS = 1920
SHORT_TOKENS = 64

# (step name, prompt id, prompt tokens)
SEQUENCE = [
    ("A_first", PROMPT_A, LONG_TOKENS),
    ("A_repeat", PROMPT_A, LONG_TOKENS),
    ("B_long_between", PROMPT_B, LONG_TOKENS),
    ("A_after_long", PROMPT_A, LONG_TOKENS),
    ("A_repeat_2", PROMPT_A, LONG_TOKENS),
    ("C_short_between", PROMPT_C, SHORT_TOKENS),
    ("A_after_short", PROMPT_A, LONG_TOKENS),
]


class PowerStatus(ctypes.Structure):
    _fields_ = [
        ("ac_line", ctypes.c_ubyte),
        ("battery_flag", ctypes.c_ubyte),
        ("battery_percent", ctypes.c_ubyte),
        ("system_status", ctypes.c_ubyte),
        ("battery_seconds", ctypes.c_ulong),
        ("full_seconds", ctypes.c_ulong),
    ]


def on_ac_power():
    status = PowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return None
    return status.ac_line == 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--pools",
        type=int,
        nargs="+",
        default=[0, 1],
        help="KV pool sizes in GB; 0 means dynamic.",
    )
    parser.add_argument("--allow-battery", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    require_ac_power(args.allow_battery)
    prevent_sleep()
    on_ac = on_ac_power()
    print(f"on AC power: {on_ac}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("a", encoding="utf-8") as results:
        for repeat in range(1, args.repeats + 1):
            for pool in args.pools:
                slot = f"pool{pool}"
                options = {"label": "INT4", "device": args.device}
                if args.device == "GPU":
                    options["cache_dir"] = "ov_cache/gpu"
                if pool:
                    options["cache_size_gb"] = pool

                with OrbitWorker() as worker:
                    reply = worker.load(slot, args.model, **options)
                    if reply["status"] != "success":
                        print(f"pool {pool}: load failed: {reply.get('error')}", flush=True)
                        continue

                    for step, prompt_id, tokens in SEQUENCE:
                        row = worker.generate(
                            slot, tokens, args.output_tokens, prompt_id=prompt_id
                        )
                        record = {
                            "repeat": repeat,
                            "pool_gb": pool,
                            "device": args.device,
                            "step": step,
                            "prompt_id": prompt_id,
                            "prompt_tokens": tokens,
                            "status": row.get("status"),
                            "ttft_ms": row.get("ttft_ms"),
                            "tpot_ms": row.get("tpot_ms_per_token"),
                            "on_ac_power": on_ac,
                            "error": row.get("error"),
                        }
                        results.write(json.dumps(record) + "\n")
                        results.flush()
                        print(
                            f"pool={pool} {step:16} {record['status']} "
                            f"TTFT {record['ttft_ms']} ms",
                            flush=True,
                        )


if __name__ == "__main__":
    main()
