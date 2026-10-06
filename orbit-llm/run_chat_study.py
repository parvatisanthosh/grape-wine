"""KV-cache management study through the real chat application.

Plays a scripted multi-turn conversation through cpp/orbit_chat.exe once per
configuration and collects per-turn metrics. The default study compares no
eviction with two OpenVINO cache-eviction budgets on a long conversation
whose last turn asks for facts stated in the first turn (a recall probe).

    python run_chat_study.py
    python run_chat_study.py --models Q3B-INT4 --device GPU
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from orbit_exec import OPENVINO_LIBS
from run_experiment import prevent_sleep, require_ac_power


ORBIT_CHAT = Path("cpp/build/Release/orbit_chat.exe")
MODELS_JSON = Path("models.json")

# Eviction settings in tokens: START:RECENT:MAX, multiples of the KV block
# size (32 on CPU, 16 on GPU). Empty = no eviction.
EVICTION = {
    "none": "",
    "evict_512": "32:128:512",
    "evict_256": "32:64:256",
}

RECALL_KEYWORDS = ["ORBIT", "47"]


def get_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--script", default="conversations/long_recall.txt")
    parser.add_argument("--models", nargs="+", default=["INT4"])
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    parser.add_argument("--eviction", nargs="+", default=list(EVICTION), choices=list(EVICTION))
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--output", default="results/chat_eviction.jsonl")
    parser.add_argument("--allow-battery", action="store_true")
    return parser.parse_args()


def run_conversation(model_path, args, eviction):
    output = Path(args.output)
    temporary = output.with_suffix(".tmp.jsonl")
    temporary.unlink(missing_ok=True)

    command = [
        str(ORBIT_CHAT),
        "--model", model_path,
        "--device", args.device,
        "--script", args.script,
        "--max-new-tokens", str(args.max_new_tokens),
        "--repetition-penalty", "1.1",
        "--json-out", str(temporary),
    ]
    if EVICTION[eviction]:
        command += ["--evict", EVICTION[eviction]]
    if args.device == "GPU":
        command += ["--cache-dir", "ov_cache/gpu"]

    environment = os.environ.copy()
    environment["PATH"] = str(OPENVINO_LIBS) + os.pathsep + environment["PATH"]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        check=False,
    )

    if not temporary.exists():
        print(completed.stdout[-2000:])
        print(completed.stderr[-2000:])
        return []

    with temporary.open(encoding="utf-8") as file:
        turns = [json.loads(line) for line in file if line.strip()]
    temporary.unlink()
    return turns


def main():
    args = get_arguments()
    require_ac_power(args.allow_battery)
    models = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    prevent_sleep()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for label in args.models:
        for eviction in args.eviction:
            print(f"\n=== {label} on {args.device}, eviction {eviction} ===", flush=True)
            turns = run_conversation(models[label]["path"], args, eviction)

            if not turns:
                print("Run produced no turns (see output above).")
                continue

            last = turns[-1]["answer"]
            recalled = all(keyword.lower() in last.lower() for keyword in RECALL_KEYWORDS)

            with output.open("a", encoding="utf-8") as file:
                for turn in turns:
                    turn.update(
                        {
                            "label": label,
                            "eviction": eviction,
                            "script": args.script,
                            "recall_correct": recalled if turn is turns[-1] else None,
                        }
                    )
                    file.write(json.dumps(turn) + "\n")

            print(f"Turns: {len(turns)}   last answer: {last[:160]!r}")
            print(f"Recall correct: {recalled}")

    summarize(output)


def summarize(path):
    frame = pd.read_json(path, lines=True)
    final = frame.groupby(["label", "device", "eviction"]).agg(
        turns=("turn", "max"),
        final_prompt_tokens=("input_tokens", "last"),
        mean_ttft_ms=("ttft_ms", "mean"),
        last_ttft_ms=("ttft_ms", "last"),
        mean_tpot_ms=("tpot_ms_per_token", "mean"),
        last_tpot_ms=("tpot_ms_per_token", "last"),
        peak_rss_mib=("rss_mib", "max"),
        peak_commit_mib=("commit_mib", "max"),
        recall=("recall_correct", "last"),
    )
    print("\n" + final.round(1).to_string())


if __name__ == "__main__":
    main()
