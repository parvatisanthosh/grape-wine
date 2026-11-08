"""Generate one model's HumanEval answers. Resumes after a disconnect.

Runs as its own process, so a model too large for this machine ends only this
process. Exit code -9 means the system ran out of memory.

    python run_generation.py --model-dir MODEL --tag int4 --problems HumanEval.jsonl.gz --out answers_int4.jsonl
"""

import argparse
import gzip
import json
import os
import platform
import time
from pathlib import Path

import openvino_genai as ov_genai

from quality_core import INSTRUCTION, end_with_newline, load_jsonl


MAX_NEW_TOKENS = 512


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many problems.")
    return parser.parse_args()


def main():
    args = parse_args()
    with gzip.open(args.problems, "rt", encoding="utf-8") as file:
        problems = [json.loads(line) for line in file]
    if args.limit:
        problems = problems[: args.limit]

    done = {row["task_id"] for row in load_jsonl(args.out)}
    remaining = [problem for problem in problems if problem["task_id"] not in done]
    print(f"[{args.tag}] {len(done)} done, {len(remaining)} to do", flush=True)
    if not remaining:
        return

    # Every model runs with the same attention backend and execution precision,
    # so only the weights differ between them.
    started = time.perf_counter()
    pipeline = ov_genai.LLMPipeline(
        args.model_dir,
        "CPU",
        ATTENTION_BACKEND="PA",
        INFERENCE_PRECISION_HINT="f32",
    )
    print(f"[{args.tag}] loaded in {time.perf_counter() - started:.0f} s", flush=True)

    config = pipeline.get_generation_config()
    config.do_sample = False
    config.repetition_penalty = 1.0
    config.max_new_tokens = MAX_NEW_TOKENS

    args.out.parent.mkdir(parents=True, exist_ok=True)
    end_with_newline(args.out)
    with args.out.open("a", encoding="utf-8") as results:
        for number, problem in enumerate(remaining, 1):
            history = ov_genai.ChatHistory()
            history.append(
                {"role": "user", "content": INSTRUCTION.format(prompt=problem["prompt"])}
            )

            started = time.perf_counter()
            output = pipeline.generate(history, config)
            seconds = time.perf_counter() - started
            tokens = output.perf_metrics.get_num_generated_tokens()

            results.write(
                json.dumps(
                    {
                        "task_id": problem["task_id"],
                        "tag": args.tag,
                        "answer": output.texts[0],
                        "tokens": tokens,
                        "seconds": round(seconds, 2),
                        "machine": platform.processor() or platform.machine(),
                    }
                )
                + "\n"
            )
            results.flush()
            os.fsync(results.fileno())  # survive a Colab disconnect

            left_hours = seconds * (len(remaining) - number) / 3600
            print(
                f"[{args.tag}] {number}/{len(remaining)} {problem['task_id']}: "
                f"{tokens} tokens, {tokens / seconds:.1f} tok/s, about {left_hours:.1f} h left",
                flush=True,
            )


if __name__ == "__main__":
    main()
