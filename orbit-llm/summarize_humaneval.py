"""Summarize the HumanEval results scored on Colab. Runs no model-written code.

    python summarize_humaneval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "colab"))

from quality_core import (  # noqa: E402
    load_jsonl,
    mcnemar_exact,
    paired_difference_interval,
    wilson_interval,
)


RESULTS = Path("results/quality_humaneval")
TAGS = ["fp16", "int8", "int4"]
COMPARISONS = [("fp16", "int8"), ("fp16", "int4"), ("int8", "int4")]


def load_runs(folder=RESULTS):
    return {
        tag: {row["task_id"]: row for row in load_jsonl(folder / f"outcomes_{tag}.jsonl")}
        for tag in TAGS
    }


def summarize_model(rows):
    passed = sum(row["outcome"] == "passed" for row in rows.values())
    low, high = wilson_interval(passed, len(rows))
    return {
        "passed": passed,
        "total": len(rows),
        "pass_rate": passed / len(rows),
        "ci_low": low,
        "ci_high": high,
        "hit_token_limit": sum(row["tokens"] >= 512 for row in rows.values()),
    }


def compare(first, second):
    """Paired comparison on the same problems."""
    tasks = sorted(first.keys() & second.keys())
    only_first = sum(
        first[t]["outcome"] == "passed" and second[t]["outcome"] != "passed" for t in tasks
    )
    only_second = sum(
        second[t]["outcome"] == "passed" and first[t]["outcome"] != "passed" for t in tasks
    )
    low, high = paired_difference_interval(only_first, only_second, len(tasks))
    return {
        "problems": len(tasks),
        "only_first_passed": only_first,
        "only_second_passed": only_second,
        "difference_ci_low": low,
        "difference_ci_high": high,
        "mcnemar_p": mcnemar_exact(only_first, only_second),
        "identical_answers": sum(first[t]["answer"] == second[t]["answer"] for t in tasks),
    }


def main():
    runs = load_runs()
    summary = {
        "models": {tag: summarize_model(runs[tag]) for tag in TAGS},
        "comparisons": {
            f"{a}_vs_{b}": compare(runs[a], runs[b]) for a, b in COMPARISONS
        },
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for tag, row in summary["models"].items():
        print(
            f"{tag}: {row['passed']}/{row['total']} = {row['pass_rate']:.1%} "
            f"(95% CI {row['ci_low']:.1%} to {row['ci_high']:.1%})"
        )
    for name, row in summary["comparisons"].items():
        print(
            f"{name}: splits {row['only_first_passed']}/{row['only_second_passed']}, "
            f"difference CI {row['difference_ci_low']:+.1%} to {row['difference_ci_high']:+.1%}, "
            f"p = {row['mcnemar_p']:.3f}, identical answers {row['identical_answers']}"
        )


if __name__ == "__main__":
    main()
