import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = {
    "ttft_ms": "TTFT ms",
    "tpot_ms_per_token": "TPOT ms",
    "throughput_tokens_per_second": "Tok/s",
    "sampled_peak_rss_mib": "Peak MiB",
}

MACHINE_STATE = [
    "sys_available_ram_mib",
    "sys_cpu_busy_percent",
]


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Summarize an ORBIT-LLM experiment JSONL file."
    )

    parser.add_argument(
        "results",
        type=Path,
        help="Experiment JSONL written by run_experiment.py.",
    )

    parser.add_argument(
        "--summary",
        type=Path,
        help="Summary CSV path (default: <results>_summary.csv).",
    )

    return parser.parse_args()


def load_rows(jsonl_path):
    with jsonl_path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def parse_config_id(config_id):
    return dict(
        part.split("=", 1)
        for part in config_id.split("|")
    )


def summarize(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["config_id"]].append(row)

    summaries = []

    for config_id, config_rows in grouped.items():
        successful = [
            row
            for row in config_rows
            if row["status"] == "success"
        ]

        summary = {
            **parse_config_id(config_id),
            "runs": len(successful),
            "failures": len(config_rows) - len(successful),
            "failure_statuses": ";".join(
                sorted(
                    {
                        row["status"]
                        for row in config_rows
                        if row["status"] != "success"
                    }
                )
            ),
        }

        for metric in list(METRICS) + MACHINE_STATE:
            values = [
                float(row[metric])
                for row in successful
                if row.get(metric) is not None
            ]

            if not values:
                summary[f"{metric}_median"] = ""
                summary[f"{metric}_min"] = ""
                summary[f"{metric}_max"] = ""
                continue

            summary[f"{metric}_median"] = round(
                statistics.median(values),
                2,
            )
            summary[f"{metric}_min"] = round(min(values), 2)
            summary[f"{metric}_max"] = round(max(values), 2)

        summaries.append(summary)

    return summaries


def sort_key(summary, config_keys):
    # Numbers sort numerically (32 < 128 < 512); text sorts alphabetically.
    key = []

    for name in config_keys:
        value = summary[name]

        try:
            key.append((0, float(value), ""))
        except ValueError:
            key.append((1, 0.0, value))

    return key


def print_table(summaries, config_keys):
    key_widths = {
        key: max(
            len(key),
            *(len(str(summary[key])) for summary in summaries),
        )
        for key in config_keys
    }

    header = "  ".join(
        f"{key:<{key_widths[key]}}"
        for key in config_keys
    )

    header += f"{'n':>4}"

    for display_name in METRICS.values():
        header += f"{display_name:>22}"

    print(header)
    print("-" * len(header))

    for summary in summaries:
        line = "  ".join(
            f"{str(summary[key]):<{key_widths[key]}}"
            for key in config_keys
        )

        line += f"{summary['runs']:>4}"

        for metric in METRICS:
            median = summary[f"{metric}_median"]

            if median == "":
                cell = summary["failure_statuses"] or "-"
            else:
                cell = (
                    f"{median:.1f} "
                    f"[{summary[f'{metric}_min']:.0f}-"
                    f"{summary[f'{metric}_max']:.0f}]"
                )

            line += f"{cell:>22}"

        print(line)


def main():
    args = get_arguments()

    rows = load_rows(args.results)

    if not rows:
        raise RuntimeError(f"No rows in {args.results}")

    config_keys = list(
        parse_config_id(rows[0]["config_id"])
    )
    summaries = sorted(
        summarize(rows),
        key=lambda summary: sort_key(summary, config_keys),
    )

    print(f"\nEXPERIMENT: {rows[0].get('experiment', '?')}")
    print("Median [min-max] over successful measured runs\n")

    print_table(summaries, config_keys)

    failures = sum(summary["failures"] for summary in summaries)

    if failures:
        print(f"\nFailed or crashed rows: {failures}")

    summary_path = args.summary or args.results.with_name(
        args.results.stem + "_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summaries[0]),
        )

        writer.writeheader()
        writer.writerows(summaries)

    print(f"\nSummary CSV saved to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
