import csv
import statistics
from collections import defaultdict
from pathlib import Path


INPUT_CSV = Path("results/matrix_clean.csv")
SUMMARY_CSV = Path("results/matrix_summary.csv")

PRECISION_ORDER = {
    "FP16": 0,
    "INT8": 1,
    "INT4": 2,
}

METRICS = [
    "ttft_ms",
    "tpot_ms_per_token",
    "throughput_tokens_per_second",
    "openvino_generation_ms",
    "sampled_peak_rss_mib",
]


def load_successful_rows(csv_path):
    grouped_rows = defaultdict(list)

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["status"] != "success":
                continue

            key = (
                row["precision"],
                int(row["requested_input_tokens"]),
                int(row["requested_output_tokens"]),
            )

            grouped_rows[key].append(row)

    return grouped_rows


def calculate_statistics(rows, metric):
    values = [
        float(row[metric])
        for row in rows
    ]

    mean_value = statistics.mean(values)
    median_value = statistics.median(values)

    if len(values) > 1:
        stdev_value = statistics.stdev(values)
    else:
        stdev_value = 0.0

    if mean_value != 0:
        cv_percent = (
            stdev_value / abs(mean_value)
        ) * 100
    else:
        cv_percent = 0.0

    return {
        "mean": mean_value,
        "median": median_value,
        "stdev": stdev_value,
        "cv_percent": cv_percent,
        "minimum": min(values),
        "maximum": max(values),
    }


def build_summary(grouped_rows):
    summaries = {}

    for key, rows in grouped_rows.items():
        precision, input_tokens, output_tokens = key

        summary = {
            "precision": precision,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "runs": len(rows),
        }

        for metric in METRICS:
            result = calculate_statistics(
                rows,
                metric,
            )

            for statistic_name, value in result.items():
                column_name = (
                    f"{metric}_{statistic_name}"
                )

                summary[column_name] = round(
                    value,
                    4,
                )

        summaries[key] = summary

    return summaries


def add_fp16_comparisons(summaries):
    for key, summary in summaries.items():
        precision, input_tokens, output_tokens = key

        if precision == "FP16":
            summary["throughput_speedup_vs_fp16"] = 1.0
            summary["memory_reduction_vs_fp16_percent"] = 0.0
            summary["ttft_reduction_vs_fp16_percent"] = 0.0
            continue

        fp16_key = (
            "FP16",
            input_tokens,
            output_tokens,
        )

        if fp16_key not in summaries:
            summary["throughput_speedup_vs_fp16"] = ""
            summary["memory_reduction_vs_fp16_percent"] = ""
            summary["ttft_reduction_vs_fp16_percent"] = ""
            continue

        fp16_summary = summaries[fp16_key]

        fp16_throughput = fp16_summary[
            "throughput_tokens_per_second_median"
        ]

        current_throughput = summary[
            "throughput_tokens_per_second_median"
        ]

        fp16_memory = fp16_summary[
            "sampled_peak_rss_mib_median"
        ]

        current_memory = summary[
            "sampled_peak_rss_mib_median"
        ]

        fp16_ttft = fp16_summary[
            "ttft_ms_median"
        ]

        current_ttft = summary[
            "ttft_ms_median"
        ]

        summary["throughput_speedup_vs_fp16"] = round(
            current_throughput / fp16_throughput,
            4,
        )

        summary[
            "memory_reduction_vs_fp16_percent"
        ] = round(
            (
                1
                - current_memory / fp16_memory
            )
            * 100,
            4,
        )

        summary[
            "ttft_reduction_vs_fp16_percent"
        ] = round(
            (
                1
                - current_ttft / fp16_ttft
            )
            * 100,
            4,
        )


def sorted_summaries(summaries):
    return sorted(
        summaries.values(),
        key=lambda row: (
            PRECISION_ORDER[row["precision"]],
            row["input_tokens"],
            row["output_tokens"],
        ),
    )


def save_summary(rows, output_path):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def print_performance_table(rows):
    print("\nMEDIAN PERFORMANCE BY CONFIGURATION")
    print("=" * 104)

    print(
        f"{'Model':<7}"
        f"{'Input':>8}"
        f"{'Output':>9}"
        f"{'TTFT ms':>12}"
        f"{'TPOT ms':>12}"
        f"{'Tokens/s':>12}"
        f"{'Peak MiB':>12}"
        f"{'TPS CV%':>11}"
        f"{'Speedup':>11}"
    )

    print("-" * 104)

    for row in rows:
        print(
            f"{row['precision']:<7}"
            f"{row['input_tokens']:>8}"
            f"{row['output_tokens']:>9}"
            f"{row['ttft_ms_median']:>12.2f}"
            f"{row['tpot_ms_per_token_median']:>12.2f}"
            f"{row['throughput_tokens_per_second_median']:>12.2f}"
            f"{row['sampled_peak_rss_mib_median']:>12.1f}"
            f"{row['throughput_tokens_per_second_cv_percent']:>11.2f}"
            f"{row['throughput_speedup_vs_fp16']:>10.2f}x"
        )


def print_quantization_comparison(rows):
    print("\nQUANTIZATION COMPARISON AGAINST MATCHED FP16")
    print("=" * 92)

    print(
        f"{'Model':<7}"
        f"{'Input':>8}"
        f"{'Output':>9}"
        f"{'TPS speedup':>14}"
        f"{'Memory saved':>16}"
        f"{'TTFT reduced':>16}"
    )

    print("-" * 92)

    for row in rows:
        if row["precision"] == "FP16":
            continue

        print(
            f"{row['precision']:<7}"
            f"{row['input_tokens']:>8}"
            f"{row['output_tokens']:>9}"
            f"{row['throughput_speedup_vs_fp16']:>13.2f}x"
            f"{row['memory_reduction_vs_fp16_percent']:>15.1f}%"
            f"{row['ttft_reduction_vs_fp16_percent']:>15.1f}%"
        )


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}"
        )

    grouped_rows = load_successful_rows(
        INPUT_CSV
    )

    if len(grouped_rows) != 18:
        raise RuntimeError(
            f"Expected 18 configurations, "
            f"found {len(grouped_rows)}."
        )

    for key, rows in grouped_rows.items():
        if len(rows) != 3:
            raise RuntimeError(
                f"Configuration {key} contains "
                f"{len(rows)} runs instead of 3."
            )

    summaries = build_summary(grouped_rows)

    add_fp16_comparisons(summaries)

    ordered_rows = sorted_summaries(
        summaries
    )

    save_summary(
        ordered_rows,
        SUMMARY_CSV,
    )

    print_performance_table(
        ordered_rows
    )

    print_quantization_comparison(
        ordered_rows
    )

    print(
        f"\nSummary CSV saved to: "
        f"{SUMMARY_CSV.resolve()}"
    )


if __name__ == "__main__":
    main()