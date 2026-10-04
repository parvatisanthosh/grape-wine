import csv
import statistics
from collections import defaultdict
from pathlib import Path


CSV_PATH = Path("results/benchmark_results.csv")

METRICS = {
    "ttft_ms": "TTFT (ms)",
    "tpot_ms_per_token": "TPOT (ms/token)",
    "throughput_tokens_per_second": "Throughput (tokens/s)",
    "openvino_generation_ms": "Generation time (ms)",
    "sampled_peak_rss_mib": "Peak RSS (MiB)",
}


def read_successful_rows(csv_path):
    grouped_rows = defaultdict(list)

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["status"] != "success":
                continue

            grouped_rows[row["precision"]].append(row)

    return grouped_rows


def calculate_statistics(rows, metric):
    values = [float(row[metric]) for row in rows]

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "stdev": (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        ),
    }


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    grouped_rows = read_successful_rows(CSV_PATH)

    precision_order = ["FP16", "INT8", "INT4"]
    summaries = {}

    for precision in precision_order:
        rows = grouped_rows.get(precision, [])

        if not rows:
            continue

        summaries[precision] = {}

        print(f"\n{'=' * 60}")
        print(f"{precision}: {len(rows)} successful measured runs")
        print("=" * 60)

        for metric, display_name in METRICS.items():
            result = calculate_statistics(rows, metric)
            summaries[precision][metric] = result

            print(
                f"{display_name:<26} "
                f"mean={result['mean']:.2f}, "
                f"median={result['median']:.2f}, "
                f"stdev={result['stdev']:.2f}, "
                f"range={result['minimum']:.2f}"
                f"–{result['maximum']:.2f}"
            )

    if "FP16" not in summaries:
        return

    fp16_throughput = summaries["FP16"][
        "throughput_tokens_per_second"
    ]["mean"]

    fp16_memory = summaries["FP16"][
        "sampled_peak_rss_mib"
    ]["mean"]

    print(f"\n{'=' * 60}")
    print("COMPARISON AGAINST FP16")
    print("=" * 60)

    for precision in ["INT8", "INT4"]:
        if precision not in summaries:
            continue

        current_throughput = summaries[precision][
            "throughput_tokens_per_second"
        ]["mean"]

        current_memory = summaries[precision][
            "sampled_peak_rss_mib"
        ]["mean"]

        speedup = current_throughput / fp16_throughput

        memory_reduction = (
            1 - current_memory / fp16_memory
        ) * 100

        print(
            f"{precision}: "
            f"{speedup:.2f}x throughput, "
            f"{memory_reduction:.1f}% lower peak RSS"
        )


if __name__ == "__main__":
    main()