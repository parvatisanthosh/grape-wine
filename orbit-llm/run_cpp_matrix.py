import argparse
import csv
import json
import sys
import time
from pathlib import Path

from orbit_exec import run_configuration


MODELS = {
    "FP16": r"C:\Users\parva\tinyllama-ov",
    "INT8": r"C:\Users\parva\tinyllama-int8",
    "INT4": r"C:\Users\parva\tinyllama-int4",
}

PROMPT_LENGTHS = [32, 128, 512]
OUTPUT_LENGTHS = [32, 128]

# Column names expected by analyze_cold_matrix.py.
CSV_COLUMNS = {
    "precision": "label",
    "requested_input_tokens": "requested_input_tokens",
    "requested_output_tokens": "requested_output_tokens",
    "iteration": "iteration",
    "prompt_id": "prompt_id",
    "ttft_ms": "ttft_ms",
    "tpot_ms_per_token": "tpot_ms_per_token",
    "throughput_tokens_per_second": "throughput_tokens_per_second",
    "openvino_generation_ms": "generation_ms",
    "sampled_peak_rss_mib": "sampled_peak_rss_mib",
    "sys_available_ram_mib": "sys_available_ram_mib",
    "sys_cpu_busy_percent": "sys_cpu_busy_percent",
    "status": "status",
    "error": "error",
}


def get_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Re-run the cold-prompt workload matrix "
            "through the C++ orbit_run executor."
        )
    )

    parser.add_argument(
        "--jsonl",
        default="results/cpp_matrix_cold_raw.jsonl",
        help="Full JSON-lines output (every field).",
    )

    parser.add_argument(
        "--csv",
        default="results/cpp_matrix_cold_raw.csv",
        help="CSV output compatible with analyze_cold_matrix.py.",
    )

    parser.add_argument(
        "--precisions",
        nargs="+",
        default=list(MODELS),
        choices=list(MODELS),
    )

    return parser.parse_args()


def jsonl_to_csv(jsonl_path, csv_path):
    with Path(jsonl_path).open("r", encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]

    with Path(csv_path).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(CSV_COLUMNS),
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    column: row.get(source, "")
                    for column, source in CSV_COLUMNS.items()
                }
            )


def main():
    args = get_arguments()

    configurations = [
        (precision, prompt_tokens, output_tokens)
        for precision in args.precisions
        for prompt_tokens in PROMPT_LENGTHS
        for output_tokens in OUTPUT_LENGTHS
    ]

    print("=" * 70)
    print("ORBIT-LLM COLD MATRIX (C++ EXECUTOR)")
    print(f"Configurations: {len(configurations)}")
    print(f"Expected measured rows: {len(configurations) * 3}")
    print("=" * 70)

    failed = 0
    matrix_start = time.perf_counter()

    for index, (precision, prompt_tokens, output_tokens) in enumerate(
        configurations,
        start=1,
    ):
        print("\n" + "#" * 70)
        print(
            f"Configuration {index}/{len(configurations)}: "
            f"{precision}, {prompt_tokens} input, "
            f"{output_tokens} output"
        )
        print("#" * 70, flush=True)

        rows = run_configuration(
            MODELS[precision],
            {
                "label": precision,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "warmup": 1,
                "runs": 3,
                "prompt_mode": "cold",
            },
            jsonl_path=args.jsonl,
        )

        if any(row["status"] != "success" for row in rows):
            failed += 1

    jsonl_to_csv(args.jsonl, args.csv)

    elapsed_minutes = (time.perf_counter() - matrix_start) / 60

    print("\n" + "=" * 70)
    print("MATRIX COMPLETE")
    print(f"Configurations with failures: {failed}")
    print(f"Elapsed time: {elapsed_minutes:.2f} minutes")
    print(f"JSON lines: {Path(args.jsonl).resolve()}")
    print(f"CSV: {Path(args.csv).resolve()}")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
