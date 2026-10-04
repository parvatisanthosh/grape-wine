import argparse
import csv
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import openvino_genai as ov_genai
import psutil


DEVICE = "CPU"
MAX_NEW_TOKENS = 50
WARMUP_RUNS = 1
MEASURED_RUNS = 3
MEMORY_SAMPLE_INTERVAL_SECONDS = 0.01

PROMPT = (
    "Write a detailed explanation of the history of the Roman Empire, "
    "including its founding, expansion, government, culture, and fall."
)

CSV_FIELDS = [
    "timestamp",
    "precision",
    "model_path",
    "device",
    "iteration",
    "requested_output_tokens",
    "actual_input_tokens",
    "actual_output_tokens",
    "pipeline_construction_ms",
    "wall_time_ms",
    "openvino_generation_ms",
    "ttft_ms",
    "tpot_ms_per_token",
    "throughput_tokens_per_second",
    "rss_before_load_mib",
    "rss_after_load_mib",
    "rss_after_warmup_mib",
    "sampled_peak_rss_mib",
    "rss_after_generation_mib",
    "status",
    "error",
]


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Benchmark one OpenVINO GenAI model."
    )

    parser.add_argument(
        "--precision",
        required=True,
        choices=["FP16", "INT8", "INT4"],
        help="Precision label stored in the CSV file.",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the OpenVINO model directory.",
    )

    parser.add_argument(
        "--output",
        default="results/benchmark_results.csv",
        help="CSV output path.",
    )

    return parser.parse_args()


def get_rss_mib(process):
    return process.memory_info().rss / (1024 * 1024)


def monitor_peak_memory(process, stop_event, peak_result):
    peak_result[0] = get_rss_mib(process)

    while not stop_event.wait(MEMORY_SAMPLE_INTERVAL_SECONDS):
        current_memory = get_rss_mib(process)

        if current_memory > peak_result[0]:
            peak_result[0] = current_memory


def append_csv_row(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_already_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)

        if not file_already_exists:
            writer.writeheader()

        writer.writerow(row)


def main():
    args = get_arguments()

    model_path = Path(args.model)
    csv_path = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    process = psutil.Process(os.getpid())

    print("=" * 60)
    print(f"Precision: {args.precision}")
    print(f"Model: {model_path}")
    print(f"Device: {DEVICE}")
    print(f"Warm-up runs: {WARMUP_RUNS}")
    print(f"Measured runs: {MEASURED_RUNS}")
    print(f"Requested output tokens: {MAX_NEW_TOKENS}")
    print("=" * 60)

    rss_before_load = get_rss_mib(process)

    print(f"RSS before model load: {rss_before_load:.1f} MiB")

    load_start = time.perf_counter()

    pipeline = ov_genai.LLMPipeline(
        str(model_path),
        DEVICE,
    )

    pipeline_construction_ms = (
        time.perf_counter() - load_start
    ) * 1000

    rss_after_load = get_rss_mib(process)

    print(f"RSS after model load: {rss_after_load:.1f} MiB")
    print(
        f"Pipeline construction time: "
        f"{pipeline_construction_ms:.2f} ms"
    )

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = MAX_NEW_TOKENS
    config.ignore_eos = True
    config.do_sample = False
    config.apply_chat_template = False

    prompt_batch = [PROMPT]

    print("\nRunning warm-up...")

    for _ in range(WARMUP_RUNS):
        pipeline.generate(prompt_batch, config)

    rss_after_warmup = get_rss_mib(process)

    print(f"RSS after warm-up: {rss_after_warmup:.1f} MiB")

    for iteration in range(1, MEASURED_RUNS + 1):
        print(f"\nRunning measured iteration {iteration}...")

        stop_event = threading.Event()
        peak_result = [get_rss_mib(process)]

        memory_thread = threading.Thread(
            target=monitor_peak_memory,
            args=(process, stop_event, peak_result),
            daemon=True,
        )

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "precision": args.precision,
            "model_path": str(model_path),
            "device": DEVICE,
            "iteration": iteration,
            "requested_output_tokens": MAX_NEW_TOKENS,
            "actual_input_tokens": "",
            "actual_output_tokens": "",
            "pipeline_construction_ms": round(
                pipeline_construction_ms, 2
            ),
            "wall_time_ms": "",
            "openvino_generation_ms": "",
            "ttft_ms": "",
            "tpot_ms_per_token": "",
            "throughput_tokens_per_second": "",
            "rss_before_load_mib": round(rss_before_load, 1),
            "rss_after_load_mib": round(rss_after_load, 1),
            "rss_after_warmup_mib": round(rss_after_warmup, 1),
            "sampled_peak_rss_mib": "",
            "rss_after_generation_mib": "",
            "status": "failed",
            "error": "",
        }

        try:
            memory_thread.start()

            wall_start = time.perf_counter()
            result = pipeline.generate(prompt_batch, config)
            wall_time_ms = (
                time.perf_counter() - wall_start
            ) * 1000

            metrics = result.perf_metrics

            input_tokens = metrics.get_num_input_tokens()
            output_tokens = metrics.get_num_generated_tokens()
            generation_duration = metrics.get_generate_duration()
            ttft = metrics.get_ttft()
            tpot = metrics.get_tpot()
            throughput = metrics.get_throughput()

            row.update(
                {
                    "actual_input_tokens": input_tokens,
                    "actual_output_tokens": output_tokens,
                    "wall_time_ms": round(wall_time_ms, 2),
                    "openvino_generation_ms": round(
                        generation_duration.mean, 2
                    ),
                    "ttft_ms": round(ttft.mean, 2),
                    "tpot_ms_per_token": round(tpot.mean, 2),
                    "throughput_tokens_per_second": round(
                        throughput.mean, 2
                    ),
                    "status": "success",
                }
            )

        except Exception as exception:
            row["error"] = (
                f"{type(exception).__name__}: {exception}"
            )

        finally:
            stop_event.set()

            if memory_thread.is_alive():
                memory_thread.join()

            row["sampled_peak_rss_mib"] = round(
                peak_result[0], 1
            )

            row["rss_after_generation_mib"] = round(
                get_rss_mib(process), 1
            )

            append_csv_row(csv_path, row)

        if row["status"] == "success":
            print(
                f"Input/output tokens: "
                f"{row['actual_input_tokens']}/"
                f"{row['actual_output_tokens']}"
            )
            print(f"TTFT: {row['ttft_ms']:.2f} ms")
            print(
                f"TPOT: "
                f"{row['tpot_ms_per_token']:.2f} ms/token"
            )
            print(
                f"Throughput: "
                f"{row['throughput_tokens_per_second']:.2f} tokens/s"
            )
            print(
                f"Generation time: "
                f"{row['openvino_generation_ms']:.2f} ms"
            )
            print(
                f"Sampled peak RSS: "
                f"{row['sampled_peak_rss_mib']:.1f} MiB"
            )
        else:
            print(f"Iteration failed: {row['error']}")

    print(f"\nResults appended to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()