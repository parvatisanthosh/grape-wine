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
WARMUP_RUNS = 1
MEASURED_RUNS = 3
MEMORY_SAMPLE_INTERVAL_SECONDS = 0.01

BASE_TEXT = (
    "The Roman Empire developed complex systems of government, trade, "
    "military organization, engineering, law, architecture, agriculture, "
    "religion, communication, and culture. Its history demonstrates how "
    "political institutions, economic systems, and military power influence "
    "the growth and decline of a civilization. "
)

CSV_FIELDS = [
    "timestamp",
    "precision",
    "model_path",
    "device",
    "iteration",
    "requested_input_tokens",
    "actual_input_tokens",
    "requested_output_tokens",
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
        description="Run one controlled OpenVINO GenAI benchmark configuration."
    )

    parser.add_argument(
        "--precision",
        required=True,
        choices=["FP16", "INT8", "INT4"],
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the OpenVINO model directory.",
    )

    parser.add_argument(
        "--prompt-tokens",
        required=True,
        type=int,
        help="Exact number of input tokens.",
    )

    parser.add_argument(
        "--output-tokens",
        required=True,
        type=int,
        help="Exact number of generated tokens.",
    )

    parser.add_argument(
        "--output",
        default="results/matrix_raw.csv",
        help="Output CSV file.",
    )

    return parser.parse_args()


def get_rss_mib(process):
    return process.memory_info().rss / (1024 * 1024)


def monitor_peak_memory(process, stop_event, peak_result):
    peak_result[0] = get_rss_mib(process)

    while not stop_event.wait(MEMORY_SAMPLE_INTERVAL_SECONDS):
        current_rss = get_rss_mib(process)
        peak_result[0] = max(peak_result[0], current_rss)


def append_csv_row(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def create_exact_token_input(tokenizer, target_tokens):
    if target_tokens < 2:
        raise ValueError("Prompt length must be at least 2 tokens.")

    repeated_text = BASE_TEXT * (
        target_tokens // 20 + 10
    )

    tokenized_input = tokenizer.encode(
        repeated_text,
        add_special_tokens=True,
        max_length=target_tokens,
        truncation=True,
    )

    actual_tokens = len(
        tokenized_input.input_ids.data[0]
    )

    if actual_tokens != target_tokens:
        raise RuntimeError(
            f"Requested {target_tokens} prompt tokens, "
            f"but tokenizer produced {actual_tokens}."
        )

    return tokenized_input


def main():
    args = get_arguments()

    model_path = Path(args.model)
    csv_path = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}"
        )

    process = psutil.Process(os.getpid())
    rss_before_load = get_rss_mib(process)

    print("=" * 60)
    print(f"Precision: {args.precision}")
    print(f"Model: {model_path}")
    print(f"Device: {DEVICE}")
    print(f"Requested input tokens: {args.prompt_tokens}")
    print(f"Requested output tokens: {args.output_tokens}")
    print(f"Measured runs: {MEASURED_RUNS}")
    print("=" * 60)

    load_start = time.perf_counter()

    pipeline = ov_genai.LLMPipeline(
        str(model_path),
        DEVICE,
    )

    pipeline_construction_ms = (
        time.perf_counter() - load_start
    ) * 1000

    rss_after_load = get_rss_mib(process)

    tokenizer = pipeline.get_tokenizer()

    tokenized_input = create_exact_token_input(
        tokenizer,
        args.prompt_tokens,
    )

    verified_input_tokens = len(
        tokenized_input.input_ids.data[0]
    )

    print(f"Verified input tokens: {verified_input_tokens}")
    print(f"RSS before loading: {rss_before_load:.1f} MiB")
    print(f"RSS after loading: {rss_after_load:.1f} MiB")
    print(
        f"Pipeline construction: "
        f"{pipeline_construction_ms:.2f} ms"
    )

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.output_tokens
    config.ignore_eos = True
    config.do_sample = False
    config.apply_chat_template = False

    print("\nRunning warm-up...")

    for _ in range(WARMUP_RUNS):
        pipeline.generate(tokenized_input, config)

    rss_after_warmup = get_rss_mib(process)

    print(
        f"RSS after warm-up: "
        f"{rss_after_warmup:.1f} MiB"
    )

    for iteration in range(1, MEASURED_RUNS + 1):
        print(f"\nMeasured iteration {iteration}...")

        stop_event = threading.Event()
        peak_result = [get_rss_mib(process)]

        memory_thread = threading.Thread(
            target=monitor_peak_memory,
            args=(process, stop_event, peak_result),
            daemon=True,
        )

        row = {
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "precision": args.precision,
            "model_path": str(model_path),
            "device": DEVICE,
            "iteration": iteration,
            "requested_input_tokens": args.prompt_tokens,
            "actual_input_tokens": "",
            "requested_output_tokens": args.output_tokens,
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
            "rss_after_warmup_mib": round(
                rss_after_warmup, 1
            ),
            "sampled_peak_rss_mib": "",
            "rss_after_generation_mib": "",
            "status": "failed",
            "error": "",
        }

        try:
            memory_thread.start()

            wall_start = time.perf_counter()

            result = pipeline.generate(
                tokenized_input,
                config,
            )

            wall_time_ms = (
                time.perf_counter() - wall_start
            ) * 1000

            metrics = result.perf_metrics

            input_tokens = metrics.get_num_input_tokens()
            output_tokens = metrics.get_num_generated_tokens()

            generation_duration = (
                metrics.get_generate_duration()
            )
            ttft = metrics.get_ttft()
            tpot = metrics.get_tpot()
            throughput = metrics.get_throughput()

            if input_tokens != args.prompt_tokens:
                raise RuntimeError(
                    f"Expected {args.prompt_tokens} input tokens, "
                    f"but OpenVINO reported {input_tokens}."
                )

            if output_tokens != args.output_tokens:
                raise RuntimeError(
                    f"Expected {args.output_tokens} output tokens, "
                    f"but OpenVINO reported {output_tokens}."
                )

            row.update(
                {
                    "actual_input_tokens": input_tokens,
                    "actual_output_tokens": output_tokens,
                    "wall_time_ms": round(
                        wall_time_ms, 2
                    ),
                    "openvino_generation_ms": round(
                        generation_duration.mean, 2
                    ),
                    "ttft_ms": round(
                        ttft.mean, 2
                    ),
                    "tpot_ms_per_token": round(
                        tpot.mean, 2
                    ),
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
                f"Tokens: "
                f"{row['actual_input_tokens']} input / "
                f"{row['actual_output_tokens']} output"
            )
            print(f"TTFT: {row['ttft_ms']:.2f} ms")
            print(
                f"TPOT: "
                f"{row['tpot_ms_per_token']:.2f} ms/token"
            )
            print(
                f"Throughput: "
                f"{row['throughput_tokens_per_second']:.2f} tok/s"
            )
            print(
                f"Peak RSS: "
                f"{row['sampled_peak_rss_mib']:.1f} MiB"
            )
        else:
            print(f"FAILED: {row['error']}")

    print(f"\nResults saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()