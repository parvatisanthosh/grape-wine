import subprocess
import sys
import time
from pathlib import Path


PYTHON = sys.executable
WORKER = Path("benchmark_matrix_worker.py")
OUTPUT_CSV = Path("results/matrix_raw.csv")

MODELS = {
    "FP16": r"C:\Users\parva\tinyllama-ov",
    "INT8": r"C:\Users\parva\tinyllama-int8",
    "INT4": r"C:\Users\parva\tinyllama-int4",
}

PROMPT_LENGTHS = [32, 128, 512]
OUTPUT_LENGTHS = [32, 128]


def main():
    configurations = []

    for precision, model_path in MODELS.items():
        for prompt_tokens in PROMPT_LENGTHS:
            for output_tokens in OUTPUT_LENGTHS:
                configurations.append(
                    {
                        "precision": precision,
                        "model": model_path,
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                    }
                )

    print("=" * 70)
    print("ORBIT-LLM WORKLOAD MATRIX")
    print(f"Python: {PYTHON}")
    print(f"Configurations: {len(configurations)}")
    print("Measured runs per configuration: 3")
    print(
        f"Expected measured rows: "
        f"{len(configurations) * 3}"
    )
    print("=" * 70)

    completed = 0
    failed = 0
    matrix_start = time.perf_counter()

    for index, configuration in enumerate(
        configurations,
        start=1,
    ):
        precision = configuration["precision"]
        prompt_tokens = configuration["prompt_tokens"]
        output_tokens = configuration["output_tokens"]

        print("\n" + "#" * 70)
        print(
            f"Configuration {index}/{len(configurations)}: "
            f"{precision}, "
            f"{prompt_tokens} input tokens, "
            f"{output_tokens} output tokens"
        )
        print("#" * 70)

        command = [
            PYTHON,
            str(WORKER),
            "--precision",
            precision,
            "--model",
            configuration["model"],
            "--prompt-tokens",
            str(prompt_tokens),
            "--output-tokens",
            str(output_tokens),
            "--output",
            str(OUTPUT_CSV),
        ]

        result = subprocess.run(
            command,
            check=False,
        )

        if result.returncode == 0:
            completed += 1
        else:
            failed += 1

            print(
                f"Configuration failed with exit code "
                f"{result.returncode}."
            )

    elapsed_minutes = (
        time.perf_counter() - matrix_start
    ) / 60

    print("\n" + "=" * 70)
    print("MATRIX COMPLETE")
    print(f"Successful configurations: {completed}")
    print(f"Failed configurations: {failed}")
    print(f"Elapsed time: {elapsed_minutes:.2f} minutes")
    print(f"Raw results: {OUTPUT_CSV.resolve()}")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()