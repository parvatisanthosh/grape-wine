import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


ORBIT_RUN = Path(__file__).parent / "cpp" / "build" / "Release" / "orbit_run.exe"

# OpenVINO core DLLs and device plugins come from the pip package.
OPENVINO_LIBS = Path(r"C:\Users\parva\venv\Lib\site-packages\openvino\libs")

# Command-line flag for each configuration key accepted by orbit_run.
OPTION_FLAGS = {
    "label": "--label",
    "device": "--device",
    "prompt_tokens": "--prompt-tokens",
    "output_tokens": "--output-tokens",
    "warmup": "--warmup",
    "runs": "--runs",
    "prompt_mode": "--prompt-mode",
    "backend": "--backend",
    "prefix_caching": "--prefix-caching",
    "max_batched_tokens": "--max-batched-tokens",
    "cache_size_gb": "--cache-size-gb",
    "kv_precision": "--kv-precision",
    "threads": "--threads",
    "cores": "--cores",
    "cache_dir": "--cache-dir",
}


def build_command(model, configuration):
    command = [str(ORBIT_RUN), "--model", str(model)]

    for key, value in configuration.items():
        if key not in OPTION_FLAGS:
            raise KeyError(f"Unknown orbit_run option: {key}")

        command.extend([OPTION_FLAGS[key], str(value)])

    return command


def run_configuration(model, configuration, jsonl_path=None):
    """Run one configuration through orbit_run.exe and return its rows.

    orbit_run writes one JSON line per measured run (or one row if loading
    fails). If the process dies without writing anything — for example when
    Windows kills it for running out of memory — a 'crashed' row is returned
    instead, so failures still end up in the experiment dataset.
    """
    if not ORBIT_RUN.exists():
        raise FileNotFoundError(
            f"orbit_run.exe not found: {ORBIT_RUN}. "
            f"Build it first: cmake --build cpp/build --config Release"
        )

    environment = os.environ.copy()
    environment["PATH"] = (
        str(OPENVINO_LIBS) + os.pathsep + environment["PATH"]
    )

    # stderr (human-readable progress) passes through to the console.
    completed = subprocess.run(
        build_command(model, configuration),
        stdout=subprocess.PIPE,
        text=True,
        env=environment,
        check=False,
    )

    rows = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.startswith("{")
    ]

    if not rows:
        rows = [
            {
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "executor": "cpp",
                "model_path": str(model),
                **configuration,
                "requested_input_tokens": configuration.get(
                    "prompt_tokens"
                ),
                "requested_output_tokens": configuration.get(
                    "output_tokens"
                ),
                "status": "crashed",
                "error": f"orbit_run exited with code "
                f"{completed.returncode} and wrote no rows",
            }
        ]

    if jsonl_path is not None:
        jsonl_path = Path(jsonl_path)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        with jsonl_path.open("a", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")

    return rows
