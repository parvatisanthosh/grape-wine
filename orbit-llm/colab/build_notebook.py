"""Build humaneval_quality.ipynb from quality_core.py and run_generation.py.

Run from orbit-llm/colab/:  python build_notebook.py
"""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).parent
PROBLEMS = HERE.parent / "data" / "raw" / "humaneval" / "HumanEval.jsonl.gz"
PROBLEMS_SHA256 = hashlib.sha256(PROBLEMS.read_bytes()).hexdigest()


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


INTRO = """
# HumanEval quality reference for Qwen2.5-Coder-3B

Part of ORBIT-LLM (task SWFW_5S_13). This measures **how much answer quality each weight
precision costs**. It does not measure speed: every timing number in the project comes
from the laptop.

All models run on this one machine with the same settings, so only the weights differ.

**How to use it**
1. Runtime > Change runtime type > **CPU**. OpenVINO cannot use Colab's NVIDIA GPUs.
2. Run all cells in order, then keep this tab open. It may take many hours.
3. If the session stops, open the notebook again and run all cells. It resumes from Drive.
4. When every model is finished or skipped, the last cell downloads the results.
5. Give the downloaded `humaneval_results.zip` to Claude.
"""

SETTINGS = f"""
# Models run in this order. FP16 is last because it may not fit in memory.
MODELS = {{
    "int4": "OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov",
    "int8": "OpenVINO/Qwen2.5-Coder-3B-Instruct-int8-ov",
    "fp16": "OpenVINO/Qwen2.5-Coder-3B-Instruct-fp16-ov",
}}
PROBLEMS_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
PROBLEMS_SHA256 = "{PROBLEMS_SHA256}"
DRIVE_FOLDER = "/content/drive/MyDrive/orbit_quality"
"""

INSTALL = """
!pip -q install "openvino==2026.3.0" "openvino-genai==2026.3.0.0" "openvino-tokenizers==2026.3.0.0"

import platform, shutil, subprocess
import openvino, openvino_genai
print("OpenVINO", openvino.__version__, "| GenAI", openvino_genai.__version__)
print(subprocess.run("lscpu | grep -E 'Model name|^CPU\\\\(s\\\\)'; free -g | head -2", shell=True, capture_output=True, text=True).stdout)
print("Free disk:", shutil.disk_usage("/content").free // 1024**3, "GB")
"""

DRIVE = """
from google.colab import drive
from pathlib import Path
drive.mount("/content/drive")
Path(DRIVE_FOLDER).mkdir(parents=True, exist_ok=True)
print("Results folder:", DRIVE_FOLDER)
"""

PROBLEMS_CELL = """
import hashlib, urllib.request
urllib.request.urlretrieve(PROBLEMS_URL, "HumanEval.jsonl.gz")
digest = hashlib.sha256(Path("HumanEval.jsonl.gz").read_bytes()).hexdigest()
assert digest == PROBLEMS_SHA256, "HumanEval file differs from the one checked on the laptop"
print("HumanEval verified")
"""

GENERATE = """
import json, shutil, subprocess, sys
from huggingface_hub import snapshot_download
import quality_core as qc

status = {}
for tag, repository in MODELS.items():
    answers = f"{DRIVE_FOLDER}/answers_{tag}.jsonl"
    model_dir = f"/content/models/{tag}"
    if len(qc.load_jsonl(answers)) >= 164:
        status[tag] = "finished"
        print(f"--- {tag}: already finished", flush=True)
        continue
    print(f"--- {tag}: downloading {repository}", flush=True)
    snapshot_download(repository, local_dir=model_dir)
    # Pipe the output here: a child process's prints do not reach the notebook.
    process = subprocess.Popen(
        [sys.executable, "-u", "run_generation.py", "--model-dir", model_dir, "--tag", tag,
         "--problems", "HumanEval.jsonl.gz", "--out", answers],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    status[tag] = "finished" if code == 0 else ("out of memory" if code == -9 else f"failed ({code})")
    print(f"--- {tag}: {status[tag]}", flush=True)
    shutil.rmtree(model_dir, ignore_errors=True)

Path(f"{DRIVE_FOLDER}/status.json").write_text(json.dumps(status))
print(status)
"""

SCORE = """
# Copy the answers, then unmount Drive BEFORE any model-written code runs.
import gzip, json, shutil, zipfile
from collections import Counter
from google.colab import drive, files
import quality_core as qc

Path("scored").mkdir(exist_ok=True)
for path in Path(DRIVE_FOLDER).glob("answers_*.jsonl"):
    shutil.copy(path, "scored")
drive.flush_and_unmount()
print("Drive unmounted")

with gzip.open("HumanEval.jsonl.gz", "rt", encoding="utf-8") as file:
    problems = {row["task_id"]: row for row in map(json.loads, file)}

# The harness must pass the official solutions before its scores mean anything.
official = Counter(
    qc.run_program(
        f"{p['prompt']}{p['canonical_solution']}\\n\\n{p['test']}\\n\\ncheck({p['entry_point']})\\n"
    )["outcome"]
    for p in problems.values()
)
print("Official solutions:", dict(official))
assert official["passed"] == len(problems), "Scoring harness is broken: do not trust the results"

scores = {}
for path in sorted(Path("scored").glob("answers_*.jsonl")):
    tag = path.stem.removeprefix("answers_")
    rows = qc.load_jsonl(path)
    outcomes = {}
    with open(f"scored/outcomes_{tag}.jsonl", "w", encoding="utf-8") as file:
        for row in rows:
            result = qc.run_program(qc.build_program(problems[row["task_id"]], row["answer"]))
            outcomes[row["task_id"]] = result["outcome"] == "passed"
            file.write(json.dumps({**row, **result}) + "\\n")
    scores[tag] = outcomes
    passed, total = sum(outcomes.values()), len(outcomes)
    low, high = qc.wilson_interval(passed, total)
    print(f"{tag}: {passed}/{total} passed = {passed / total:.1%} (95% CI {low:.1%} to {high:.1%})")

reference = "fp16" if len(scores.get("fp16", {})) == len(problems) else "int8"
for tag, outcomes in scores.items():
    if tag == reference or reference not in scores:
        continue
    shared = outcomes.keys() & scores[reference].keys()
    only_reference = sum(scores[reference][t] and not outcomes[t] for t in shared)
    only_this = sum(outcomes[t] and not scores[reference][t] for t in shared)
    print(f"{reference} vs {tag}: {only_reference} only {reference}, {only_this} only {tag}, "
          f"McNemar p = {qc.mcnemar_exact(only_reference, only_this):.3f}")

with zipfile.ZipFile("humaneval_results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for path in Path("scored").iterdir():
        archive.write(path, path.name)
files.download("humaneval_results.zip")
"""


def main():
    core = (HERE / "quality_core.py").read_text(encoding="utf-8")
    generation = (HERE / "run_generation.py").read_text(encoding="utf-8")

    notebook = {
        "cells": [
            markdown(INTRO),
            code(SETTINGS),
            code(INSTALL),
            code(DRIVE),
            code("%%writefile quality_core.py\n" + core),
            code("%%writefile run_generation.py\n" + generation),
            code(PROBLEMS_CELL),
            code(GENERATE),
            code(SCORE),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "colab": {"name": "humaneval_quality.ipynb"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (HERE / "humaneval_quality.ipynb").write_text(
        json.dumps(notebook, indent=1), encoding="utf-8"
    )
    print("Wrote humaneval_quality.ipynb")


if __name__ == "__main__":
    main()
