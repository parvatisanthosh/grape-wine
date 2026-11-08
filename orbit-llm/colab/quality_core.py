"""HumanEval scoring: build a test program from a model's answer and run it.

Standard library only. build_notebook.py copies this file into the Colab
notebook, and tests/test_quality_core.py checks it locally.

Model-written code is untrusted. On Colab it runs only after Google Drive has
been unmounted, in a separate process with limits on time, memory and file size.
"""

import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import resource
except ImportError:  # Windows
    resource = None


INSTRUCTION = (
    "Complete the following Python function. Return the complete function in one "
    "```python code block, including any imports it needs. Do not add tests or "
    "usage examples.\n\n```python\n{prompt}\n```"
)

TIMEOUT_S = 10
MEMORY_LIMIT_BYTES = 2 * 1024**3
FILE_LIMIT_BYTES = 10 * 1024**2

FENCED_BLOCK = re.compile(r"```[a-zA-Z]*[ \t]*\r?\n(.*?)```", re.DOTALL)


def extract_code(text, entry_point):
    """Return the code in a model's answer.

    Prefers the first complete code block that defines the function. An answer
    cut off by the token limit can leave a block unclosed, so that is handled
    too. Text with no code fence is returned unchanged.
    """
    blocks = FENCED_BLOCK.findall(text)
    for block in blocks:
        if f"def {entry_point}" in block:
            return block
    if blocks:
        return blocks[0]
    if "```" in text:
        after_fence = text.split("```", 1)[1]
        return after_fence.split("\n", 1)[1] if "\n" in after_fence else ""
    return text


def build_program(problem, answer):
    """Combine the prompt's preamble, the model's code and the official test."""
    entry_point = problem["entry_point"]
    prompt = problem["prompt"]
    code = extract_code(answer, entry_point)

    if re.search(rf"^\s*def {re.escape(entry_point)}\b", code, re.MULTILINE):
        # A full function: keep what the prompt defines before it (imports and
        # helper functions), and replace the function itself.
        match = re.search(rf"^def {re.escape(entry_point)}\b", prompt, re.MULTILINE)
        preamble = prompt[: match.start()] if match else ""
        body = preamble + code
    else:
        # Only the function body: it continues the prompt.
        body = prompt + code

    return f"{body}\n\n{problem['test']}\n\ncheck({entry_point})\n"


def _limit_resources():
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_LIMIT_BYTES, FILE_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run_program(program, timeout_s=TIMEOUT_S):
    """Run a test program. Returns {"outcome": passed|failed|timeout, "detail": ...}."""
    with tempfile.TemporaryDirectory(prefix="humaneval_") as folder:
        (Path(folder) / "candidate.py").write_text(program, encoding="utf-8")
        options = {}
        if resource is not None:
            options = {"preexec_fn": _limit_resources, "start_new_session": True}

        environment = {"PYTHONDONTWRITEBYTECODE": "1", "HOME": folder}
        if os.name == "nt":  # Python cannot start on Windows without this
            environment["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")

        process = subprocess.Popen(
            [sys.executable, "-I", "candidate.py"],
            cwd=folder,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **options,
        )
        try:
            _, errors = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            if resource is not None:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.communicate()
            return {"outcome": "timeout", "detail": f"over {timeout_s} s"}

    if process.returncode == 0:
        return {"outcome": "passed", "detail": ""}
    lines = errors.strip().splitlines()
    return {"outcome": "failed", "detail": lines[-1][:200] if lines else "no error text"}


def load_jsonl(path):
    """Read a JSON-lines file, skipping a last line cut off by a disconnect."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def end_with_newline(path):
    """Close a line cut off by a disconnect, so the next row starts cleanly."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as file:
        file.seek(-1, os.SEEK_END)
        if file.read(1) != b"\n":
            file.write(b"\n")


def wilson_interval(passed, total, z=1.96):
    """95% confidence interval for a proportion."""
    if total == 0:
        return (0.0, 1.0)
    share = passed / total
    denominator = 1 + z * z / total
    centre = (share + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total**2)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def paired_difference_interval(only_first, only_second, total, z=1.96):
    """95% confidence interval for first's pass rate minus second's, same problems."""
    difference = (only_first - only_second) / total
    variance = (only_first + only_second - (only_first - only_second) ** 2 / total) / total**2
    margin = z * math.sqrt(variance)
    return (difference - margin, difference + margin)


def mcnemar_exact(only_first, only_second):
    """Two-sided exact McNemar p-value from the problems the two models split on."""
    discordant = only_first + only_second
    if discordant == 0:
        return 1.0
    smaller = min(only_first, only_second)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2**discordant
    return min(1.0, 2 * tail)
