"""Unit tests for HumanEval scoring. Every problem here is written by hand.

Run from orbit-llm/:  python -m pytest tests -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab"))

from quality_core import (  # noqa: E402
    build_program,
    end_with_newline,
    extract_code,
    load_jsonl,
    mcnemar_exact,
    paired_difference_interval,
    run_program,
    wilson_interval,
)


PROBLEM = {
    "entry_point": "add",
    "prompt": 'from typing import List\n\n\ndef add(a: int, b: int) -> int:\n    """Add."""\n',
    "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
}

HELPER_PROBLEM = {
    "entry_point": "twice",
    "prompt": "def helper(x):\n    return x * 2\n\n\ndef twice(x):\n",
    "test": "def check(candidate):\n    assert candidate(4) == 8\n",
}


def test_extract_prefers_the_block_defining_the_function():
    text = "```\nprint(1)\n```\nthen\n```python\ndef add(a, b):\n    return a + b\n```"
    assert "return a + b" in extract_code(text, "add")


def test_extract_handles_answer_cut_off_inside_a_block():
    text = "Here it is:\n```python\ndef add(a, b):\n    return a + b\n"
    assert extract_code(text, "add").startswith("def add")


def test_extract_returns_plain_text_unchanged():
    assert extract_code("    return a + b\n", "add") == "    return a + b\n"


def test_full_function_passes():
    answer = "```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```"
    assert run_program(build_program(PROBLEM, answer))["outcome"] == "passed"


def test_body_only_answer_continues_the_prompt():
    assert run_program(build_program(PROBLEM, "    return a + b\n"))["outcome"] == "passed"


def test_full_function_keeps_prompt_imports_and_helpers():
    answer = "```python\ndef twice(x):\n    return helper(x)\n```"
    assert run_program(build_program(HELPER_PROBLEM, answer))["outcome"] == "passed"


def test_wrong_answer_fails_with_a_reason():
    result = run_program(build_program(PROBLEM, "    return a - b\n"))
    assert result["outcome"] == "failed"
    assert "AssertionError" in result["detail"]


def test_endless_loop_times_out():
    result = run_program("while True:\n    pass\n", timeout_s=1)
    assert result["outcome"] == "timeout"


def test_load_jsonl_skips_a_cut_off_last_line(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text('{"task_id": "a"}\n{"task_id": ', encoding="utf-8")
    assert load_jsonl(path) == [{"task_id": "a"}]
    assert load_jsonl(tmp_path / "missing.jsonl") == []


def test_rows_added_after_a_cut_off_line_are_kept(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text('{"task_id": "a"}\n{"task_id": ', encoding="utf-8")
    end_with_newline(path)
    with path.open("a", encoding="utf-8") as file:
        file.write('{"task_id": "b"}\n')
    assert [row["task_id"] for row in load_jsonl(path)] == ["a", "b"]


def test_end_with_newline_leaves_complete_and_missing_files_alone(tmp_path):
    path = tmp_path / "answers.jsonl"
    end_with_newline(path)
    assert not path.exists()
    path.write_text('{"task_id": "a"}\n', encoding="utf-8")
    end_with_newline(path)
    assert path.read_text(encoding="utf-8") == '{"task_id": "a"}\n'


def test_wilson_interval_matches_known_value():
    low, high = wilson_interval(82, 164)
    assert low == pytest.approx(0.424, abs=0.002)
    assert high == pytest.approx(0.576, abs=0.002)


def test_paired_difference_interval():
    assert paired_difference_interval(0, 0, 100) == (0.0, 0.0)
    low, high = paired_difference_interval(8, 6, 164)
    assert low == pytest.approx(-0.0325, abs=0.0005)
    assert high == pytest.approx(0.0569, abs=0.0005)


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)
