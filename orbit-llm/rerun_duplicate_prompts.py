"""Re-measure the evaluation v2 requests whose prompts were not unique (Finding 29).

Before the fix in cpp/orbit_run.cpp (create_exact_token_input), a Qwen prompt
of about 4,090 tokens or more lost its tag to the tokenizer's left truncation,
so two such prompts of the same length were identical. Five requests then hit
the prefix cache on a prompt they should never have seen:

  test trace:   247 and 286, in orbit, rule and orbit_nocorr (Q3B-INT4 @ iGPU)
  tuning trace: 184, 239 and 288, in the Q3B-INT4 @ iGPU sweep

Every other request is unaffected: below the limit the old and new builders give
identical token ids (cpp/repro/check_prompt_ids.cpp), and above it a request
that hit nothing took the same time as any other prompt of its length.

Each affected row is served again with the fixed worker: same configuration and
KV pool, on a fresh pipeline after one short warm-up request (the original ran
mid-block on a warm pipeline). Rows are judged with the same limits and written
to results/evaluation_v2/corrections.jsonl. `evaluate_v2.py analyze` reports the
results both as measured and with these corrections applied.

Run from orbit-llm/ on mains power:  python rerun_duplicate_prompts.py
"""

import json
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "colab"))

import evaluate_v2 as ev  # noqa: E402
from evaluate_controller import ResidentService  # noqa: E402
from orbit_worker_client import OrbitWorker  # noqa: E402
from quality_core import load_jsonl  # noqa: E402
from run_experiment import prevent_sleep, require_ac_power  # noqa: E402

RESULTS = ev.RESULTS_DIR
OUT = RESULTS / "corrections.jsonl"
CONFIGURATION = "Q3B-INT4@gpu"
AFFECTED = [
    # (trace, results file, system, index)
    *[("traces/test.jsonl", f"pass_{p}.jsonl", system, index)
      for p, system in (("A", "orbit"), ("B", "rule"), ("C", "orbit_nocorr"))
      for index in (247, 286)],
    *[("traces/tuning.jsonl", "sweep_tuning.jsonl", "fixed", index) for index in (184, 239, 288)],
]
# Fields of the original row that describe ORBIT's decision, not the measurement.
DECISION_FIELDS = ("pass", "mode", "pred_ttft_ms", "pred_tpot_ms")


def warmup_request(request):
    return Namespace(**{**vars(request), "prompt_id": "warmup", "prompt_tokens": 64,
                        "output_tokens": 4, "prompt_key": "warmup:64"})


def main():
    require_ac_power(False)
    prevent_sleep()
    context = ev.Context.load()
    done = {(row["file"], row["system"], row["index"]) for row in load_jsonl(OUT)} if OUT.exists() else set()
    label, placement = CONFIGURATION.split("@")

    for trace_path, file_name, system, index in AFFECTED:
        if (file_name, system, index) in done:
            continue
        trace = ev.load_trace(trace_path)
        request = next(r for r in trace if r.index == index)
        original = next(row for row in load_jsonl(RESULTS / file_name)
                        if row["system"] == system and row["index"] == index
                        and row["candidate"] == CONFIGURATION)
        assert original["candidate"] == CONFIGURATION, original

        service = ResidentService(context.models, 1, OrbitWorker, kv_blocks=ev.pool_for(trace))
        try:
            service.run(label, placement, warmup_request(request))
            row = service.run(label, placement, request)
        finally:
            service.close()

        record = {
            "file": file_name,
            "system": system,
            **{key: original[key] for key in DECISION_FIELDS if key in original},
            **ev.request_fields(request),
            "candidate": CONFIGURATION,
            "switched": original.get("switched", False),
            **ev.judge(request, row, original.get("load_ms_charged") or 0.0),
            "corrected": "unique prompt, re-measured (Finding 29)",
            "original_ttft_ms": original.get("ttft_ms"),
            "original_violations": original.get("violations"),
        }
        ev.append(OUT, record)
        print(f"{file_name} {system} {index}: TTFT {original.get('ttft_ms')} -> {record.get('ttft_ms')} ms, "
              f"violations '{original.get('violations')}' -> '{record['violations']}'", flush=True)


if __name__ == "__main__":
    main()
