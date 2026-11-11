# ORBIT evaluation v2 — plan, fixed before any run

Written on 2026-11-09, before any v2 data exists. The first evaluation could not separate
ORBIT from a static configuration (Finding 24): its synthetic scenarios had one dominant
choice, and its follow-ups never hit the prefix cache. This plan fixes the metric, the
systems and the analysis in advance, so the result cannot be tuned after the fact.

## 1. Workload: real request sizes and real conversations

All three streams come from `data/raw/` (see `data/README.md`) and are sampled with a
fixed seed. The pipeline sees exact-length prompts, so only lengths come from the data,
not text. Lengths are counted with the TinyLlama (Llama-2) tokenizer.

| Stream | Source | What is taken | Share |
|---|---|---|---|
| Chat | OpenAssistant oasst1, English conversation paths | Turn structure; prompt = whole history so far; output = assistant reply length | 50% |
| One-shot | Azure 2023 trace, conversation file | Prompt and output token counts | 25% |
| Code | Azure 2023 trace, code file | Prompt and output token counts | 25% |

- **Users take turns.** Four chat users are active at once, in a seeded random order. A
  follow-up turn repeats its conversation's earlier prompt as a prefix, so prefix
  retention (Finding 26) matters, as it does for a real assistant.
- **Caps, so the laptop can serve it:** prompts at most 4,096 tokens, outputs at most 256.
  The share of requests capped is reported. TinyLlama (2,048-token context) cannot serve
  requests over its limit, so those can go only to the 3B model.
- **No arrival times.** Requests run one at a time, so Azure timestamps are not used. The
  evaluation measures configuration choice, not queueing.

## 2. Service levels, fixed in advance

| Class | Limit | Reason |
|---|---|---|
| Chat and one-shot | TTFT ≤ 2 s and decode ≥ 10 tokens/s | First feedback within about 2 s; generation faster than reading speed |
| Code | TTFT ≤ 1.5 s and the code-quality floor below | Code help is waited on, and wrong code is worse than slow code |
| Every request | Peak memory within free RAM minus 512 MiB | No paging, no out-of-memory |

**Code-quality floor:** a model may serve code requests only if its HumanEval pass rate,
measured on the same notebook and settings as Finding 25, is at least 50%.
- Qwen2.5-Coder-3B is measured (82–84%).
- TinyLlama is **not yet measured**, so it cannot serve code until it is.

## 3. Systems compared on the same requests

1. **OpenVINO default:** INT8 on CPU, OpenVINO's default export for models above 1B.
2. **Best fixed configuration:** the one configuration with the fewest violations on a
   separate tuning trace with a different seed, then frozen. Its result on the test trace
   itself is reported too, as an upper bound for any fixed choice.
3. **Simple rule:** 3B INT4 on the iGPU for code or prompts over 2,048 tokens; otherwise
   TinyLlama INT4 on the iGPU.
4. **ORBIT without online correction:** shows what Layer 3 adds.
5. **ORBIT:** all three layers.

- **Same KV pool for every system:** a fixed pool sized for the four chat users, from the
  Finding 26 rule, so only the configuration choice differs.
- **Pool ablation:** ORBIT with the default dynamic pool, run once.
- **Refusals count as violations:** ORBIT serves its best near-miss rather than refusing.

## 4. Analysis, fixed in advance

- **Primary metric:** SLA-violation rate over all test requests.
- **Primary test:** ORBIT vs best fixed configuration, using exact McNemar on paired
  outcomes, two-sided at α = 0.05.
- **Secondary:**
  - Wilcoxon signed-rank and a 10,000-sample bootstrap 95% CI on the log ratio of
    per-request total time;
  - the same for TTFT;
  - mean quality and peak memory;
  - the same comparisons against the other three systems.
- **Noise floor:** the best fixed configuration run twice on the same trace (an A/A run).
  A difference smaller than the A/A difference is not claimed.
- **Requests no configuration can meet:** some limits above may be impossible on this
  laptop, for example a 1.5 s TTFT on a long code prompt with the 3B model. The limits
  stay as fixed here. Such requests fail for every system, so they are concordant pairs
  and do not affect McNemar's test. Violations are also reported per class, so it stays
  visible which limits this hardware cannot meet.

## 5. Order of work

1. **Dry run:** ORBIT decides but nothing runs. Count how many test requests it would
   serve with a different configuration from the best fixed one. If almost none differ,
   the workload cannot test ORBIT, and that is reported instead of running it.
2. **Pilot:** 60 requests. It checks the setup and gives the discordance rate used for
   the sample-size calculation.
3. **Full run:** at the sample size from step 2, overnight on mains power with other
   apps closed.

## 6. Still needed from outside

- **TinyLlama HumanEval** (Colab, same notebook): without it, TinyLlama cannot serve code
  requests (section 2). This needs the user's approval.

## 7. Dry-run result (2026-11-09, before any v2 measurement)

`python evaluate_v2.py dry-run --tuning traces/tuning.jsonl --test traces/test.jsonl`.
Output: `results/evaluation_v2_dry_run.txt`. These are Twin predictions, not measurements.
TinyLlama's HumanEval score was not available yet, so TinyLlama was barred from code.

| Test trace, 300 requests | Best fixed (INT8 @ iGPU) | OpenVINO default | Simple rule | ORBIT |
|---|---|---|---|---|
| Predicted violations | 112 | 168 | 97 | 96 |
| Of which code (74 requests) | 74 | 74 | 58 | 58 |

- **ORBIT vs best fixed:** different choices on 271 of 300 requests. The predicted
  discordant pairs are 16 (only best fixed broke) and 0 (only ORBIT broke). All 16 are
  code requests that a fixed TinyLlama may not serve.
- **ORBIT vs simple rule:** nearly the same (96 vs 97). On this laptop, routing by request
  type captures almost all of the predicted benefit.
- **The tuning choice is close:** INT8 @ iGPU 117, INT4 @ iGPU 119, Q3B-INT4 @ iGPU 119
  predicted violations. The measured tuning sweep may pick a different one.
- **Predicted serving time:** about 0.6–1.2 h per system for the test trace, before loads.

## 8. Changes made after the dry run, before any measurement

1. **Two passes, because of memory.** Each system keeps its own worker and models. All
   five at once would need about 12 GB, more than this laptop has free.
   - Pass A runs ORBIT, best fixed and a second copy of best fixed (the A/A noise run),
     interleaved per request. It holds the primary test.
   - Pass B runs default, rule and ORBIT without correction, interleaved per request.
2. **Chat follow-ups reuse only the earlier prompt, not the earlier reply.** The worker
   builds prompts from a fixed text, so the reply is not part of the next prompt. Prefix
   reuse is therefore a little smaller than in a real chat. Real chat prompts here are
   short (median about 62 tokens), so prefix caching matters little in this workload.
3. **Proposed secondary analysis, not part of the primary test:** a phase with the iGPU
   busy (as in v1), where a fixed rule cannot adapt. The dry run suggests that ORBIT can
   only beat the rule when conditions change. This is labelled exploratory because it was
   added after seeing the dry run.

## 9. TinyLlama's HumanEval score (2026-11-10, before any v2 measurement)

TinyLlama passes 6.1–7.3% (Finding 28), below the 50% floor in section 2, so it may not
serve code requests. The dry run, re-run with the measured scores, is unchanged
(`results/evaluation_v2_dry_run.txt`).

## 10. Three passes instead of two (2026-11-10, before any v2 measurement)

A memory check before the first run showed that section 8's passes need too much memory at
once. A system that keeps two models loaded (the 3B model on the iGPU plus TinyLlama) uses
about 3.9 GB, so pass B's three systems would need about 8–9 GB, and about 7 GB is free
with other apps closed. Each pass now has two systems:

| Pass | Systems, interleaved per request |
|---|---|
| A | ORBIT, best fixed (the primary test, unchanged) |
| B | simple rule, best fixed again (the A/A noise run) |
| C | ORBIT without correction, OpenVINO default |

The primary test is unchanged: McNemar on ORBIT vs best fixed, run interleaved in pass A.
The A/A noise run and the other comparisons now span two passes. They are paired by
request but not run at the same time, so they also include any drift between passes. That
makes the noise floor conservative (larger).

## 11. Memory rule, after the smoke test (2026-11-10, before any v2 measurement)

A 6-request smoke test with the real models ran every step end to end. It counted one ORBIT
request as a memory violation because free RAM fell to 345 MiB. In a pass, both systems
keep their models loaded, and other apps were open. Free RAM belongs to the whole machine,
so it cannot judge one system.

**Change:** a memory violation now means the run itself failed, for example out of memory.
Free RAM is still recorded for every request and reported. The overnight script refuses to
start a pass with less than 7 GB free, because the two systems of a pass peak at about
6 GB together (3.4 GB for a system with the 3B model and TinyLlama loaded, plus 2.6 GB for
the 3B model alone). The smoke-test rows are not evaluation data.

## 12. One system loaded at a time (2026-11-11, before any v2 measurement)

On the evening of the run, other apps (including this assistant's app) left only about
2 GB available. Two systems loaded together (~6 GB) would have pushed Windows into
paging and corrupted the timings.

**Change:** each pass cuts the test trace into blocks of 25 requests. For every block,
the pass's two systems serve it one after the other, in a seeded random order, and only
one system is loaded at any moment (peak ~3.4 GB).
- Both systems still serve the same requests, minutes apart instead of seconds.
- ORBIT keeps what it has learned between blocks. Loaded models and the prefix cache do
  not carry over, which is the same for every system.
- The analysis is unchanged. The A/A run (best fixed twice, in passes A and B) measures
  how much the timing drifts.

Available memory is recorded for every request, so any request served under pressure
can be identified.

## 13. After the measurement (2026-11-11): one fault found and corrected, one deviation

Everything above was fixed before any v2 data existed. This section was written after
the overnight run and changes no metric, system or test.

**Fault: some long prompts were not unique.** Five requests had a 3B time to first token
of about 50 ms on a 4,096-token prompt. Other prompts that long took about 23 s.
- **Cause:** the 3B model's tokenizer cuts over-long text from the left, at 32,768 tokens.
  For prompts above about 4,090 tokens, the synthetic text was longer than that, so its
  request tag was cut off. Two such prompts of the same length were then identical, and
  the second one hit the prefix cache.
- **Affected requests:** test requests 247 and 286 (in orbit, rule and orbit_nocorr) and
  tuning requests 184, 239 and 288 (in the 3B sweep). No other request is affected.
  `cpp/repro/check_prompt_ids.cpp` shows that, below the limit, the old and fixed prompt
  builders give identical token ids, on both models.
- **Fix:** `create_exact_token_input` now builds only as much text as it needs, and
  refuses to run if the tokenizer drops the start of the text.
- **Correction:** the 9 affected rows were re-measured with unique prompts
  (`rerun_duplicate_prompts.py`, written to `results/evaluation_v2/corrections.jsonl`).
  They were re-run in the same configuration, on a fresh pipeline after one warm-up
  request. All 9 now take 24–25 s and violate the TTFT limit.
- **Effect:** the systems that served those requests gain 2 violations each. The tuning
  sweep's choice is unchanged. `evaluate_v2.py analyze --corrections` prints both
  versions, and the report uses the corrected one.

**Deviation: no 60-request pilot (section 5, step 2).** The 6-request smoke test checked
the setup, and the full 300-request test trace was then run directly. No sample size was
derived from a pilot. The primary test is significant at this size anyway (Finding 29).
