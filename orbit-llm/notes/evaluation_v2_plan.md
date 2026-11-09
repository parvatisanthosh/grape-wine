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
