"""Python twin of repro_eviction_prefix.cpp, for testing other OpenVINO releases.

The OpenVINO GenAI Python package wraps the same C++ library, so a release can be
checked by installing it in a separate environment, without rebuilding anything.

    python repro_eviction_prefix.py MODEL_DIR DEVICE on|off [TURNS]

Prints one line per turn, then the OpenVINO GenAI version.
"""

import sys

import openvino_genai as og


def main():
    model, device, prefix = sys.argv[1], sys.argv[2], sys.argv[3] == "on"
    turns = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    scheduler = og.SchedulerConfig()
    scheduler.enable_prefix_caching = prefix
    scheduler.use_cache_eviction = True
    scheduler.cache_eviction_config = og.CacheEvictionConfig(
        32, 64, 256, og.AggregationMode.NORM_SUM
    )

    pipeline = og.LLMPipeline(model, device, scheduler_config=scheduler)
    config = og.GenerationConfig()
    config.max_new_tokens = 80

    history = og.ChatHistory()
    message = (
        "Tell me one more fact about the history of computing, in two sentences. "
        "Remember that my project is called ORBIT and my lucky number is 47."
    )
    print("OpenVINO GenAI", og.__version__, flush=True)
    for turn in range(1, turns + 1):
        history.append({"role": "user", "content": message})
        try:
            result = pipeline.generate(history, config)
        except Exception as error:  # noqa: BLE001 - the error text is the result
            print(f"turn {turn} FAILED: {str(error).strip().splitlines()[-1]}", flush=True)
            return 1
        history.append({"role": "assistant", "content": result.texts[0]})
        print(f"turn {turn} OK, prompt tokens {result.perf_metrics.get_num_input_tokens()}", flush=True)
    print(f"all {turns} turns OK (prefix caching {'on' if prefix else 'off'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
