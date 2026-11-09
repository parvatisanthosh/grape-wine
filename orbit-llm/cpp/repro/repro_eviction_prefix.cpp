// Minimal reproduction: cache eviction together with prefix caching in a
// multi-turn chat. Draft report: notes/upstream_issue_eviction_prefix_caching.md.
//
//   repro_eviction_prefix MODEL_DIR DEVICE PREFIX_CACHING(on|off) [TURNS]
//
// Each turn appends a user message to a ChatHistory, so every prompt contains the
// earlier turns, and generates 80 tokens. The eviction budget is
// CacheEvictionConfig(32, 64, 256), so eviction starts after a few turns.
// Prints one line per turn: OK with the prompt length, or FAILED with the error.

#include <iostream>
#include <string>

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: repro_eviction_prefix MODEL_DIR DEVICE on|off [TURNS]\n";
        return 2;
    }
    const std::string model = argv[1];
    const std::string device = argv[2];
    const bool prefix_caching = std::string(argv[3]) == "on";
    const int turns = argc == 5 ? std::stoi(argv[4]) : 8;

    ov::genai::SchedulerConfig scheduler;
    scheduler.enable_prefix_caching = prefix_caching;
    scheduler.use_cache_eviction = true;
    scheduler.cache_eviction_config =
        ov::genai::CacheEvictionConfig(32, 64, 256, ov::genai::AggregationMode::NORM_SUM);

    ov::genai::LLMPipeline pipeline(model, device, ov::genai::scheduler_config(scheduler));
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 80;

    ov::genai::ChatHistory history;
    const std::string message =
        "Tell me one more fact about the history of computing, in two sentences. "
        "Remember that my project is called ORBIT and my lucky number is 47.";

    for (int turn = 1; turn <= turns; ++turn) {
        history.push_back({{"role", "user"}, {"content", message}});
        try {
            auto result = pipeline.generate(history, config);  // get_num_input_tokens() is non-const
            const std::string reply = result.texts.at(0);
            history.push_back({{"role", "assistant"}, {"content", reply}});
            std::cout << "turn " << turn << " OK, prompt tokens "
                      << result.perf_metrics.get_num_input_tokens() << "\n";
        } catch (const std::exception& error) {
            std::cout << "turn " << turn << " FAILED: " << error.what() << "\n";
            return 1;
        }
    }
    std::cout << "all " << turns << " turns OK (prefix caching " << (prefix_caching ? "on" : "off") << ")\n";
    return 0;
}
