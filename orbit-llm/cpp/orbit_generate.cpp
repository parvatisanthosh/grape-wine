// orbit_generate: answer a file of chat prompts with one fixed configuration.
//
// Used for task-level quality on this laptop (Finding 30): the same prompts and
// greedy settings as colab/run_generation.py, but through the GenAI C++ API on
// any device and KV-cache precision. Resumes: ids already in --out are skipped.
//
//   orbit_generate --model DIR --device GPU --kv-precision u8 \
//                  --prompts prompts.jsonl --out answers.jsonl [--max-new-tokens 512]
//
// Input rows:  {"id": "...", "content": "..."}  (one user message each)
// Output rows: {"id", "answer", "tokens", "input_tokens", "ttft_ms", "tpot_ms", "seconds",
//               "device", "kv_precision", "cache_dir"}
//
// Leave --cache-dir empty when comparing KV precisions on GPU: a model-cache hit
// returns the first precision's compiled blob (Finding 30).

#define ORBIT_RUN_NO_MAIN
#include "orbit_run.cpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <set>

namespace {

struct GenerateOptions {
    Options pipeline;  // device, kv precision, cache dir, prefix caching (orbit_run.cpp)
    std::string prompts;
    std::string out;
    std::size_t max_new_tokens = 512;
    std::size_t limit = 0;
};

GenerateOptions parse(int argc, char* argv[]) {
    GenerateOptions options;
    options.pipeline.prefix_caching = "off";  // every prompt is new; keep runs independent
    for (int index = 1; index < argc; index += 2) {
        const std::string name = argv[index];
        if (index + 1 >= argc) throw std::invalid_argument("Missing value for " + name);
        const std::string value = argv[index + 1];
        if (name == "--model") options.pipeline.model = value;
        else if (name == "--device") options.pipeline.device = value;
        else if (name == "--kv-precision") options.pipeline.kv_precision = value;
        else if (name == "--cache-dir") options.pipeline.cache_dir = value;
        else if (name == "--prompts") options.prompts = value;
        else if (name == "--out") options.out = value;
        else if (name == "--max-new-tokens") options.max_new_tokens = parse_size(name, value);
        else if (name == "--limit") options.limit = parse_size(name, value);
        else throw std::invalid_argument("Unknown option " + name);
    }
    if (options.pipeline.model.empty() || options.prompts.empty() || options.out.empty()) {
        throw std::invalid_argument("--model, --prompts and --out are required");
    }
    return options;
}

std::set<std::string> done_ids(const std::string& path) {
    std::set<std::string> ids;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        try {
            ids.insert(nlohmann::json::parse(line).at("id").get<std::string>());
        } catch (const std::exception&) {
            // a last line cut off by an interruption: generate that prompt again
        }
    }
    return ids;
}

}  // namespace

int main(int argc, char* argv[]) {
    SetConsoleOutputCP(CP_UTF8);
    GenerateOptions options;
    try {
        options = parse(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 2;
    }

    std::vector<nlohmann::json> prompts;
    {
        std::ifstream file(options.prompts);
        if (!file) {
            std::cerr << "Cannot read " << options.prompts << "\n";
            return 2;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) prompts.push_back(nlohmann::json::parse(line));
        }
    }
    if (options.limit && prompts.size() > options.limit) prompts.resize(options.limit);
    const auto done = done_ids(options.out);

    std::cerr << "Loading " << options.pipeline.model << " on " << options.pipeline.device
              << ", KV " << options.pipeline.kv_precision << " ...\n";
    const auto load_start = std::chrono::steady_clock::now();
    ov::genai::LLMPipeline pipeline(options.pipeline.model, options.pipeline.device,
                                    build_pipeline_properties(options.pipeline));
    std::cerr << "Loaded in "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count()
              << " s\n";

    // Same settings as colab/run_generation.py: the model's own generation
    // config, then greedy, no repetition penalty, at most 512 new tokens.
    ov::genai::GenerationConfig config = pipeline.get_generation_config();
    config.do_sample = false;
    config.repetition_penalty = 1.0f;
    config.max_new_tokens = options.max_new_tokens;

    std::ofstream out(options.out, std::ios::app | std::ios::binary);
    std::size_t number = 0;
    for (const auto& prompt : prompts) {
        ++number;
        const std::string id = prompt.at("id").get<std::string>();
        if (done.count(id)) continue;

        ov::genai::ChatHistory history;
        history.push_back({{"role", "user"}, {"content", prompt.at("content").get<std::string>()}});
        const auto start = std::chrono::steady_clock::now();
        const auto result = pipeline.generate(history, config);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        auto perf = result.perf_metrics;
        nlohmann::json row = {
            {"id", id},
            {"answer", result.texts.at(0)},
            {"tokens", perf.get_num_generated_tokens()},
            {"input_tokens", perf.get_num_input_tokens()},
            {"ttft_ms", perf.get_ttft().mean},
            {"tpot_ms", perf.get_tpot().mean},
            {"seconds", seconds},
            {"device", options.pipeline.device},
            {"kv_precision", options.pipeline.kv_precision},
            {"cache_dir", options.pipeline.cache_dir},
        };
        out << row.dump() << "\n";
        out.flush();
        std::cerr << "[" << number << "/" << prompts.size() << "] " << id << ": "
                  << perf.get_num_generated_tokens() << " tokens, " << seconds << " s\n";
    }
    return 0;
}
