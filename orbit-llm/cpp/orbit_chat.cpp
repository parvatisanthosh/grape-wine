// orbit_chat: a local LLM chat application on the OpenVINO GenAI C++ API.
//
// Multi-turn chat with streamed output, using the model's own chat template
// (ov::genai::ChatHistory). The inference configuration — device, CPU core
// placement and the KV-cache management features of OpenVINO (paged attention,
// prefix caching, KV-cache precision and cache eviction) — is chosen on the
// command line, and every turn reports what it cost: prompt size, TTFT,
// decode speed and memory.
//
// Interactive:  orbit_chat --model DIR [options]
// Scripted:     orbit_chat --model DIR --script conversation.txt --json-out turns.jsonl

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "openvino/genai/cache_eviction.hpp"
#include "openvino/genai/chat_history.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/scheduler_config.hpp"
#include "openvino/runtime/properties.hpp"
#include "telemetry.hpp"

namespace {

struct Options {
    std::string model;
    std::string device = "CPU";
    std::string cores = "any";             // any | pcore | ecore (CPU only)
    std::size_t max_new_tokens = 256;
    float temperature = 0.0f;              // 0 = greedy (reproducible)
    float repetition_penalty = 1.0f;
    std::string prefix_caching = "on";     // on | off
    std::string kv_precision = "default";  // default | f32 | f16 | u8 (CPU only)
    std::string evict;                     // "start:recent:max" in tokens, empty = off
    std::string cache_dir;                 // compiled-model cache (use for GPU)
    std::string system_prompt;
    std::string script;                    // file with one user message per line
    std::string json_out;                  // per-turn metrics as JSON lines
};

void print_usage() {
    std::cerr <<
        "Usage: orbit_chat --model DIR [options]\n"
        "\n"
        "Inference configuration:\n"
        "  --device NAME               CPU | GPU (default CPU)\n"
        "  --cores TYPE                any | pcore | ecore (CPU only, default any)\n"
        "  --cache-dir DIR             Compiled-model cache; cuts GPU load time\n"
        "\n"
        "KV-cache management:\n"
        "  --prefix-caching on|off     Reuse the KV cache of earlier turns (default on)\n"
        "  --kv-precision TYPE         default | f32 | f16 | u8 (CPU only)\n"
        "  --evict START:RECENT:MAX    Cache eviction: keep the first START and last\n"
        "                              RECENT tokens, cap the cache at MAX tokens\n"
        "\n"
        "Generation:\n"
        "  --max-new-tokens N          Answer length limit (default 256)\n"
        "  --temperature T             0 = greedy (default), > 0 = sampling\n"
        "  --repetition-penalty P      Default 1.0 (off)\n"
        "  --system TEXT               System prompt\n"
        "\n"
        "Scripted runs:\n"
        "  --script FILE               Play one user message per line (# = comment)\n"
        "  --json-out FILE             Append per-turn metrics as JSON lines\n"
        "\n"
        "Chat commands: /reset  /stats  /help  /exit\n";
}

std::size_t parse_size(const std::string& name, const std::string& value) {
    try {
        std::size_t consumed = 0;
        const long long parsed = std::stoll(value, &consumed);
        if (consumed != value.size() || parsed < 0) {
            throw std::invalid_argument(value);
        }
        return static_cast<std::size_t>(parsed);
    } catch (const std::exception&) {
        throw std::invalid_argument(name + " expects a non-negative integer, got '" + value + "'");
    }
}

Options parse_arguments(int argc, char* argv[]) {
    Options options;

    for (int index = 1; index < argc; ++index) {
        const std::string name = argv[index];
        if (name == "--help" || name == "-h") {
            print_usage();
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("Missing value for " + name);
        }
        const std::string value = argv[++index];

        if (name == "--model") options.model = value;
        else if (name == "--device") options.device = value;
        else if (name == "--cores") options.cores = value;
        else if (name == "--max-new-tokens") options.max_new_tokens = parse_size(name, value);
        else if (name == "--temperature") options.temperature = std::stof(value);
        else if (name == "--repetition-penalty") options.repetition_penalty = std::stof(value);
        else if (name == "--prefix-caching") options.prefix_caching = value;
        else if (name == "--kv-precision") options.kv_precision = value;
        else if (name == "--evict") options.evict = value;
        else if (name == "--cache-dir") options.cache_dir = value;
        else if (name == "--system") options.system_prompt = value;
        else if (name == "--script") options.script = value;
        else if (name == "--json-out") options.json_out = value;
        else throw std::invalid_argument("Unknown option " + name);
    }

    if (options.model.empty()) throw std::invalid_argument("--model is required");
    if (options.device != "CPU" && options.device != "GPU") throw std::invalid_argument("--device must be CPU or GPU");
    if (options.prefix_caching != "on" && options.prefix_caching != "off") {
        throw std::invalid_argument("--prefix-caching must be on or off");
    }
    if (options.device != "CPU" && (options.cores != "any" || options.kv_precision != "default")) {
        // Explicit KV precision crashes the GPU paged-attention path on
        // GenAI 2026.3 (FINDINGS Finding 17).
        throw std::invalid_argument("--cores and --kv-precision apply to the CPU device only");
    }
    return options;
}

struct EvictionSizes {
    std::size_t start = 0;
    std::size_t recent = 0;
    std::size_t max = 0;
};

EvictionSizes parse_eviction(const std::string& text) {
    std::vector<std::size_t> parts;
    std::stringstream stream(text);
    std::string part;
    while (std::getline(stream, part, ':')) {
        parts.push_back(parse_size("--evict", part));
    }
    if (parts.size() != 3) {
        throw std::invalid_argument("--evict expects START:RECENT:MAX, got '" + text + "'");
    }
    return {parts[0], parts[1], parts[2]};
}

ov::element::Type to_element_type(const std::string& name) {
    if (name == "f32") return ov::element::f32;
    if (name == "f16") return ov::element::f16;
    if (name == "u8") return ov::element::u8;
    throw std::invalid_argument("--kv-precision must be default, f32, f16 or u8");
}

ov::AnyMap build_properties(const Options& options) {
    // Paged attention with LLMPipeline's own latency-oriented defaults
    // (unlimited prefill batch, prefix caching), see FINDINGS Finding 1.
    ov::genai::SchedulerConfig scheduler;
    scheduler.max_num_batched_tokens = std::numeric_limits<std::size_t>::max();
    scheduler.enable_prefix_caching = options.prefix_caching == "on";

    if (!options.evict.empty()) {
        const auto sizes = parse_eviction(options.evict);
        scheduler.use_cache_eviction = true;
        scheduler.cache_eviction_config = ov::genai::CacheEvictionConfig(
            sizes.start, sizes.recent, sizes.max, ov::genai::AggregationMode::NORM_SUM);
    }

    ov::AnyMap properties;
    properties.insert(ov::genai::scheduler_config(scheduler));

    if (options.kv_precision != "default") {
        properties.insert(ov::hint::kv_cache_precision(to_element_type(options.kv_precision)));
    }
    if (options.cores == "pcore") {
        properties.insert(ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::PCORE_ONLY));
    } else if (options.cores == "ecore") {
        properties.insert(ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::ECORE_ONLY));
    }
    if (!options.cache_dir.empty()) {
        properties.insert(ov::cache_dir(options.cache_dir));
    }
    return properties;
}

std::string json_escape(const std::string& text) {
    std::ostringstream escaped;
    for (const char character : text) {
        switch (character) {
            case '"': escaped << "\\\""; break;
            case '\\': escaped << "\\\\"; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default:
                if (static_cast<unsigned char>(character) < 0x20) {
                    escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                            << static_cast<int>(character) << std::dec;
                } else {
                    escaped << character;
                }
        }
    }
    return escaped.str();
}

std::string now_iso8601() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
    localtime_s(&local, &now);
    std::ostringstream text;
    text << std::put_time(&local, "%Y-%m-%dT%H:%M:%S");
    return text.str();
}

struct TurnMetrics {
    int turn = 0;
    std::size_t input_tokens = 0;
    std::size_t output_tokens = 0;
    double ttft_ms = 0;
    double tpot_ms = 0;
    double throughput = 0;
    double generate_ms = 0;
    double rss_mib = 0;
    double commit_mib = 0;
    orbit::PowerState power;
};

void print_metrics(const TurnMetrics& metrics) {
    std::cout << std::fixed << std::setprecision(1)
              << "  [turn " << metrics.turn << " | prompt " << metrics.input_tokens << " tok"
              << " | answer " << metrics.output_tokens << " tok"
              << " | TTFT " << metrics.ttft_ms << " ms"
              << " | " << metrics.throughput << " tok/s"
              << " | RSS " << metrics.rss_mib << " MiB"
              << " | commit " << metrics.commit_mib << " MiB]\n";
}

void write_json(const Options& options,
                const TurnMetrics& metrics,
                const std::string& user,
                const std::string& answer) {
    std::ofstream file(options.json_out, std::ios::app);
    if (!file) {
        throw std::runtime_error("Cannot open " + options.json_out);
    }
    file << std::fixed << std::setprecision(2)
         << "{\"timestamp\": \"" << now_iso8601() << "\""
         << ", \"model_path\": \"" << json_escape(options.model) << "\""
         << ", \"device\": \"" << options.device << "\""
         << ", \"cores\": \"" << options.cores << "\""
         << ", \"prefix_caching\": " << (options.prefix_caching == "on" ? "true" : "false")
         << ", \"kv_precision\": \"" << options.kv_precision << "\""
         << ", \"evict\": \"" << options.evict << "\""
         << ", \"turn\": " << metrics.turn
         << ", \"input_tokens\": " << metrics.input_tokens
         << ", \"output_tokens\": " << metrics.output_tokens
         << ", \"ttft_ms\": " << metrics.ttft_ms
         << ", \"tpot_ms_per_token\": " << metrics.tpot_ms
         << ", \"throughput_tokens_per_second\": " << metrics.throughput
         << ", \"generation_ms\": " << metrics.generate_ms
         << ", \"rss_mib\": " << metrics.rss_mib
         << ", \"commit_mib\": " << metrics.commit_mib
         << ", \"on_ac_power\": " << (metrics.power.known ? (metrics.power.on_ac ? "true" : "false") : "null")
         << ", \"battery_percent\": " << metrics.power.battery_percent
         << ", \"user\": \"" << json_escape(user) << "\""
         << ", \"answer\": \"" << json_escape(answer) << "\"}\n";
}

std::vector<std::string> read_script(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open script " + path);
    }
    std::vector<std::string> lines;
    std::string line;
    bool first_line = true;
    while (std::getline(file, line)) {
        // Editors such as PowerShell's Set-Content save UTF-8 with a BOM.
        if (first_line && line.rfind("\xEF\xBB\xBF", 0) == 0) line.erase(0, 3);
        first_line = false;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        lines.push_back(line);
    }
    return lines;
}

ov::genai::ChatHistory new_history(const Options& options) {
    ov::genai::ChatHistory history;
    if (!options.system_prompt.empty()) {
        history.push_back({{"role", "system"}, {"content", options.system_prompt}});
    }
    return history;
}

}  // namespace

int main(int argc, char* argv[]) {
    SetConsoleOutputCP(CP_UTF8);

    Options options;
    try {
        options = parse_arguments(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n\n";
        print_usage();
        return 2;
    }

    std::cout << "Loading " << options.model << " on " << options.device
              << (options.cores != "any" ? " (" + options.cores + ")" : "") << " ...\n";

    std::unique_ptr<ov::genai::LLMPipeline> pipeline;
    try {
        const auto start = std::chrono::steady_clock::now();
        pipeline = std::make_unique<ov::genai::LLMPipeline>(options.model, options.device, build_properties(options));
        const double load_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::cout << std::fixed << std::setprecision(1) << "Loaded in " << load_ms / 1000.0 << " s"
                  << " | RSS " << orbit::get_rss_mib() << " MiB"
                  << " | free RAM " << orbit::get_available_ram_mib() << " MiB\n"
                  << "KV cache: paged attention, prefix caching " << options.prefix_caching
                  << ", precision " << options.kv_precision
                  << (options.evict.empty() ? ", no eviction" : ", eviction " + options.evict + " tokens") << "\n";

        const auto power = orbit::get_power_state();
        if (power.known && !power.on_ac) {
            std::cout << "Warning: running on battery (" << power.battery_percent
                      << "%). Inference can be several times slower than on AC power.\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "Failed to load the model: " << error.what() << "\n";
        return 1;
    }

    ov::genai::GenerationConfig config = pipeline->get_generation_config();
    config.max_new_tokens = options.max_new_tokens;
    config.repetition_penalty = options.repetition_penalty;
    if (options.temperature > 0.0f) {
        config.do_sample = true;
        config.temperature = options.temperature;
    } else {
        config.do_sample = false;
    }

    const bool scripted = !options.script.empty();
    std::vector<std::string> script;
    if (scripted) {
        script = read_script(options.script);
    }
    std::size_t script_position = 0;

    auto history = new_history(options);
    bool show_stats = true;
    int turn = 0;

    auto streamer = [](std::string chunk) {
        std::cout << chunk << std::flush;
        return ov::genai::StreamingStatus::RUNNING;
    };

    std::cout << (scripted ? "\nPlaying " + options.script + "\n" : "\nType a message (/help for commands).\n");

    while (true) {
        std::string user;
        std::cout << "\nyou> " << std::flush;

        if (scripted) {
            if (script_position >= script.size()) break;
            user = script[script_position++];
            std::cout << user << "\n";
        } else if (!std::getline(std::cin, user)) {
            break;
        }

        if (user.empty()) continue;
        if (user == "/exit" || user == "/quit") break;
        if (user == "/help") { print_usage(); continue; }
        if (user == "/stats") {
            show_stats = !show_stats;
            std::cout << "Per-turn stats " << (show_stats ? "on" : "off") << "\n";
            continue;
        }
        if (user == "/reset") {
            history = new_history(options);
            turn = 0;
            std::cout << "Conversation cleared.\n";
            continue;
        }

        history.push_back({{"role", "user"}, {"content", user}});
        std::cout << "llm> " << std::flush;

        try {
            const auto result = pipeline->generate(history, config, streamer);
            std::cout << "\n";

            const std::string answer = result.texts.at(0);
            history.push_back({{"role", "assistant"}, {"content", answer}});

            auto perf = result.perf_metrics;
            TurnMetrics metrics;
            metrics.turn = ++turn;
            metrics.input_tokens = perf.get_num_input_tokens();
            metrics.output_tokens = perf.get_num_generated_tokens();
            metrics.ttft_ms = perf.get_ttft().mean;
            metrics.tpot_ms = perf.get_tpot().mean;
            metrics.throughput = perf.get_throughput().mean;
            metrics.generate_ms = perf.get_generate_duration().mean;
            metrics.rss_mib = orbit::get_rss_mib();
            metrics.commit_mib = orbit::get_commit_mib();
            metrics.power = orbit::get_power_state();

            if (show_stats) print_metrics(metrics);
            if (!options.json_out.empty()) write_json(options, metrics, user, answer);
        } catch (const std::exception& error) {
            std::cout << "\n";
            std::cerr << "Generation failed: " << error.what() << "\n"
                      << "Conversation cleared after the error.\n";
            history = new_history(options);
            turn = 0;
            if (scripted) return 1;
        }
    }

    std::cout << "\nBye.\n";
    return 0;
}
