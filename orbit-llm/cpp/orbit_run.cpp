// orbit_run: run one controlled OpenVINO GenAI benchmark configuration.
//
// Loads one model with one configuration (device, attention backend, KV-cache
// and threading settings), runs warm-up and measured generations on prompts of
// an exact token length, and emits one JSON line per measured run. Human-readable
// progress goes to stderr; JSON lines go to stdout and, optionally, a file.
//
// This is the ORBIT-LLM executor: the Python orchestration layer chooses a
// configuration, calls this program, and learns from the rows it writes.

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/scheduler_config.hpp"
#include "openvino/genai/version.hpp"
#include "openvino/runtime/properties.hpp"

namespace {

constexpr auto MEMORY_SAMPLE_INTERVAL = std::chrono::milliseconds(10);
constexpr auto CPU_LOAD_WINDOW = std::chrono::milliseconds(200);

// The first four variants match benchmark_cold_worker.py so C++ and Python
// results are directly comparable.
const std::map<std::string, std::pair<std::string, std::string>> PROMPT_VARIANTS = {
    {"warmup",
     {"Computer systems initialization workload. ",
      "Processors execute instructions and manage memory resources. "}},
    {"run_1",
     {"Ancient Roman history analysis workload. ",
      "Roman institutions influenced government law trade and engineering. "}},
    {"run_2",
     {"Astronomy and space science analysis workload. ",
      "Stars planets galaxies gravity and radiation shape the universe. "}},
    {"run_3",
     {"Marine biology and ocean science workload. ",
      "Ocean ecosystems contain diverse organisms currents and habitats. "}},
    {"run_4",
     {"Volcano and earthquake geology workload. ",
      "Tectonic plates magma faults and eruptions reshape continents. "}},
    {"run_5",
     {"Medieval architecture and construction workload. ",
      "Cathedrals castles arches and buttresses required skilled masons. "}},
};

struct Options {
    std::string model;
    std::string label;
    std::string device = "CPU";
    std::size_t prompt_tokens = 0;
    std::size_t output_tokens = 0;
    int warmup = 1;
    int runs = 3;
    std::string prompt_mode = "cold";      // cold | repeat
    std::string backend = "pa";            // pa | sdpa
    std::string prefix_caching = "on";     // on | off (PA backend only)
    std::size_t max_batched_tokens = 0;    // 0 = unlimited (PA backend only)
    std::size_t cache_size_gb = 0;         // 0 = dynamic allocation (PA backend only)
    std::string kv_precision = "default";  // default | f32 | f16 | bf16 | u8 | u4
    int threads = 0;                       // 0 = plugin default
    std::string cores = "any";             // any | pcore | ecore (CPU only)
    std::string output;
};

void print_usage() {
    std::cerr <<
        "Usage: orbit_run --model DIR --prompt-tokens N --output-tokens N [options]\n"
        "\n"
        "Workload:\n"
        "  --label TEXT              Precision/model label stored in results (default: folder name)\n"
        "  --warmup N                Unmeasured warm-up runs (default 1)\n"
        "  --runs N                  Measured runs, 1-5 (default 3)\n"
        "  --prompt-mode MODE        cold: unique prompt per run; repeat: same prompt every run\n"
        "\n"
        "Configuration:\n"
        "  --device NAME             CPU | GPU (default CPU)\n"
        "  --backend NAME            pa (paged attention, default) | sdpa\n"
        "  --prefix-caching on|off   PA backend only (default on, same as LLMPipeline default)\n"
        "  --max-batched-tokens N    PA backend only, 0 = unlimited (default 0)\n"
        "  --cache-size-gb N         PA backend only, 0 = grow dynamically (default 0)\n"
        "  --kv-precision TYPE       default | f32 | f16 | bf16 | u8 | u4\n"
        "  --threads N               CPU inference threads, 0 = plugin default\n"
        "  --cores TYPE              any | pcore | ecore (CPU only)\n"
        "\n"
        "Output:\n"
        "  --output FILE             Also append JSON lines to FILE\n";
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

void require_one_of(const std::string& name, const std::string& value, const std::vector<std::string>& allowed) {
    for (const auto& candidate : allowed) {
        if (value == candidate) {
            return;
        }
    }
    std::string message = name + " must be one of:";
    for (const auto& candidate : allowed) {
        message += " " + candidate;
    }
    throw std::invalid_argument(message + " (got '" + value + "')");
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
        else if (name == "--label") options.label = value;
        else if (name == "--device") options.device = value;
        else if (name == "--prompt-tokens") options.prompt_tokens = parse_size(name, value);
        else if (name == "--output-tokens") options.output_tokens = parse_size(name, value);
        else if (name == "--warmup") options.warmup = static_cast<int>(parse_size(name, value));
        else if (name == "--runs") options.runs = static_cast<int>(parse_size(name, value));
        else if (name == "--prompt-mode") options.prompt_mode = value;
        else if (name == "--backend") options.backend = value;
        else if (name == "--prefix-caching") options.prefix_caching = value;
        else if (name == "--max-batched-tokens") options.max_batched_tokens = parse_size(name, value);
        else if (name == "--cache-size-gb") options.cache_size_gb = parse_size(name, value);
        else if (name == "--kv-precision") options.kv_precision = value;
        else if (name == "--threads") options.threads = static_cast<int>(parse_size(name, value));
        else if (name == "--cores") options.cores = value;
        else if (name == "--output") options.output = value;
        else throw std::invalid_argument("Unknown option " + name);
    }

    if (options.model.empty()) throw std::invalid_argument("--model is required");
    if (options.prompt_tokens < 2) throw std::invalid_argument("--prompt-tokens must be at least 2");
    if (options.output_tokens < 1) throw std::invalid_argument("--output-tokens must be at least 1");
    if (options.runs < 1 || options.runs > 5) throw std::invalid_argument("--runs must be between 1 and 5");

    require_one_of("--prompt-mode", options.prompt_mode, {"cold", "repeat"});
    require_one_of("--device", options.device, {"CPU", "GPU"});
    require_one_of("--backend", options.backend, {"pa", "sdpa"});
    require_one_of("--prefix-caching", options.prefix_caching, {"on", "off"});
    require_one_of("--kv-precision", options.kv_precision, {"default", "f32", "f16", "bf16", "u8", "u4"});
    require_one_of("--cores", options.cores, {"any", "pcore", "ecore"});

    if (options.backend == "sdpa" &&
        (options.prefix_caching == "off" || options.max_batched_tokens != 0 || options.cache_size_gb != 0)) {
        throw std::invalid_argument(
            "--prefix-caching, --max-batched-tokens and --cache-size-gb apply to the PA backend only");
    }
    if (options.device != "CPU" && (options.threads != 0 || options.cores != "any")) {
        throw std::invalid_argument("--threads and --cores apply to the CPU device only");
    }
    if (options.label.empty()) {
        const auto slash = options.model.find_last_of("/\\");
        options.label = slash == std::string::npos ? options.model : options.model.substr(slash + 1);
    }

    return options;
}

// ---------------------------------------------------------------------------
// Process and machine telemetry (Windows).

double bytes_to_mib(std::uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

double get_rss_mib() {
    PROCESS_MEMORY_COUNTERS counters{};
    GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters));
    return bytes_to_mib(counters.WorkingSetSize);
}

double get_lifetime_peak_rss_mib() {
    PROCESS_MEMORY_COUNTERS counters{};
    GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters));
    return bytes_to_mib(counters.PeakWorkingSetSize);
}

// Committed private memory. Unlike the working set, this includes memory that
// has been allocated but not yet touched (e.g. a pre-allocated KV cache), and it
// is what counts against the system commit limit when predicting OOM.
double get_commit_mib() {
    PROCESS_MEMORY_COUNTERS_EX counters{};
    GetProcessMemoryInfo(GetCurrentProcess(),
                         reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                         sizeof(counters));
    return bytes_to_mib(counters.PrivateUsage);
}

double get_lifetime_peak_commit_mib() {
    PROCESS_MEMORY_COUNTERS counters{};
    GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters));
    return bytes_to_mib(counters.PeakPagefileUsage);
}

double get_available_ram_mib() {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return bytes_to_mib(status.ullAvailPhys);
}

double get_total_ram_mib() {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return bytes_to_mib(status.ullTotalPhys);
}

std::uint64_t filetime_to_u64(const FILETIME& time) {
    return (static_cast<std::uint64_t>(time.dwHighDateTime) << 32) | time.dwLowDateTime;
}

// System-wide CPU busy percentage over a short window. Called while this
// process is idle, so it measures background load from other programs.
double sample_system_cpu_busy_percent() {
    FILETIME idle_a, kernel_a, user_a, idle_b, kernel_b, user_b;
    GetSystemTimes(&idle_a, &kernel_a, &user_a);
    std::this_thread::sleep_for(CPU_LOAD_WINDOW);
    GetSystemTimes(&idle_b, &kernel_b, &user_b);

    const auto idle = filetime_to_u64(idle_b) - filetime_to_u64(idle_a);
    // Kernel time includes idle time.
    const auto total = (filetime_to_u64(kernel_b) - filetime_to_u64(kernel_a)) +
                       (filetime_to_u64(user_b) - filetime_to_u64(user_a));
    if (total == 0) {
        return 0.0;
    }
    return 100.0 * static_cast<double>(total - idle) / static_cast<double>(total);
}

// Samples this process's working set in the background and keeps the maximum.
class PeakMemorySampler {
public:
    PeakMemorySampler() : m_peak(get_rss_mib()), m_thread([this] { run(); }) {}

    ~PeakMemorySampler() {
        stop();
    }

    double stop() {
        if (m_thread.joinable()) {
            m_stop = true;
            m_thread.join();
        }
        return m_peak;
    }

private:
    void run() {
        while (!m_stop) {
            const double current = get_rss_mib();
            if (current > m_peak) {
                m_peak = current;
            }
            std::this_thread::sleep_for(MEMORY_SAMPLE_INTERVAL);
        }
    }

    std::atomic<bool> m_stop{false};
    std::atomic<double> m_peak;
    std::thread m_thread;
};

// ---------------------------------------------------------------------------
// JSON line output.

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

class JsonRow {
public:
    void set(const std::string& key, const std::string& value) {
        put(key, "\"" + json_escape(value) + "\"");
    }
    void set(const std::string& key, const char* value) {
        set(key, std::string(value));
    }
    void set(const std::string& key, bool value) {
        put(key, value ? "true" : "false");
    }
    void set(const std::string& key, int value) {
        put(key, std::to_string(value));
    }
    void set(const std::string& key, std::size_t value) {
        put(key, std::to_string(value));
    }
    void set(const std::string& key, double value, int decimals = 2) {
        std::ostringstream number;
        number << std::fixed << std::setprecision(decimals) << value;
        put(key, number.str());
    }
    void set_null(const std::string& key) {
        put(key, "null");
    }

    std::string str() const {
        std::string line = "{";
        for (std::size_t index = 0; index < m_fields.size(); ++index) {
            if (index > 0) {
                line += ", ";
            }
            line += "\"" + m_fields[index].first + "\": " + m_fields[index].second;
        }
        return line + "}";
    }

private:
    // Keeps insertion order; replacing a key keeps its original position.
    void put(const std::string& key, const std::string& raw_value) {
        for (auto& field : m_fields) {
            if (field.first == key) {
                field.second = raw_value;
                return;
            }
        }
        m_fields.emplace_back(key, raw_value);
    }

    std::vector<std::pair<std::string, std::string>> m_fields;
};

void emit_row(const JsonRow& row, const Options& options) {
    const std::string line = row.str();
    std::cout << line << std::endl;

    if (!options.output.empty()) {
        std::ofstream file(options.output, std::ios::app);
        if (!file) {
            throw std::runtime_error("Cannot open output file: " + options.output);
        }
        file << line << "\n";
    }
}

std::string now_iso8601() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
    localtime_s(&local, &now);
    std::ostringstream text;
    text << std::put_time(&local, "%Y-%m-%dT%H:%M:%S");
    return text.str();
}

// ---------------------------------------------------------------------------
// Pipeline configuration and prompts.

ov::element::Type to_element_type(const std::string& name) {
    if (name == "f32") return ov::element::f32;
    if (name == "f16") return ov::element::f16;
    if (name == "bf16") return ov::element::bf16;
    if (name == "u8") return ov::element::u8;
    if (name == "u4") return ov::element::u4;
    throw std::invalid_argument("Unsupported KV-cache precision: " + name);
}

ov::AnyMap build_pipeline_properties(const Options& options) {
    ov::AnyMap properties;

    if (options.backend == "sdpa") {
        properties["ATTENTION_BACKEND"] = std::string("SDPA");
    } else {
        // Start from LLMPipeline's own latency-oriented defaults. A plain
        // SchedulerConfig{} would silently cap prefill at 256 tokens per step
        // and disable prefix caching, changing the behaviour being measured.
        ov::genai::SchedulerConfig scheduler;
        scheduler.max_num_batched_tokens = options.max_batched_tokens == 0
                                               ? std::numeric_limits<std::size_t>::max()
                                               : options.max_batched_tokens;
        scheduler.enable_prefix_caching = options.prefix_caching == "on";
        scheduler.cache_size = options.cache_size_gb;
        properties.insert(ov::genai::scheduler_config(scheduler));
    }

    if (options.kv_precision != "default") {
        properties.insert(ov::hint::kv_cache_precision(to_element_type(options.kv_precision)));
    }
    if (options.threads > 0) {
        properties.insert(ov::inference_num_threads(options.threads));
    }
    if (options.cores == "pcore") {
        properties.insert(ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::PCORE_ONLY));
    } else if (options.cores == "ecore") {
        properties.insert(ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::ECORE_ONLY));
    }

    return properties;
}

ov::genai::TokenizedInputs create_exact_token_input(ov::genai::Tokenizer& tokenizer,
                                                    std::size_t target_tokens,
                                                    const std::string& prompt_id) {
    const auto& [opening, body] = PROMPT_VARIANTS.at(prompt_id);

    std::string text = opening;
    for (std::size_t repeat = 0; repeat < target_tokens + 10; ++repeat) {
        text += body;
    }

    auto tokenized = tokenizer.encode(text,
                                      {ov::genai::add_special_tokens(true),
                                       ov::genai::max_length(target_tokens),
                                       ov::genai::truncation(true)});

    const std::size_t actual_tokens = tokenized.input_ids.get_shape().at(1);
    if (actual_tokens != target_tokens) {
        throw std::runtime_error(prompt_id + ": requested " + std::to_string(target_tokens) +
                                 " tokens, but tokenizer produced " + std::to_string(actual_tokens));
    }
    return tokenized;
}

std::string prompt_id_for(const Options& options, int iteration) {
    if (options.prompt_mode == "repeat") {
        return "run_1";
    }
    return iteration == 0 ? "warmup" : "run_" + std::to_string(iteration);
}

// Fields shared by every row of this invocation: configuration and load state.
JsonRow make_base_row(const Options& options) {
    JsonRow row;
    row.set("timestamp", now_iso8601());
    row.set("executor", "cpp");
    row.set("genai_version", ov::genai::get_version().buildNumber);
    row.set("label", options.label);
    row.set("model_path", options.model);
    row.set("device", options.device);
    row.set("backend", options.backend);
    if (options.backend == "pa") {
        row.set("prefix_caching", options.prefix_caching == "on");
        row.set("max_batched_tokens", options.max_batched_tokens);
        row.set("cache_size_gb", options.cache_size_gb);
    } else {
        row.set_null("prefix_caching");
        row.set_null("max_batched_tokens");
        row.set_null("cache_size_gb");
    }
    row.set("kv_precision", options.kv_precision);
    row.set("threads", options.threads);
    row.set("cores", options.cores);
    row.set("prompt_mode", options.prompt_mode);
    row.set("requested_input_tokens", options.prompt_tokens);
    row.set("requested_output_tokens", options.output_tokens);
    row.set("sys_total_ram_mib", get_total_ram_mib(), 1);
    return row;
}

}  // namespace

int main(int argc, char* argv[]) {
    Options options;
    try {
        options = parse_arguments(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n\n";
        print_usage();
        return 2;
    }

    JsonRow base = make_base_row(options);

    std::cerr << std::string(60, '=') << "\n"
              << "Label: " << options.label << "\n"
              << "Model: " << options.model << "\n"
              << "Device: " << options.device << ", backend: " << options.backend << "\n"
              << "Input/output tokens: " << options.prompt_tokens << "/" << options.output_tokens << "\n"
              << "Warm-up runs: " << options.warmup << ", measured runs: " << options.runs << "\n"
              << std::string(60, '=') << "\n";

    const double rss_before_load = get_rss_mib();
    base.set("rss_before_load_mib", rss_before_load, 1);
    base.set("sys_available_ram_before_load_mib", get_available_ram_mib(), 1);

    std::unique_ptr<ov::genai::LLMPipeline> pipeline;
    std::map<std::string, ov::genai::TokenizedInputs> prompts;

    try {
        const auto load_start = std::chrono::steady_clock::now();
        pipeline = std::make_unique<ov::genai::LLMPipeline>(
            options.model, options.device, build_pipeline_properties(options));
        const double construction_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();

        base.set("pipeline_construction_ms", construction_ms);
        base.set("rss_after_load_mib", get_rss_mib(), 1);
        base.set("commit_after_load_mib", get_commit_mib(), 1);

        auto tokenizer = pipeline->get_tokenizer();
        for (int iteration = 0; iteration <= options.runs; ++iteration) {
            const std::string prompt_id = prompt_id_for(options, iteration);
            if (prompts.count(prompt_id) == 0) {
                prompts.emplace(prompt_id, create_exact_token_input(tokenizer, options.prompt_tokens, prompt_id));
            }
        }

        std::cerr << "Pipeline construction: " << construction_ms << " ms\n";
    } catch (const std::exception& error) {
        // Load failures (e.g. out of memory) are data, not just errors.
        base.set("iteration", 0);
        base.set("status", "load_failed");
        base.set("error", error.what());
        base.set("lifetime_peak_rss_mib", get_lifetime_peak_rss_mib(), 1);
        base.set("lifetime_peak_commit_mib", get_lifetime_peak_commit_mib(), 1);
        emit_row(base, options);
        std::cerr << "LOAD FAILED: " << error.what() << "\n";
        return 1;
    }

    ov::genai::GenerationConfig config;
    config.max_new_tokens = options.output_tokens;
    config.ignore_eos = true;
    config.do_sample = false;
    config.apply_chat_template = false;

    std::cerr << "Running warm-up...\n";
    try {
        for (int run = 0; run < options.warmup; ++run) {
            pipeline->generate(prompts.at(prompt_id_for(options, 0)), config);
        }
    } catch (const std::exception& error) {
        base.set("iteration", 0);
        base.set("status", "warmup_failed");
        base.set("error", error.what());
        base.set("lifetime_peak_rss_mib", get_lifetime_peak_rss_mib(), 1);
        base.set("lifetime_peak_commit_mib", get_lifetime_peak_commit_mib(), 1);
        emit_row(base, options);
        std::cerr << "WARM-UP FAILED: " << error.what() << "\n";
        return 1;
    }
    base.set("rss_after_warmup_mib", get_rss_mib(), 1);
    base.set("commit_after_warmup_mib", get_commit_mib(), 1);

    int failed_runs = 0;

    for (int iteration = 1; iteration <= options.runs; ++iteration) {
        const std::string prompt_id = prompt_id_for(options, iteration);
        std::cerr << "\nMeasured iteration " << iteration << " using " << prompt_id << "...\n";

        JsonRow row = base;
        row.set("timestamp", now_iso8601());
        row.set("iteration", iteration);
        row.set("prompt_id", prompt_id);
        row.set("sys_available_ram_mib", get_available_ram_mib(), 1);
        row.set("sys_cpu_busy_percent", sample_system_cpu_busy_percent(), 1);
        row.set("status", "failed");
        row.set("error", "");

        PeakMemorySampler sampler;
        bool succeeded = false;

        try {
            const auto wall_start = std::chrono::steady_clock::now();
            const auto result = pipeline->generate(prompts.at(prompt_id), config);
            const double wall_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_start).count();

            auto metrics = result.perf_metrics;
            const std::size_t input_tokens = metrics.get_num_input_tokens();
            const std::size_t output_tokens = metrics.get_num_generated_tokens();

            if (input_tokens != options.prompt_tokens) {
                throw std::runtime_error("Expected " + std::to_string(options.prompt_tokens) +
                                         " input tokens, OpenVINO reported " + std::to_string(input_tokens));
            }
            if (output_tokens != options.output_tokens) {
                throw std::runtime_error("Expected " + std::to_string(options.output_tokens) +
                                         " output tokens, OpenVINO reported " + std::to_string(output_tokens));
            }

            row.set("actual_input_tokens", input_tokens);
            row.set("actual_output_tokens", output_tokens);
            row.set("wall_time_ms", wall_ms);
            row.set("generation_ms", static_cast<double>(metrics.get_generate_duration().mean));
            row.set("ttft_ms", static_cast<double>(metrics.get_ttft().mean));
            row.set("tpot_ms_per_token", static_cast<double>(metrics.get_tpot().mean));
            row.set("throughput_tokens_per_second", static_cast<double>(metrics.get_throughput().mean));
            row.set("status", "success");
            succeeded = true;
            std::cerr << "TTFT: " << metrics.get_ttft().mean << " ms, TPOT: " << metrics.get_tpot().mean
                      << " ms/token, throughput: " << metrics.get_throughput().mean << " tok/s\n";
        } catch (const std::exception& error) {
            row.set("error", error.what());
            std::cerr << "FAILED: " << error.what() << "\n";
            ++failed_runs;
        }

        const double peak_rss = sampler.stop();
        row.set("sampled_peak_rss_mib", peak_rss, 1);
        row.set("rss_after_generation_mib", get_rss_mib(), 1);
        row.set("lifetime_peak_rss_mib", get_lifetime_peak_rss_mib(), 1);
        row.set("commit_after_generation_mib", get_commit_mib(), 1);
        row.set("lifetime_peak_commit_mib", get_lifetime_peak_commit_mib(), 1);
        emit_row(row, options);

        if (succeeded) {
            std::cerr << "Peak RSS: " << peak_rss << " MiB\n";
        }
    }

    return failed_runs == 0 ? 0 : 1;
}
