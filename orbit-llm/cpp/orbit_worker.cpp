// orbit_worker: a long-lived OpenVINO GenAI process that keeps models loaded.
//
// orbit_run loads one configuration, measures it and exits, so every request
// pays a full model load and the prefix cache dies with the process. The
// worker instead stays alive and keeps every loaded pipeline in memory. That
// lets ORBIT measure, instead of simulate:
//   - the cost of staying on a resident model (no reload),
//   - the real cost of a switch (unload + load),
//   - prefix-cache reuse across separate requests,
//   - how much memory is actually returned when a model is unloaded.
//
// Protocol: the caller writes one command per line on stdin; the worker
// answers every command with exactly one JSON line on stdout. Human-readable
// progress goes to stderr. On start-up it writes a {"command": "ready"} line.
//
//   load SLOT --model DIR [orbit_run configuration flags]
//   generate SLOT --prompt-tokens N --output-tokens N [--prompt-id ID]
//   unload SLOT
//   status
//   quit
//
// SLOT is any name the caller chooses, e.g. "INT4@gpu". Several slots can be
// loaded at the same time. Loading into an occupied slot is refused: the
// caller must unload first, so a switch is measured as unload + load.

// Reuse every helper in orbit_run.cpp (argument parsing, pipeline properties,
// exact-length prompts, telemetry, JSON rows) without its main().
#define ORBIT_RUN_NO_MAIN
#include "orbit_run.cpp"

namespace {

// One loaded model: its configuration, its pipeline and its prepared prompts.
struct Slot {
    Options options;
    std::unique_ptr<ov::genai::LLMPipeline> pipeline;
    // Tokenized prompts, keyed by (prompt id, length), built once per slot.
    std::map<std::pair<std::string, std::size_t>, ov::genai::TokenizedInputs> prompts;
    int generations = 0;
};

using Slots = std::map<std::string, Slot>;

// Split a command line into words. Double quotes allow spaces in a word.
std::vector<std::string> split_command(const std::string& line) {
    std::vector<std::string> words;
    std::string current;
    bool quoted = false;
    bool has_word = false;

    for (const char character : line) {
        if (character == '"') {
            quoted = !quoted;
            has_word = true;
            continue;
        }
        if (!quoted && (character == ' ' || character == '\t' || character == '\r')) {
            if (has_word) {
                words.push_back(current);
                current.clear();
                has_word = false;
            }
            continue;
        }
        current += character;
        has_word = true;
    }

    if (quoted) {
        throw std::invalid_argument("Unclosed quote in command");
    }
    if (has_word) {
        words.push_back(current);
    }
    return words;
}

// Parse load flags with orbit_run's own parser, so the worker accepts and
// validates exactly the same configuration options. orbit_run requires a
// prompt and output length, which mean nothing at load time, so placeholder
// values go first; any flag the caller passes later overrides them.
Options parse_slot_options(const std::vector<std::string>& words, std::size_t first) {
    std::vector<std::string> arguments = {"orbit_worker", "--prompt-tokens", "2", "--output-tokens", "1"};
    arguments.insert(arguments.end(), words.begin() + static_cast<std::ptrdiff_t>(first), words.end());

    std::vector<char*> argv;
    for (auto& argument : arguments) {
        argv.push_back(argument.data());
    }
    return parse_arguments(static_cast<int>(argv.size()), argv.data());
}

// Every reply starts with these fields.
JsonRow make_reply(const std::string& command, const std::string& slot) {
    JsonRow row;
    row.set("timestamp", now_iso8601());
    row.set("executor", "cpp_worker");
    row.set("command", command);
    row.set("slot", slot);
    return row;
}

void add_memory(JsonRow& row, const std::string& when) {
    row.set("rss_" + when + "_mib", get_rss_mib(), 1);
    row.set("commit_" + when + "_mib", get_commit_mib(), 1);
    row.set("sys_available_ram_" + when + "_mib", get_available_ram_mib(), 1);
}

// One JSON line per reply. std::endl flushes, so the caller never waits on a
// reply that is sitting in an output buffer.
void send(const JsonRow& row) {
    std::cout << row.str() << std::endl;
}

std::string slot_names(const Slots& slots) {
    std::string names;
    for (const auto& [name, slot] : slots) {
        if (!names.empty()) {
            names += ",";
        }
        names += name;
    }
    return names;
}

// ---------------------------------------------------------------------------
// Commands

void handle_load(const std::vector<std::string>& words, Slots& slots) {
    if (words.size() < 2) {
        throw std::invalid_argument("usage: load SLOT --model DIR [options]");
    }
    const std::string name = words[1];
    if (slots.count(name) != 0) {
        throw std::invalid_argument("slot '" + name + "' is already loaded; unload it first");
    }

    Slot slot;
    slot.options = parse_slot_options(words, 2);
    const Options& options = slot.options;

    JsonRow row = make_reply("load", name);
    row.set("label", options.label);
    row.set("model_path", options.model);
    row.set("device", options.device);
    row.set("backend", options.backend);
    row.set("cores", options.cores);
    row.set("threads", options.threads);
    row.set("kv_precision", options.kv_precision);
    row.set("prefix_caching", options.prefix_caching == "on");
    row.set("cache_dir", options.cache_dir);
    row.set("slots_already_loaded", slots.size());
    add_memory(row, "before");

    std::cerr << "Loading " << name << " (" << options.label << " on " << options.device << ") ...\n";
    const auto start = std::chrono::steady_clock::now();
    slot.pipeline = std::make_unique<ov::genai::LLMPipeline>(
        options.model, options.device, build_pipeline_properties(options));
    const double construction_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

    row.set("pipeline_construction_ms", construction_ms);
    add_memory(row, "after");
    row.set("lifetime_peak_commit_mib", get_lifetime_peak_commit_mib(), 1);

    slots.emplace(name, std::move(slot));
    row.set("loaded_slots", slot_names(slots));
    row.set("status", "success");
    std::cerr << "Loaded " << name << " in " << construction_ms << " ms\n";
    send(row);
}

void handle_generate(const std::vector<std::string>& words, Slots& slots) {
    if (words.size() < 2) {
        throw std::invalid_argument("usage: generate SLOT --prompt-tokens N --output-tokens N [--prompt-id ID]");
    }
    const std::string name = words[1];
    const auto found = slots.find(name);
    if (found == slots.end()) {
        throw std::invalid_argument("slot '" + name + "' is not loaded");
    }
    Slot& slot = found->second;

    std::size_t prompt_tokens = 0;
    std::size_t output_tokens = 0;
    std::string prompt_id = "run_1";
    for (std::size_t index = 2; index < words.size(); index += 2) {
        if (index + 1 >= words.size()) {
            throw std::invalid_argument("Missing value for " + words[index]);
        }
        const std::string& flag = words[index];
        const std::string& value = words[index + 1];
        if (flag == "--prompt-tokens") prompt_tokens = parse_size(flag, value);
        else if (flag == "--output-tokens") output_tokens = parse_size(flag, value);
        else if (flag == "--prompt-id") prompt_id = value;
        else throw std::invalid_argument("Unknown generate option " + flag);
    }
    if (prompt_tokens < 2) throw std::invalid_argument("--prompt-tokens must be at least 2");
    if (output_tokens < 1) throw std::invalid_argument("--output-tokens must be at least 1");
    if (!is_known_prompt_id(prompt_id)) {
        throw std::invalid_argument("Unknown --prompt-id '" + prompt_id +
                                    "' (use warmup or run_1 ... run_5, optionally with #tag)");
    }

    // The slot's configuration plus this request's sizes, so the row has the
    // same fields as an orbit_run row and can join the same dataset.
    Options request = slot.options;
    request.prompt_tokens = prompt_tokens;
    request.output_tokens = output_tokens;
    request.prompt_mode = "worker";

    JsonRow row = make_base_row(request);
    row.set("executor", "cpp_worker");
    row.set("command", "generate");
    row.set("slot", name);
    row.set("prompt_id", prompt_id);
    // 1 = first request after this slot was loaded.
    row.set("generation_index", slot.generations + 1);
    row.set("loaded_slots", slot_names(slots));
    row.set("sys_available_ram_mib", get_available_ram_mib(), 1);
    const auto power = orbit::get_power_state();
    if (power.known) {
        row.set("on_ac_power", power.on_ac);
    } else {
        row.set_null("on_ac_power");
    }
    row.set("battery_percent", power.battery_percent);
    row.set("sys_cpu_busy_percent", sample_system_cpu_busy_percent(), 1);
    row.set("status", "failed");
    row.set("error", "");

    const auto key = std::make_pair(prompt_id, prompt_tokens);
    if (slot.prompts.count(key) == 0) {
        auto tokenizer = slot.pipeline->get_tokenizer();
        slot.prompts.emplace(key, create_exact_token_input(tokenizer, prompt_tokens, prompt_id));
    }

    ov::genai::GenerationConfig config;
    config.max_new_tokens = output_tokens;
    config.ignore_eos = true;
    config.do_sample = false;
    config.apply_chat_template = false;

    std::cerr << "Generating on " << name << ": " << prompt_tokens << " -> " << output_tokens
              << " tokens (" << prompt_id << ")\n";

    PeakMemorySampler sampler;
    RunTelemetry telemetry;
    try {
        const auto wall_start = std::chrono::steady_clock::now();
        const auto result = slot.pipeline->generate(slot.prompts.at(key), config);
        const double wall_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_start).count();

        auto metrics = result.perf_metrics;
        const std::size_t input_count = metrics.get_num_input_tokens();
        const std::size_t output_count = metrics.get_num_generated_tokens();
        if (input_count != prompt_tokens || output_count != output_tokens) {
            throw std::runtime_error("Expected " + std::to_string(prompt_tokens) + "/" +
                                     std::to_string(output_tokens) + " tokens, OpenVINO reported " +
                                     std::to_string(input_count) + "/" + std::to_string(output_count));
        }

        row.set("actual_input_tokens", input_count);
        row.set("actual_output_tokens", output_count);
        row.set("wall_time_ms", wall_ms);
        row.set("generation_ms", static_cast<double>(metrics.get_generate_duration().mean));
        row.set("ttft_ms", static_cast<double>(metrics.get_ttft().mean));
        row.set("tpot_ms_per_token", static_cast<double>(metrics.get_tpot().mean));
        row.set("throughput_tokens_per_second", static_cast<double>(metrics.get_throughput().mean));
        row.set("status", "success");
        std::cerr << "TTFT: " << metrics.get_ttft().mean << " ms, TPOT: " << metrics.get_tpot().mean
                  << " ms/token\n";
    } catch (const std::exception& error) {
        // A failed generation is data: record it and keep the worker alive.
        row.set("error", error.what());
        std::cerr << "GENERATE FAILED: " << error.what() << "\n";
    }

    const double peak_rss = sampler.stop();
    telemetry.finish(row, get_base_cpu_mhz());
    row.set("sampled_peak_rss_mib", peak_rss, 1);
    row.set("rss_after_generation_mib", get_rss_mib(), 1);
    row.set("commit_after_generation_mib", get_commit_mib(), 1);
    row.set("lifetime_peak_commit_mib", get_lifetime_peak_commit_mib(), 1);

    ++slot.generations;
    send(row);
}

void handle_unload(const std::vector<std::string>& words, Slots& slots) {
    if (words.size() != 2) {
        throw std::invalid_argument("usage: unload SLOT");
    }
    const std::string name = words[1];
    const auto found = slots.find(name);
    if (found == slots.end()) {
        throw std::invalid_argument("slot '" + name + "' is not loaded");
    }

    JsonRow row = make_reply("unload", name);
    row.set("generations_served", found->second.generations);
    const double rss_before = get_rss_mib();
    const double commit_before = get_commit_mib();
    add_memory(row, "before");

    const auto start = std::chrono::steady_clock::now();
    slots.erase(found);  // destroys the pipeline and its prompts
    const double unload_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

    row.set("unload_ms", unload_ms);
    add_memory(row, "after");
    // How much memory the process actually got back. OpenVINO's allocators
    // may keep some of it, so this need not equal what the load took.
    row.set("freed_rss_mib", rss_before - get_rss_mib(), 1);
    row.set("freed_commit_mib", commit_before - get_commit_mib(), 1);
    row.set("loaded_slots", slot_names(slots));
    row.set("status", "success");
    std::cerr << "Unloaded " << name << " in " << unload_ms << " ms\n";
    send(row);
}

void handle_status(const Slots& slots) {
    JsonRow row = make_reply("status", "");
    row.set("loaded_slots", slot_names(slots));
    row.set("slot_count", slots.size());
    add_memory(row, "now");
    row.set("status", "success");
    send(row);
}

}  // namespace

int main() {
    Slots slots;

    JsonRow ready = make_reply("ready", "");
    ready.set("genai_version", ov::genai::get_version().buildNumber);
    ready.set("sys_total_ram_mib", get_total_ram_mib(), 1);
    add_memory(ready, "now");
    ready.set("status", "success");
    send(ready);
    std::cerr << "orbit_worker ready. Commands: load, generate, unload, status, quit\n";

    std::string line;
    while (std::getline(std::cin, line)) {
        std::vector<std::string> words;
        std::string command;
        try {
            words = split_command(line);
            if (words.empty()) {
                continue;  // blank lines get no reply
            }
            command = words[0];

            if (command == "quit") {
                JsonRow row = make_reply("quit", "");
                row.set("status", "success");
                send(row);
                break;
            } else if (command == "load") {
                handle_load(words, slots);
            } else if (command == "generate") {
                handle_generate(words, slots);
            } else if (command == "unload") {
                handle_unload(words, slots);
            } else if (command == "status") {
                handle_status(slots);
            } else {
                throw std::invalid_argument("Unknown command '" + command +
                                            "' (use load, generate, unload, status, quit)");
            }
        } catch (const std::exception& error) {
            // Every command gets exactly one reply, errors included, so the
            // caller can never get out of step with the worker.
            JsonRow row = make_reply(command.empty() ? "unknown" : command, words.size() > 1 ? words[1] : "");
            row.set("status", "error");
            row.set("error", error.what());
            send(row);
            std::cerr << "ERROR: " << error.what() << "\n";
        }
    }

    slots.clear();
    return 0;
}