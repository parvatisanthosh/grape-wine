// Checks the synthetic prompt builder (Finding 29).
//
// Usage: check_prompt_ids MODEL_DIR TOKENS PROMPT_ID [PROMPT_ID ...]
//
// For each prompt id, builds the prompt the old way (target_tokens + 10 body
// repeats, which can exceed the tokenizer's left-truncation limit) and the new
// way (create_exact_token_input in orbit_run.cpp), then reports:
//   - whether the two give the same token ids (they should, unless the old
//     text was truncated);
//   - whether prompts with different ids are distinct (they must be).
// Exit code 0 only if every new prompt is distinct from every other.

#define ORBIT_RUN_NO_MAIN
#include "../orbit_run.cpp"

#include <iostream>
#include <vector>

namespace {

std::vector<std::int64_t> ids_of(const ov::Tensor& tensor) {
    const auto* data = tensor.data<const std::int64_t>();
    return {data, data + tensor.get_shape().at(1)};
}

std::vector<std::int64_t> old_prompt(ov::genai::Tokenizer& tokenizer, std::size_t target,
                                     const std::string& prompt_id) {
    const auto [variant, tag] = split_prompt_id(prompt_id);
    const auto& [opening, body] = PROMPT_VARIANTS.at(variant);
    std::string text = tag.empty() ? opening : "Request " + tag + ". " + opening;
    for (std::size_t repeat = 0; repeat < target + 10; ++repeat) {
        text += body;
    }
    auto ids = ids_of(tokenizer.encode(text, {ov::genai::add_special_tokens(true)}).input_ids);
    ids.resize(std::min(ids.size(), target));
    return ids;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: check_prompt_ids MODEL_DIR TOKENS PROMPT_ID [PROMPT_ID ...]\n";
        return 2;
    }
    ov::genai::Tokenizer tokenizer(argv[1]);
    const std::size_t target = std::stoul(argv[2]);

    std::vector<std::vector<std::int64_t>> fresh;
    bool distinct = true;
    for (int index = 3; index < argc; ++index) {
        const std::string prompt_id = argv[index];
        const auto before = old_prompt(tokenizer, target, prompt_id);
        const auto after = ids_of(create_exact_token_input(tokenizer, target, prompt_id).input_ids);
        std::cout << prompt_id << ": old==new " << (before == after ? "yes" : "NO")
                  << ", old length " << before.size() << "\n";
        for (std::size_t other = 0; other < fresh.size(); ++other) {
            if (fresh[other] == after) {
                std::cout << "  SAME as " << argv[3 + other] << "\n";
                distinct = false;
            }
        }
        fresh.push_back(after);
    }
    std::cout << (distinct ? "OK: all prompts distinct\n" : "FAILED: duplicate prompts\n");
    return distinct ? 0 : 1;
}
