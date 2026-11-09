// Minimal reproduction: an explicit KV-cache precision on the GPU paged-attention
// pipeline. Draft report: notes/upstream_issue_gpu_kv_precision.md.
//
//   repro_gpu_kv_precision MODEL_DIR PRECISION [CACHE_DIR]
//
// PRECISION is default, u8, u4, f16 or f32. With CACHE_DIR, ov::cache_dir is set
// as well, so a compiled-model cache can be ruled in or out. Prints one line:
// OK, or FAILED with the exception text.

#include <iostream>
#include <map>
#include <string>

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: repro_gpu_kv_precision MODEL_DIR PRECISION [CACHE_DIR]\n";
        return 2;
    }
    const std::string model = argv[1];
    const std::string precision = argv[2];

    const std::map<std::string, ov::element::Type> types = {
        {"u8", ov::element::u8}, {"u4", ov::element::u4}, {"f16", ov::element::f16}, {"f32", ov::element::f32}};

    ov::AnyMap properties;
    if (precision != "default") {
        if (types.count(precision) == 0) {
            std::cerr << "Unknown precision " << precision << "\n";
            return 2;
        }
        properties.insert(ov::hint::kv_cache_precision(types.at(precision)));
    }
    if (argc == 4) {
        properties.insert(ov::cache_dir(argv[3]));
    }

    try {
        ov::genai::LLMPipeline pipeline(model, "GPU", properties);
        ov::genai::GenerationConfig config;
        config.max_new_tokens = 8;
        const std::string text = pipeline.generate("The capital of France is", config);
        std::cout << "OK precision=" << precision << " output=" << text << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cout << "FAILED precision=" << precision << ": " << error.what() << "\n";
        return 1;
    }
}
