// kvcalc.hpp — how much KV cache a model needs, and what that leaves you.
//
// Everything is a pure function of the model geometry and the memory budget, so
// it is testable without a GPU. The hybrid-attention case is the one that bites:
// a model with 24 linear-attention layers and 8 full-attention layers carries KV
// for 8 layers, not 32. Counting all of them overestimates by 4x.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace kvcalc {

// Attention geometry. MLA models would add a compressed-latent term; not modelled
// here because the interesting failure mode is the hybrid one.
struct ModelSpec {
    int layers = 32;
    int full_attn_layers = 8;   // layers whose KV grows with context
    int kv_heads = 4;
    int head_dim = 256;
    double dtype_bytes = 2.0;   // bf16/fp16 = 2, fp8 = 1
};

// What the card gives you.
struct Budget {
    double vram_mib = 8151.0;
    double util = 0.85;
    double weights_mib = 3891.0;
    double overhead_mib = 0.0;  // activations + cuda graphs; calibrate once
};

struct Estimate {
    double kv_per_token = 0.0;     // bytes
    double kv_budget_mib = 0.0;    // after weights and overhead
    double max_tokens = 0.0;       // KV cells that fit
    bool fits_any = false;
};

// bytes of KV per token, from the geometry alone
double kv_bytes_per_token(const ModelSpec& spec);

// full estimate for one budget
Estimate estimate(const ModelSpec& spec, const Budget& budget);

// sequences of `context` tokens that fit
double concurrency(const Estimate& est, long long context);

// smallest viable context: 1 sequence must fit
bool fits_context(const Estimate& est, long long context);

struct SweepRow {
    long long context;
    double kv_per_sequence_mib;
    double concurrency;
};

std::vector<SweepRow> sweep(const Estimate& est, const std::vector<long long>& contexts);

// Parse the text-tower hyper-parameters out of a HuggingFace config.json. This is
// a deliberately small scanner: the file is JSON, but pulling in a parser for six
// integers is not worth a dependency.
bool read_config(const std::string& path, ModelSpec& spec, std::string& error);

// Parse "8192", "8k", "1M".
bool parse_size(const std::string& text, long long& out);

std::string human_mib(double mib);

}  // namespace kvcalc
