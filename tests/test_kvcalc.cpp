// Tests are plain asserts: no framework, so `make test` works with a compiler and
// nothing else. The numbers come from a real machine (see README).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "kvcalc.hpp"

namespace {

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (condition) {
        std::printf("  ok   %s\n", what.c_str());
    } else {
        std::printf("  FAIL %s\n", what.c_str());
        ++failures;
    }
}

void expect_near(double got, double want, double tolerance, const std::string& what) {
    const bool ok = std::fabs(got - want) <= tolerance;
    if (ok) {
        std::printf("  ok   %s (%.4f)\n", what.c_str(), got);
    } else {
        std::printf("  FAIL %s: got %.4f, want %.4f +- %.4f\n", what.c_str(), got, want, tolerance);
        ++failures;
    }
}

}  // namespace

int main() {
    // Qwen3.5-4B on an RTX 5060 Laptop: 32 layers, 8 of them full attention,
    // 4 kv heads, head_dim 256, bf16 cache.
    kvcalc::ModelSpec spec;
    spec.layers = 32;
    spec.full_attn_layers = 8;
    spec.kv_heads = 4;
    spec.head_dim = 256;
    spec.dtype_bytes = 2.0;

    std::printf("geometry\n");
    expect_near(kvcalc::kv_bytes_per_token(spec), 32768.0, 0.5, "32 KiB per token");

    // Counting every layer would give 4x this, which is the mistake the tool exists
    // to prevent.
    kvcalc::ModelSpec naive = spec;
    naive.full_attn_layers = naive.layers;
    expect_near(kvcalc::kv_bytes_per_token(naive) / kvcalc::kv_bytes_per_token(spec), 4.0, 0.001,
                "assuming all layers would overestimate 4x");

    // Budget from the measured machine: 8151 MiB total, util 0.85, 4B AWQ weights.
    kvcalc::Budget budget;
    budget.vram_mib = 8151.0;
    budget.util = 0.85;
    budget.weights_mib = 3891.0;

    std::printf("\nbudget\n");
    const auto est = kvcalc::estimate(spec, budget);
    expect_near(est.kv_budget_mib, 6928.35 - 3891.0, 1.0, "KV budget after weights");
    expect(est.fits_any, "some KV fits");
    expect_near(est.max_tokens, 97195.0, 50.0, "max KV tokens without overhead");

    // The engine reported 76_933 tokens. The difference is a constant ~633 MiB of
    // activation and CUDA-graph memory; feed it back in and the formula should land
    // within a rounding error.
    kvcalc::Budget calibrated = budget;
    calibrated.overhead_mib = 633.0;
    const auto cal = kvcalc::estimate(spec, calibrated);
    const double error_pct = std::fabs(cal.max_tokens - 76933.0) / 76933.0 * 100.0;
    expect(error_pct < 0.5, "calibrated formula within 0.5 % of the engine's 76 933 tokens");
    std::printf("       (error %.3f %%)\n", error_pct);

    std::printf("\nconcurrency\n");
    expect_near(kvcalc::concurrency(cal, 32768), 2.35, 0.05, "2.35 sequences at 32 K");
    expect(!kvcalc::fits_context(cal, 131072), "128 K on this card does not fit");
    expect(kvcalc::fits_context(cal, 4096), "4 K fits");

    std::printf("\nparsing\n");
    long long value = 0;
    expect(kvcalc::parse_size("8192", value) && value == 8192, "8192");
    expect(kvcalc::parse_size("8k", value) && value == 8192, "8k");
    expect(kvcalc::parse_size("1M", value) && value == 1048576, "1M");
    expect(!kvcalc::parse_size("abc", value), "rejects garbage");

    // Fibonacci-style sanity: doubling the context halves the concurrency.
    expect_near(kvcalc::concurrency(cal, 8192) / kvcalc::concurrency(cal, 16384), 2.0, 0.001,
                "concurrency halves when context doubles");

    std::printf("\n%s\n", failures == 0 ? "all tests passed" : "TESTS FAILED");
    return failures == 0 ? 0 : 1;
}
