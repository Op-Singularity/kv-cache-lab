// kvcalc — KV cache budget calculator
//
//   kvcalc --layers 32 --full-attn 8 --kv-heads 4 --head-dim 256
//          --vram-mib 8151 --util 0.85 --weights-mib 3891
//          --check 76933 --sweep 4k,8k,32k,128k
//
//   kvcalc --from-config ~/models/qwen3.5-9b-awq/config.json --weights-mib 8660 --util 0.86
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "kvcalc.hpp"

namespace {

void usage() {
    std::printf(
        "usage: kvcalc [options]\n\n"
        "model geometry\n"
        "  --from-config PATH   read text-tower hyper-parameters from a config.json\n"
        "  --layers N           total layers (default 32)\n"
        "  --full-attn N        layers carrying a growing KV cache (default 8)\n"
        "  --kv-heads N         key/value heads (default 4)\n"
        "  --head-dim N         head dimension (default 256)\n"
        "  --dtype bf16|fp8     KV element size (default bf16 = 2 bytes)\n\n"
        "budget\n"
        "  --vram-mib N         total VRAM (default 8151)\n"
        "  --util F             fraction the engine may use (default 0.85)\n"
        "  --weights-mib N      resident weights (default 3891)\n"
        "  --overhead-mib N     activations + graphs (default 0; calibrate once)\n\n"
        "checks\n"
        "  --check N            tokens reported by an engine, for validation\n"
        "  --sweep LIST         contexts to tabulate, e.g. 4k,8k,32k,128k\n"
        "  -h, --help\n");
}

}  // namespace

int main(int argc, char** argv) {
    kvcalc::ModelSpec spec;
    kvcalc::Budget budget;
    kvcalc::Estimate unused_estimate;
    (void)unused_estimate;

    long long check_tokens = 0;
    std::vector<long long> contexts = {2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    std::string config_path;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next_number = [&]() -> double {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", arg.c_str());
                std::exit(2);
            }
            return std::atof(argv[++i]);
        };

        if (arg == "-h" || arg == "--help") {
            usage();
            return 0;
        } else if (arg == "--from-config") {
            if (i + 1 >= argc) { std::fprintf(stderr, "--from-config needs a path\n"); return 2; }
            config_path = argv[++i];
        } else if (arg == "--layers") {
            spec.layers = static_cast<int>(next_number());
        } else if (arg == "--full-attn") {
            spec.full_attn_layers = static_cast<int>(next_number());
        } else if (arg == "--kv-heads") {
            spec.kv_heads = static_cast<int>(next_number());
        } else if (arg == "--head-dim") {
            spec.head_dim = static_cast<int>(next_number());
        } else if (arg == "--dtype") {
            if (i + 1 >= argc) { std::fprintf(stderr, "--dtype needs bf16 or fp8\n"); return 2; }
            const std::string kind = argv[++i];
            if (kind == "fp8") spec.dtype_bytes = 1.0;
            else if (kind == "bf16" || kind == "fp16") spec.dtype_bytes = 2.0;
            else { std::fprintf(stderr, "unknown dtype %s\n", kind.c_str()); return 2; }
        } else if (arg == "--vram-mib") {
            budget.vram_mib = next_number();
        } else if (arg == "--util") {
            budget.util = next_number();
        } else if (arg == "--weights-mib") {
            budget.weights_mib = next_number();
        } else if (arg == "--overhead-mib") {
            budget.overhead_mib = next_number();
        } else if (arg == "--check") {
            check_tokens = static_cast<long long>(next_number());
        } else if (arg == "--sweep") {
            if (i + 1 >= argc) { std::fprintf(stderr, "--sweep needs a list\n"); return 2; }
            contexts.clear();
            std::string list = argv[++i];
            size_t start = 0;
            while (start <= list.size()) {
                const size_t comma = list.find(',', start);
                const std::string piece = list.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                long long value = 0;
                if (kvcalc::parse_size(piece, value)) contexts.push_back(value);
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
        } else {
            std::fprintf(stderr, "unknown option %s\n", arg.c_str());
            return 2;
        }
    }

    if (!config_path.empty()) {
        std::string error;
        if (!kvcalc::read_config(config_path, spec, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        std::printf("from %s: layers=%d full_attn=%d kv_heads=%d head_dim=%d\n\n", config_path.c_str(),
                    spec.layers, spec.full_attn_layers, spec.kv_heads, spec.head_dim);
    }

    const kvcalc::Estimate est = kvcalc::estimate(spec, budget);

    std::printf("model\n");
    std::printf("  layers            : %d (%d full attention, %d linear)\n", spec.layers,
                spec.full_attn_layers, spec.layers - spec.full_attn_layers);
    std::printf("  kv heads x dim    : %d x %d\n", spec.kv_heads, spec.head_dim);
    std::printf("  kv element bytes  : %.0f\n", spec.dtype_bytes);
    std::printf("\nkv per token        : %.0f bytes (%.1f KiB)\n", est.kv_per_token,
                est.kv_per_token / 1024.0);
    std::printf("vram x util         : %.0f x %.2f = %.0f MiB\n", budget.vram_mib, budget.util,
                budget.vram_mib * budget.util);
    std::printf("minus weights       : %.0f MiB\n", budget.weights_mib);
    if (budget.overhead_mib > 0.0) std::printf("minus overhead      : %.0f MiB\n", budget.overhead_mib);
    std::printf("left for kv         : %s\n", kvcalc::human_mib(est.kv_budget_mib).c_str());
    if (!est.fits_any) {
        std::printf("\nresult              : weights do not fit; no KV budget left\n");
        return 1;
    }
    std::printf("max kv tokens       : %.0f\n", est.max_tokens);

    if (!contexts.empty()) {
        std::printf("\n%10s %14s %14s %s\n", "context", "kv/sequence", "concurrency", "");
        for (const auto& row : kvcalc::sweep(est, contexts)) {
            std::printf("%10lld %14s %14.2f %s\n", row.context,
                        kvcalc::human_mib(row.kv_per_sequence_mib).c_str(), row.concurrency,
                        row.concurrency >= 1.0 ? "" : "does not fit");
        }
    }

    if (check_tokens > 0) {
        const double implied_kv = static_cast<double>(check_tokens) * est.kv_per_token / (1024.0 * 1024.0);
        const double implied_other = budget.vram_mib * budget.util - budget.weights_mib - implied_kv;
        const double error = (est.max_tokens - static_cast<double>(check_tokens)) /
                             static_cast<double>(check_tokens) * 100.0;
        std::printf("\nvalidation against %lld tokens reported by the engine\n", check_tokens);
        std::printf("  implied kv        : %s\n", kvcalc::human_mib(implied_kv).c_str());
        std::printf("  implied other     : %s (activations + graphs)\n",
                    kvcalc::human_mib(implied_other).c_str());
        std::printf("  formula error     : %+.2f %%\n", error);
        std::printf("  recalibrate with  : --overhead-mib %.0f\n", implied_other);
    }

    return 0;
}
