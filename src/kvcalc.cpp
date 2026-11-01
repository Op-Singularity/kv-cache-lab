#include "kvcalc.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace kvcalc {
namespace {
constexpr double kMib = 1024.0 * 1024.0;

// first integer value after "key"
bool find_int(const std::string& text, const std::string& key, long long& out) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = text.find(needle);
    if (pos == std::string::npos) return false;
    pos = text.find(':', pos + needle.size());
    if (pos == std::string::npos) return false;
    ++pos;
    while (pos < text.size() && !(std::isdigit(static_cast<unsigned char>(text[pos])) || text[pos] == '-')) ++pos;
    if (pos >= text.size()) return false;
    out = std::atoll(text.c_str() + pos);
    return true;
}
}  // namespace

double kv_bytes_per_token(const ModelSpec& spec) {
    // K and V, per full-attention layer, per kv head, per head dim
    return static_cast<double>(spec.full_attn_layers) * 2.0 *
           static_cast<double>(spec.kv_heads) * static_cast<double>(spec.head_dim) *
           spec.dtype_bytes;
}

Estimate estimate(const ModelSpec& spec, const Budget& budget) {
    Estimate est;
    est.kv_per_token = kv_bytes_per_token(spec);
    est.kv_budget_mib = budget.vram_mib * budget.util - budget.weights_mib - budget.overhead_mib;
    if (est.kv_budget_mib <= 0.0 || est.kv_per_token <= 0.0) {
        est.max_tokens = 0.0;
        est.fits_any = false;
        return est;
    }
    est.max_tokens = est.kv_budget_mib * kMib / est.kv_per_token;
    est.fits_any = est.max_tokens >= 1.0;
    return est;
}

double concurrency(const Estimate& est, long long context) {
    if (context <= 0) return 0.0;
    return est.max_tokens / static_cast<double>(context);
}

bool fits_context(const Estimate& est, long long context) { return concurrency(est, context) >= 1.0; }

std::vector<SweepRow> sweep(const Estimate& est, const std::vector<long long>& contexts) {
    std::vector<SweepRow> rows;
    rows.reserve(contexts.size());
    for (long long ctx : contexts) {
        rows.push_back(SweepRow{ctx, static_cast<double>(ctx) * est.kv_per_token / kMib, concurrency(est, ctx)});
    }
    return rows;
}

bool read_config(const std::string& path, ModelSpec& spec, std::string& error) {
    std::ifstream file(path);
    if (!file) {
        error = "cannot open " + path;
        return false;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();

    // multimodal configs keep the text tower under "text_config"
    const size_t text_config = text.find("\"text_config\"");
    if (text_config != std::string::npos) text = text.substr(text_config);

    long long value = 0;
    if (find_int(text, "num_hidden_layers", value)) spec.layers = static_cast<int>(value);
    if (find_int(text, "num_key_value_heads", value)) spec.kv_heads = static_cast<int>(value);
    if (find_int(text, "head_dim", value)) spec.head_dim = static_cast<int>(value);

    // full-attention layers: count occurrences of "full_attention" in the layer
    // pattern, which newer configs spell out explicitly
    int full = 0;
    const std::string pattern = "\"full_attention\"";
    for (size_t pos = text.find(pattern); pos != std::string::npos; pos = text.find(pattern, pos + 1)) ++full;
    if (full > 0) spec.full_attn_layers = full;

    return true;
}

bool parse_size(const std::string& text, long long& out) {
    if (text.empty()) return false;
    char suffix = 0;
    std::string number = text;
    const char last = text.back();
    if (last == 'k' || last == 'K' || last == 'm' || last == 'M') {
        suffix = static_cast<char>(std::tolower(last));
        number = text.substr(0, text.size() - 1);
    }
    try {
        const long long value = std::stoll(number);
        if (suffix == 'k') out = value * 1024;
        else if (suffix == 'm') out = value * 1024 * 1024;
        else out = value;
    } catch (...) {
        return false;
    }
    return out > 0;
}

std::string human_mib(double mib) {
    char buffer[64];
    if (mib >= 1024.0) std::snprintf(buffer, sizeof buffer, "%.2f GiB", mib / 1024.0);
    else std::snprintf(buffer, sizeof buffer, "%.0f MiB", mib);
    return std::string(buffer);
}

}  // namespace kvcalc
