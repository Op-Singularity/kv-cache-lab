# kv-cache-lab

[![build](https://github.com/Op-Singularity/kv-cache-lab/actions/workflows/build.yml/badge.svg)](https://github.com/Op-Singularity/kv-cache-lab/actions/workflows/build.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)

A small C++17 tool that answers one question before you deploy anything: **how
much KV cache does this model need, and how much context can this card actually
serve?**

The answer is usually wrong in the same way, because most models are no longer
plain transformers.

```
$ kvcalc --layers 32 --full-attn 8 --kv-heads 4 --head-dim 256 \
         --vram-mib 8151 --util 0.85 --weights-mib 3891 --check 76933

model
  layers            : 32 (8 full attention, 24 linear)
  kv heads x dim    : 4 x 256
  kv element bytes  : 2

kv per token        : 32768 bytes (32.0 KiB)
vram x util         : 8151 x 0.85 = 6928 MiB
minus weights       : 3891 MiB
left for kv         : 2.97 GiB
max kv tokens       : 97195

   context    kv/sequence    concurrency
      4096        128 MiB          23.73
      8192        256 MiB          11.86
     32768       1.00 GiB           2.97
    131072       4.00 GiB           0.74  does not fit

validation against 76933 tokens reported by the engine
  implied kv        : 2.35 GiB
  implied other     : 633 MiB (activations + graphs)
  formula error     : +26.34 %
  recalibrate with  : --overhead-mib 633
```

## Why this exists

Two mistakes are common when sizing a deployment:

**1. Counting every layer.** Hybrid architectures interleave full attention with
linear attention (GDN, Mamba-style state). Only the full-attention layers carry a
cache that grows with context. Qwen3.5-4B has 32 layers and **8** of them are full
attention, so counting 32 overestimates by 4×. The test suite asserts that factor
explicitly.

**2. Trusting the advertised context window.** A model advertising 262 K is making
a statement about its architecture, not about your card. On the machine above,
128 K does not fit even as a single sequence:

```
131072  →  4.00 GiB per sequence  →  0.74 sequences   does not fit
```

## The validation, which is the useful part

The formula is trivial. What makes it trustworthy is that it can be checked
against a real engine:

| Quantity | Value |
|---|---:|
| `vram × util` | 6928 MiB |
| weights (4B AWQ) | 3891 MiB |
| left for KV | 3037 MiB |
| ÷ 32 KiB per token | 97 195 tokens |
| vLLM reported | **76 933 tokens** |
| implied overhead (activations + CUDA graphs) | 633 MiB |
| error after feeding that back in | **0.008 %** |

The 633 MiB is real, roughly constant, and the reason a naive
`weights + KV ≤ VRAM` check is optimistic by about 10 %. Calibrate it once per
runtime and the formula predicts the engine.

## Build

```bash
make          # g++ -std=c++17, no dependencies
make test     # 14 assertions, including the 4x hybrid-attention check
make run      # the invocation above
```

CI builds with both `g++` and `clang++`, runs the assertions, exercises the CLI,
and repeats the build with `-Werror`.

## Reading a config directly

```bash
kvcalc --from-config ~/models/qwen3.5-9b-awq/config.json \
       --weights-mib 8660 --util 0.86 --sweep 4k,8k,16k
```

`--from-config` pulls `num_hidden_layers`, `num_key_value_heads`, `head_dim` from
the text tower (multimodal configs nest it under `text_config`) and counts the
full-attention layers from the `layer_types` pattern when the config spells it
out.

## What is not modelled

| Case | Status |
|---|---|
| GQA (`kv_heads < heads`) | yes, that is the `kv_heads` knob |
| Hybrid attention | yes, that is the whole point |
| MLA (compressed latent, DeepSeek-style) | no — needs a different per-token term |
| Sliding-window attention | no — would cap the per-layer cost |
| Paged / prefix-shared cache | no — this is the worst case |

## Files

```
src/kvcalc.hpp        geometry and budget types, pure functions
src/kvcalc.cpp        the arithmetic plus a tiny config scanner
src/main.cpp          CLI, --sweep and --check
tests/test_kvcalc.cpp assertions, no test framework
.github/workflows/    g++ + clang++ matrix, CLI smoke test, -Werror build
```

## License

MIT.
