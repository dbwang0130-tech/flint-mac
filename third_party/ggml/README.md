# 引用的 ggml 文件（MIT）

来自 [llama.cpp](https://github.com/ggml-org/llama.cpp)（master, 2026-10-23），许可证见同目录 `LICENSE`（MIT, The ggml authors）：

| 文件 | 上游路径 | 用途 |
|---|---|---|
| `ggml-common.h` | `ggml/src/ggml-common.h` | 量化块结构与 IQ 码本 |
| `ggml-metal-impl.h` | `ggml/src/ggml-metal/ggml-metal-impl.h` | `common.h` 依赖的常量 |
| `dequantize.h` | `ggml/src/ggml-metal/kernels/dequantize.h` | Metal 16 元素解码函数 |
| `common.h` | `ggml/src/ggml-metal/kernels/common.h` | Metal 公共定义 |

`src/ref/dequant_gen.c` 由 `ggml/src/ggml-quants.c` 原样抽取的 `dequantize_row_*` 函数构成（同为 MIT）。
`src/metal/prefill.metal` 的批量矩阵乘按 `ggml/src/ggml-metal/kernels/mul_mm.metal` 的 tensor-op 路径改写。
