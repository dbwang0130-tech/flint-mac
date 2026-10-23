// CPU 参考解码：复用 ggml 的块定义、码本与 dequantize_row_*（MIT, ggml authors）。
// dequant_gen.c 由 ggml-quants.c 原样抽取，勿手改。
#include <assert.h>
#include <stdint.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#define GGML_RESTRICT restrict

static inline float fp16_to_fp32(ggml_half h) {
    _Float16 f;
    memcpy(&f, &h, sizeof(f));
    return (float)f;
}
#define GGML_FP16_TO_FP32(x) fp16_to_fp32(x)

#include "dequant_gen.c"
