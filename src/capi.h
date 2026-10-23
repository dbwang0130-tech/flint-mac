// flint 引擎的 C 接口（供 Rust 服务通过 FFI 调用）。单会话、非线程安全：调用方需在同一线程串行使用。
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct flint_engine flint_engine;

typedef struct {
    const char *model_path;    // GGUF 路径
    const char *metallib_dir;  // 含 flint.metallib / fused.metallib / prefill.metallib 的目录
    double pool_gb;            // 0 = 权重全部常驻；>0 = 专家流式加载的槽池大小（GB）；<0 = 自动
    int32_t n_ctx;             // 上下文长度（全注意力层 KV 按此分配）
} flint_config;

// 失败返回 NULL，错误信息写入 err（可为 NULL）
flint_engine *flint_open(const flint_config *cfg, char *err, int32_t err_len);
void flint_close(flint_engine *e);

int32_t flint_n_ctx(const flint_engine *e);
int32_t flint_n_vocab(void);
int32_t flint_pos(const flint_engine *e);

// 回退到位置 pos（≤ 当前位置）。SWA 层环形缓冲只保留最近 1024 个位置：
// 回退超过 511 个 token 时，调用方需从 max(0, pos-512) 重新 prefill 才能保证结果正确
int32_t flint_truncate(flint_engine *e, int32_t pos);

// 从当前位置起处理 n 个 token；返回最后一个位置贪心的下一个 token，失败返回 -1
int32_t flint_prefill(flint_engine *e, const int32_t *tokens, int32_t n, char *err, int32_t err_len);
// 处理一个 token；返回贪心的下一个 token，失败返回 -1
int32_t flint_decode(flint_engine *e, int32_t token, char *err, int32_t err_len);

// 最近一次 prefill / decode 的 logits（flint_n_vocab() 个 float），到下一次调用前有效
const float *flint_logits(const flint_engine *e);

#ifdef __cplusplus
}
#endif
