// Flint 的固定结构（host 与 Metal 共用）。引擎只支持这一个模型：加载时校验 GGUF 与这里一致。
#pragma once

#ifdef __METAL_VERSION__
#define FL_CONST constant constexpr
#else
#define FL_CONST constexpr
#endif

FL_CONST int N_LAYER = 49;
FL_CONST int N_DENSE = 2;  // 层 0、1 为 dense FFN
FL_CONST int N_EMBD = 2048;
FL_CONST int N_HEAD = 32;
FL_CONST int N_HEAD_KV = 4;
FL_CONST int HEAD_DIM = 128;
FL_CONST int GQA = N_HEAD / N_HEAD_KV;  // 8
FL_CONST int Q_DIM = N_HEAD * HEAD_DIM;  // 4096
FL_CONST int KV_DIM = N_HEAD_KV * HEAD_DIM;  // 512
FL_CONST int N_FF_DENSE = 6144;
FL_CONST int N_FF_EXP = 512;
FL_CONST int N_FF_SH = 1024;
FL_CONST int N_EXPERT = 256;
FL_CONST int N_USED = 6;
FL_CONST int N_VOCAB = 150144;
FL_CONST int N_SWA = 513;  // 含当前 token
FL_CONST float RMS_EPS = 1e-6f;
FL_CONST float ROPE_THETA = 10000.0f;

// prefill 分块上限；SWA 环形缓冲需容纳 窗口 + 一个块 的 key
FL_CONST int MAX_CHUNK = 512;
FL_CONST int SWA_RING = 1024;  // >= N_SWA + MAX_CHUNK - 1，取 2 的幂

// RMSNorm 平方和槽：写 x 的 kernel 原子累加 Σx²，下一阶段读一个标量即得 scale。
// 槽 2*il = 第 il 层 attn_norm 输入，2*il+1 = ffn_norm 输入，2*N_LAYER = output_norm 输入；按 token 奇偶双缓冲。
FL_CONST int SS_SLOTS = 2 * N_LAYER + 1;

// 全注意力层（不用 RoPE）：0, 1, 5, 9, ..., 45, 48；其余为 SWA 层（用 RoPE）
inline bool layer_is_full(int il) { return il == 0 || il == N_LAYER - 1 || il % 4 == 1; }
