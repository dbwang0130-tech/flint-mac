// prefill.metal 的参数（host 与 Metal 共用）。
#pragma once

#ifndef __METAL_VERSION__
#include <cstdint>
typedef uint32_t uint;
typedef uint64_t ulong;
#endif

// Y[t][m] = Σ_k W[m][k] · X[t][k]，t < N（token 数），m < M（权重行数）
struct MmArgs {
    uint M, K, N;
    uint row_bytes;  // W 一行的字节数
    uint x_stride;   // X 行跨度（float 个数）
    uint y_stride;   // Y 行跨度（float 个数）
};

// 专家版：按分块表逐块计算。tiles[i] = {矩阵号（专家或槽）, 段起点, 段长}，段内 token 在 X/Y 中连续
struct MmIdArgs {
    MmArgs mm;
    ulong mat_stride;  // 相邻矩阵（专家/槽）的字节距离
};

// 批量 QK norm + RoPE + 写 KV：qkv 为 [T][Q_DIM + 2*KV_DIM]，第 t 个 token 位置 p0 + t
struct QkPfArgs {
    uint p0, T, is_full;
};

// prefill 注意力：本块 T 个 query（位置 p0..p0+T-1），key 来自本层 cache（已含本块）
struct AttnPfArgs {
    uint p0, T, is_full;
    uint q_stride;  // Q 行跨度（float 个数）
};

// 按行的小 kernel 通用参数
struct RowsArgs {
    uint n_rows;
    uint width;  // 每行元素数
};
