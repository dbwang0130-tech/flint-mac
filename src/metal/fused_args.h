// host 与 Metal 共用的 kernel 参数（形状都是 flint_config.h 里的常量，这里只放运行时变量）。
#pragma once
// fused.metal（v2 融合 kernel）的参数

#ifndef __METAL_VERSION__
#include <cstdint>
typedef uint32_t uint;
#endif

struct GateUpArgs {
    uint n_rows;   // 每个矩阵的行数（专家 512 / 共享 1024 / dense 6144）
    uint use_ids;  // 1：矩阵 m 取 ids[m] 号专家
};

struct AttnDecArgs {
    uint pos;             // 当前 token 位置
    uint is_full;         // 全注意力层（NoPE）；否则 SWA + RoPE
    uint n_split;         // key 区间切分数（>1 时写部分结果，由 k_attn_combine 合并）
    uint keys_per_split;
};

struct FRowArgs {
    uint K;
    uint row_bytes;
    uint row;  // 0xFFFFFFFF：行号从 GPU buffer 读
};
