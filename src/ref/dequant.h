// CPU 参考解码入口（实现见 dequant.c）。
#pragma once

#include <cstdint>

#include "../gguf.h"

namespace flint::ref {

// 解码 tensor 的第 row 行（dims[0] 个元素）到 out。3D 专家张量用 row = expert * dims[1] + r。
void dequant_row(const Tensor &t, uint64_t row, float *out);

}  // namespace flint::ref
