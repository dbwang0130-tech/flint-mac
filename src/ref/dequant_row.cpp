#include "dequant.h"

#include <cstring>
#include <stdexcept>

extern "C" {
void dequantize_row_q8_0(const void *x, float *y, int64_t k);
void dequantize_row_q4_K(const void *x, float *y, int64_t k);
void dequantize_row_q5_K(const void *x, float *y, int64_t k);
void dequantize_row_q6_K(const void *x, float *y, int64_t k);
void dequantize_row_iq2_s(const void *x, float *y, int64_t k);
void dequantize_row_iq3_xxs(const void *x, float *y, int64_t k);
void dequantize_row_iq3_s(const void *x, float *y, int64_t k);
void dequantize_row_iq4_xs(const void *x, float *y, int64_t k);
}

namespace flint::ref {

void dequant_row(const Tensor &t, uint64_t row, float *out) {
    const int64_t k = t.dims[0];
    const uint8_t *p = t.data + row * t.row_bytes();
    switch (t.type) {
        case GgmlType::F32: memcpy(out, p, k * 4); return;
        case GgmlType::F16:
            for (int64_t i = 0; i < k; i++) out[i] = float(reinterpret_cast<const _Float16 *>(p)[i]);
            return;
        case GgmlType::BF16:
            for (int64_t i = 0; i < k; i++) {
                uint32_t u = uint32_t(reinterpret_cast<const uint16_t *>(p)[i]) << 16;
                memcpy(&out[i], &u, 4);
            }
            return;
        case GgmlType::Q8_0: dequantize_row_q8_0(p, out, k); return;
        case GgmlType::Q4_K: dequantize_row_q4_K(p, out, k); return;
        case GgmlType::Q5_K: dequantize_row_q5_K(p, out, k); return;
        case GgmlType::Q6_K: dequantize_row_q6_K(p, out, k); return;
        case GgmlType::IQ2_S: dequantize_row_iq2_s(p, out, k); return;
        case GgmlType::IQ3_XXS: dequantize_row_iq3_xxs(p, out, k); return;
        case GgmlType::IQ3_S: dequantize_row_iq3_s(p, out, k); return;
        case GgmlType::IQ4_XS: dequantize_row_iq4_xs(p, out, k); return;
        default: throw std::runtime_error(std::string("dequant_row: unsupported type ") + type_traits(t.type).name);
    }
}

}  // namespace flint::ref
