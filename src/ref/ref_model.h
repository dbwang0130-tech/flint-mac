// Flint CPU 参考前向（fp32，逐 token）。只求正确，不求快；用作 Metal kernel 的对照。
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../gguf.h"

namespace flint {

struct HParams {
    int n_layer, n_embd, n_head, n_head_kv, head_dim, n_vocab;
    int n_ff_dense, n_ff_exp, n_ff_shexp, n_expert, n_expert_used, n_dense_lead;
    int n_swa;
    float rms_eps, rope_theta;
    std::vector<bool> is_swa, has_rope;

    static HParams from_gguf(const GgufFile &f);
};

namespace ref {

class RefModel {
public:
    explicit RefModel(const GgufFile &f, int n_threads = 10);

    // 处理位置 pos 的 token（必须按 0,1,2… 顺序调用），返回 n_vocab 个 logits。
    std::vector<float> forward(int32_t token);

    // 每层 MoE 选中的专家（最近一次 forward），调试用。
    const std::vector<std::vector<int>> &last_experts() const { return last_experts_; }

    const HParams &hp() const { return hp_; }

    // 调试：>=0 时只跑前 n 层，forward 后 last_hidden() 为该层输出
    int debug_n_layers = -1;
    const std::vector<float> &last_hidden() const { return last_hidden_; }

private:
    void matvec(const Tensor &w, const float *x, float *y, uint64_t row0, uint64_t n_rows) const;
    void matvec(const Tensor &w, const float *x, float *y) const { matvec(w, x, y, 0, w.dims[1]); }
    const Tensor &lt(int il, const char *name) const;

    const GgufFile &f_;
    HParams hp_;
    int n_threads_;
    int pos_ = 0;
    std::vector<std::vector<float>> k_cache_, v_cache_;  // [layer][pos * kv_dim]
    std::vector<std::vector<int>> last_experts_;
    std::vector<float> last_hidden_;
};

}  // namespace ref
}  // namespace flint
