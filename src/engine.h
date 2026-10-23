// Flint Metal 推理引擎（decode，batch=1）。
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gguf.h"
#include "ref/ref_model.h"

namespace flint {

struct EngineOptions {
    std::string metallib_path;
    std::string fused_metallib_path;
    std::string prefill_metallib_path;
    int n_ctx = 4096;
    // 非 0：专家流式加载模式。非专家权重常驻，专家放进 pool_gb 大小的槽池，按需从 SSD 读（LRU）；<0 = 自动取最大
    double pool_gb = 0;
    std::string model_path;  // 流式模式下用于 pread
};

class Engine {
public:
    Engine(const GgufFile &f, const EngineOptions &opt);
    ~Engine();

    // 处理下一个位置的 token。want_logits 时把 logits 拷回（否则只在 GPU 上 argmax）。
    // 返回贪心的下一个 token。
    int32_t step(int32_t token, std::vector<float> *logits = nullptr);

    // 批量 prefill：从当前位置起处理 tokens，返回贪心的下一个 token（logits 为最后一个位置）
    int32_t prefill(const std::vector<int32_t> &tokens, std::vector<float> *logits = nullptr);

    // 测试：用 GPU 解码 tensor 的一行。
    std::vector<float> debug_get_row(const Tensor &t, uint32_t row);
    // 测试批量矩阵乘（prefill）：X 为 [T][K]；experts 非空时按段用不同专家（mm_id）
    // 测试 prefill 注意力：qkv [n_tok][5120] 原始投影，按 chunk 分块，返回注意力输出 [n_tok][4096]
    std::vector<float> debug_attn(int il, const std::vector<float> &qkv, int n_tok, int chunk);
    std::vector<float> debug_mm(const Tensor &t, const std::vector<float> &X, int T, const std::vector<int> &experts = {},
                                const std::vector<int> &segs = {});

    // 调试：>=0 时只跑前 n 层；step 后 debug_hidden() 为该层输出
    int debug_n_layers = -1;
    std::vector<float> debug_hidden() const;
    std::vector<int> debug_experts() const;
    // 最近一次 step 每层选中的专家 [N_LAYER][6]（dense 层无意义）
    std::vector<int> routing() const;
    // 预测的每层专家（下一层 router 作用于上一层 ffn 输入；需 streaming 或 FLINT_OPT=pred）
    // [N_LAYER][10]，按分数降序；ahead=1 来自上一层，ahead=2 来自上上层
    std::vector<int> routing_pred(int ahead = 1) const;

    // profile：每次调度单独计时（会比正常模式慢一些，只看相对占比）
    void enable_profiling();
    std::vector<std::pair<std::string, double>> last_profile();

    int pos() const { return pos_; }
    void reset() { pos_ = 0; }
    void truncate(int pos) { pos_ = pos; }
    int n_ctx() const;
    const float *logits_ptr() const;  // 最近一次 step / prefill 的 logits（GPU 共享缓冲区）
    double last_gpu_ms = 0, last_encode_ms = 0;
    // 流式模式统计（累计）
    struct StreamStats {
        uint64_t hits = 0, misses = 0, bytes = 0, prefetched = 0;
        double load_ms = 0;  // CPU 处理未命中（含 pread）的总时间
        int n_slots = 0;
        double slot_mb = 0;
    };
    StreamStats stream_stats() const;  // 最近一次 step 的 GPU 执行时间 / CPU 编码时间
    const HParams &hp() const { return hp_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    HParams hp_;
    int pos_ = 0;
};

}  // namespace flint
