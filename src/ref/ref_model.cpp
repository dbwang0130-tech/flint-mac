#include "ref_model.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <thread>

#include "dequant.h"

namespace flint {

HParams HParams::from_gguf(const GgufFile &f) {
    HParams h{};
    const std::string a = f.get_string("general.architecture");
    if (a != "flint") throw std::runtime_error("not a flint model: " + a);
    auto k = [&](const char *s) { return a + "." + s; };
    h.n_layer = f.get_int(k("block_count"));
    h.n_embd = f.get_int(k("embedding_length"));
    h.n_head = f.get_int(k("attention.head_count"));
    h.n_head_kv = f.get_int(k("attention.head_count_kv"));
    h.head_dim = f.get_int(k("attention.key_length"));
    h.n_ff_dense = f.get_int(k("feed_forward_length"));
    h.n_ff_exp = f.get_int(k("expert_feed_forward_length"));
    h.n_ff_shexp = f.get_int(k("expert_shared_feed_forward_length"));
    h.n_expert = f.get_int(k("expert_count"));
    h.n_expert_used = f.get_int(k("expert_used_count"));
    h.n_dense_lead = f.get_int(k("leading_dense_block_count"));
    h.n_swa = f.get_int(k("attention.sliding_window"));
    h.rms_eps = f.get_float(k("attention.layer_norm_rms_epsilon"));
    h.rope_theta = f.get_float(k("rope.freq_base"));
    h.is_swa = f.get_bool_array(k("attention.sliding_window_pattern"));
    h.has_rope = f.get_bool_array(k("attention.rope_pattern"));
    h.n_vocab = f.tensor("output.weight").dims[1];
    if (f.get_int(k("expert_gating_func")) != 2 || !f.get_bool(k("expert_weights_norm")) ||
        f.get_float(k("expert_weights_scale")) != 1.0)
        throw std::runtime_error("unexpected routing config (expect sigmoid + norm, scale 1)");
    return h;
}

namespace ref {

namespace {

void rmsnorm(const float *x, const float *w, float *y, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += double(x[i]) * x[i];
    const float scale = 1.0f / std::sqrt(float(ss / n) + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * scale * w[i];
}

// NEOX 风格：第 i 维与第 i + d/2 维成对旋转。
void rope_neox(float *x, int d, int pos, float theta) {
    for (int i = 0; i < d / 2; i++) {
        const float freq = std::pow(theta, -2.0f * i / d);
        const float a = pos * freq, c = std::cos(a), s = std::sin(a);
        const float x0 = x[i], x1 = x[i + d / 2];
        x[i] = x0 * c - x1 * s;
        x[i + d / 2] = x0 * s + x1 * c;
    }
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

}  // namespace

RefModel::RefModel(const GgufFile &f, int n_threads)
    : f_(f), hp_(HParams::from_gguf(f)), n_threads_(n_threads), k_cache_(hp_.n_layer), v_cache_(hp_.n_layer) {}

const Tensor &RefModel::lt(int il, const char *name) const {
    return f_.tensor("blk." + std::to_string(il) + "." + name);
}

void RefModel::matvec(const Tensor &w, const float *x, float *y, uint64_t row0, uint64_t n_rows) const {
    const int k = w.dims[0];
    auto work = [&](uint64_t begin, uint64_t end) {
        std::vector<float> row(k);
        for (uint64_t r = begin; r < end; r++) {
            dequant_row(w, row0 + r, row.data());
            double acc = 0;
            for (int i = 0; i < k; i++) acc += double(row[i]) * x[i];
            y[r] = float(acc);
        }
    };
    const int nt = n_rows >= 256 ? n_threads_ : 1;
    std::vector<std::thread> threads;
    const uint64_t chunk = (n_rows + nt - 1) / nt;
    for (int t = 0; t < nt; t++) {
        const uint64_t b = t * chunk, e = std::min<uint64_t>(n_rows, b + chunk);
        if (b < e) threads.emplace_back(work, b, e);
    }
    for (auto &th : threads) th.join();
}

std::vector<float> RefModel::forward(int32_t token) {
    const HParams &h = hp_;
    const int E = h.n_embd, D = h.head_dim, HQ = h.n_head, HKV = h.n_head_kv;
    const int kv_dim = HKV * D, group = HQ / HKV;
    const int pos = pos_++;

    std::vector<float> x(E), xn(E), q(HQ * D), kk(kv_dim), vv(kv_dim), att(HQ * D), o(E);
    dequant_row(f_.tensor("token_embd.weight"), token, x.data());
    last_experts_.assign(h.n_layer, {});

    const int n_run = debug_n_layers >= 0 ? debug_n_layers : h.n_layer;
    for (int il = 0; il < n_run; il++) {
        // ---- attention ----
        rmsnorm(x.data(), (const float *)lt(il, "attn_norm.weight").data, xn.data(), E, h.rms_eps);
        matvec(lt(il, "attn_q.weight"), xn.data(), q.data());
        matvec(lt(il, "attn_k.weight"), xn.data(), kk.data());
        matvec(lt(il, "attn_v.weight"), xn.data(), vv.data());
        const float *qn = (const float *)lt(il, "attn_q_norm.weight").data;
        const float *kn = (const float *)lt(il, "attn_k_norm.weight").data;
        for (int hh = 0; hh < HQ; hh++) {
            rmsnorm(&q[hh * D], qn, &q[hh * D], D, h.rms_eps);
            if (h.has_rope[il]) rope_neox(&q[hh * D], D, pos, h.rope_theta);
        }
        for (int hh = 0; hh < HKV; hh++) {
            rmsnorm(&kk[hh * D], kn, &kk[hh * D], D, h.rms_eps);
            if (h.has_rope[il]) rope_neox(&kk[hh * D], D, pos, h.rope_theta);
        }
        auto &kc = k_cache_[il], &vc = v_cache_[il];
        kc.insert(kc.end(), kk.begin(), kk.end());
        vc.insert(vc.end(), vv.begin(), vv.end());

        const float *sinks = (const float *)lt(il, "attn_sinks.weight").data;
        const int first = h.is_swa[il] ? std::max(0, pos - h.n_swa + 1) : 0;
        const float scale = 1.0f / std::sqrt(float(D));
        std::vector<float> s(pos + 1);
        for (int hh = 0; hh < HQ; hh++) {
            const int kvh = hh / group;
            float m = sinks[hh];
            for (int j = first; j <= pos; j++) {
                const float *kj = &kc[j * kv_dim + kvh * D];
                float d = 0;
                for (int i = 0; i < D; i++) d += q[hh * D + i] * kj[i];
                s[j] = d * scale;
                m = std::max(m, s[j]);
            }
            double denom = std::exp(sinks[hh] - m);
            std::vector<double> acc(D, 0.0);
            for (int j = first; j <= pos; j++) {
                const double p = std::exp(s[j] - m);
                denom += p;
                const float *vj = &vc[j * kv_dim + kvh * D];
                for (int i = 0; i < D; i++) acc[i] += p * vj[i];
            }
            for (int i = 0; i < D; i++) att[hh * D + i] = float(acc[i] / denom);
        }
        matvec(lt(il, "attn_output.weight"), att.data(), o.data());
        for (int i = 0; i < E; i++) x[i] += o[i];

        // ---- FFN ----
        rmsnorm(x.data(), (const float *)lt(il, "ffn_norm.weight").data, xn.data(), E, h.rms_eps);
        std::vector<float> y(E, 0.0f);
        auto swiglu = [&](const Tensor &gate, const Tensor &up, const Tensor &down, uint64_t up_row0,
                          uint64_t down_row0, int n_ff, float weight) {
            std::vector<float> g(n_ff), u(n_ff), out(E);
            matvec(gate, xn.data(), g.data(), up_row0, n_ff);
            matvec(up, xn.data(), u.data(), up_row0, n_ff);
            for (int i = 0; i < n_ff; i++) g[i] = silu(g[i]) * u[i];
            matvec(down, g.data(), out.data(), down_row0, E);
            for (int i = 0; i < E; i++) y[i] += weight * out[i];
        };

        if (il < h.n_dense_lead) {
            swiglu(lt(il, "ffn_gate.weight"), lt(il, "ffn_up.weight"), lt(il, "ffn_down.weight"), 0, 0,
                   h.n_ff_dense, 1.0f);
        } else {
            std::vector<float> logits(h.n_expert), probs(h.n_expert);
            matvec(lt(il, "ffn_gate_inp.weight"), xn.data(), logits.data());
            for (int e = 0; e < h.n_expert; e++) probs[e] = 1.0f / (1.0f + std::exp(-logits[e]));
            std::vector<int> idx(h.n_expert);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + h.n_expert_used, idx.end(),
                              [&](int a, int b) { return probs[a] > probs[b]; });
            idx.resize(h.n_expert_used);
            float wsum = 0;
            for (int e : idx) wsum += probs[e];
            wsum = std::max(wsum, 6.103515625e-5f);
            for (int e : idx) {
                swiglu(lt(il, "ffn_gate_exps.weight"), lt(il, "ffn_up_exps.weight"), lt(il, "ffn_down_exps.weight"),
                       uint64_t(e) * h.n_ff_exp, uint64_t(e) * E, h.n_ff_exp, probs[e] / wsum);
            }
            last_experts_[il] = idx;
            swiglu(lt(il, "ffn_gate_shexp.weight"), lt(il, "ffn_up_shexp.weight"), lt(il, "ffn_down_shexp.weight"),
                   0, 0, h.n_ff_shexp, 1.0f);
        }
        for (int i = 0; i < E; i++) x[i] += y[i];
    }

    last_hidden_ = x;
    rmsnorm(x.data(), (const float *)f_.tensor("output_norm.weight").data, xn.data(), E, h.rms_eps);
    std::vector<float> logits(h.n_vocab);
    matvec(f_.tensor("output.weight"), xn.data(), logits.data());
    return logits;
}

}  // namespace ref
}  // namespace flint
