// flint-metal 命令行：
//   flint check-deq <model.gguf>                 GPU 解码 vs CPU 解码（每种量化类型抽样）
//   flint check <model.gguf> <oracle.bin>        整模型 logits vs llama.cpp
//   flint bench <model.gguf> [n_gen] [n_prompt]  贪心生成测速
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <map>
#include <string>

#include "engine.h"
#include "metal/flint_config.h"
#include "oracle_io.h"
#include "ref/dequant.h"

using namespace flint;
using clk = std::chrono::steady_clock;

static std::string metallib_path(const char *argv0) {
    std::string p = argv0;
    auto slash = p.rfind('/');
    return (slash == std::string::npos ? std::string(".") : p.substr(0, slash)) + "/flint.metallib";
}

static int check_deq(const GgufFile &f, Engine &eng) {
    std::map<GgmlType, int> seen;
    int bad = 0;
    for (const auto &t : f.tensors()) {
        if (t.dims.size() < 2 || t.type == GgmlType::F32 || seen[t.type]++ >= 3) continue;
        const uint64_t n_rows = t.nelements() / t.dims[0];
        for (uint64_t row : {uint64_t(0), n_rows / 3, n_rows - 1}) {
            auto gpu = eng.debug_get_row(t, uint32_t(row));
            std::vector<float> cpu(t.dims[0]);
            ref::dequant_row(t, row, cpu.data());
            double max_err = 0;
            for (size_t i = 0; i < cpu.size(); i++) max_err = std::max(max_err, double(std::fabs(gpu[i] - cpu[i])));
            const bool ok = max_err < 1e-5;
            bad += !ok;
            if (!ok || row == 0)
                printf("%-8s %-36s row %8llu  max|gpu-cpu| = %.3g %s\n", type_traits(t.type).name, t.name.c_str(),
                       (unsigned long long)row, max_err, ok ? "ok" : "MISMATCH");
        }
    }
    printf(bad ? "%d MISMATCHES\n" : "all dequant checks passed\n", bad);
    return bad != 0;
}

static int check(Engine &eng, const Oracle &oracle) {
    const auto t_start = clk::now();
    int agree = 0;
    double kl_sum = 0, kl_max = 0;
    std::vector<float> logits;
    double nll_ours = 0, nll_ref = 0;
    int n_nll = 0;
    for (size_t p = 0; p < oracle.tokens.size(); p++) {
        eng.step(oracle.tokens[p], &logits);
        if (p + 1 < oracle.tokens.size() && p < oracle.nll.size()) {
            const double mine = nll_of(logits.data(), oracle.n_vocab, oracle.tokens[p + 1]);
            if (getenv("FLINT_NLL_TRACE") && std::fabs(mine - oracle.nll[p]) > 0.1)
                printf("  nll pos %4zu: llama %.3f metal %.3f\n", p, oracle.nll[p], mine);
            nll_ours += mine;
            nll_ref += oracle.nll[p];
            n_nll++;
        }
        if (!oracle.has(p)) continue;
        auto d = compare_logits(oracle.at(p), logits.data(), oracle.n_vocab);
        agree += d.top1_a == d.top1_b;
        kl_sum += d.kl;
        kl_max = std::max(kl_max, d.kl);
        printf("pos %2zu  top1 llama=%6d metal=%6d %s  KL=%.2e  max|d|=%.3f\n", p, d.top1_a, d.top1_b,
               d.top1_a == d.top1_b ? "ok " : "MISMATCH", d.kl, d.max_abs);
    }
    const size_t n = (oracle.tokens.size() + oracle.stride - 1) / oracle.stride;
    printf("top1 agreement %d/%zu, mean KL %.2e, max KL %.2e\n", agree, n, kl_sum / n, kl_max);
    const double secs = std::chrono::duration<double>(clk::now() - t_start).count();
    printf("%zu tokens in %.1f s (%.2f tok/s incl. logits copy)\n", oracle.tokens.size(), secs, oracle.tokens.size() / secs);
    auto ss = eng.stream_stats();
    if (ss.n_slots)
        printf("stream: hit rate %.1f%%, %.1f MB/token from SSD, CPU stall %.2f ms/token\n", 100.0 * ss.hits / double(ss.hits + ss.misses),
               ss.bytes / 1e6 / oracle.tokens.size(), ss.load_ms / oracle.tokens.size());
    if (n_nll)
        printf("mean NLL over %d tokens: llama.cpp %.4f, metal %.4f (ppl %.3f vs %.3f)\n", n_nll, nll_ref / n_nll,
               nll_ours / n_nll, std::exp(nll_ref / n_nll), std::exp(nll_ours / n_nll));
    return agree * 10 < int(n) * 9;
}

// 逐层对比 GPU 与 CPU 参考的隐藏状态（第一个 token 起，跑 n_pos 个位置）
static int check_layers(const GgufFile &f, Engine &eng, const Oracle &oracle, int n_pos) {
    for (int nl = 1; nl <= 49; nl++) {
        ref::RefModel ref(f);
        ref.debug_n_layers = nl;
        Engine &g = eng;
        g.debug_n_layers = nl;
        double worst = 0;
        int worst_pos = 0;
        for (int p = 0; p < n_pos; p++) {
            ref.forward(oracle.tokens[p]);
            g.step(oracle.tokens[p]);
            auto a = ref.last_hidden(), b = g.debug_hidden();
            double num = 0, den = 0;
            for (size_t i = 0; i < a.size(); i++) { num += (a[i] - b[i]) * (a[i] - b[i]); den += a[i] * a[i]; }
            const double rel = std::sqrt(num / den);
            if (rel > worst) { worst = rel; worst_pos = p; }
        }
        printf("layers %2d: worst rel L2 err %.2e (pos %d)", nl, worst, worst_pos);
        if (nl > 2) {
            auto ge = g.debug_experts();
            printf("  ref experts:");
            for (int e : ref.last_experts()[nl - 1]) printf(" %3d", e);
            printf("  gpu:");
            for (int e : ge) printf(" %3d", e);
        }
        printf("\n");
        fflush(stdout);
        g.reset();
    }
    return 0;
}

static int profile(Engine &eng, int depth) {
    eng.enable_profiling();
    int32_t t = 150003;
    for (int i = 0; i < depth; i++) t = eng.step(i == 0 ? 150003 : 1000 + i);
    std::map<std::string, std::pair<double, int>> agg;
    const int n = 16;
    double gpu = 0;
    for (int i = 0; i < n; i++) {
        t = eng.step(t);
        gpu += eng.last_gpu_ms;
        for (auto &[label, ms] : eng.last_profile()) { agg[label].first += ms / n; agg[label].second++; }
    }
    std::vector<std::pair<std::string, std::pair<double, int>>> rows(agg.begin(), agg.end());
    std::sort(rows.begin(), rows.end(), [](auto &a, auto &b) { return a.second.first > b.second.first; });
    printf("profile @ depth %d, GPU %.2f ms/token (profiling overhead included)\n", depth, gpu / n);
    for (auto &[label, v] : rows)
        printf("  %-40s %7.3f ms  %5.1f%%  (%d dispatch/token)\n", label.c_str(), v.first, v.first / (gpu / n) * 100, v.second / n);
    return 0;
}

// 记录路由轨迹：先 teacher-force 跑 prompt token，再贪心生成 n_gen 个；每 token 一行 47 层 × 6 个专家 id
static int trace(Engine &eng, const Oracle &prompt, int n_gen, const char *out_path) {
    FILE *f = fopen(out_path, "w");
    int32_t t = 0;
    // 预测路由：.pred / .pred2 每行 47 层 × 10 个（按分数降序）
    FILE *fp = fopen((std::string(out_path) + ".pred").c_str(), "w");
    FILE *fp2 = fopen((std::string(out_path) + ".pred2").c_str(), "w");
    auto dump = [&]() {
        auto r = eng.routing(), p1 = eng.routing_pred(1), p2 = eng.routing_pred(2);
        for (int il = N_DENSE; il < N_LAYER; il++) {
            const bool last = il == N_LAYER - 1;
            for (int k = 0; k < N_USED; k++) fprintf(f, "%d%c", r[il * N_USED + k], last && k == N_USED - 1 ? '\n' : ' ');
            for (int k = 0; k < 10; k++) {
                fprintf(fp, "%d%c", p1[il * 10 + k], last && k == 9 ? '\n' : ' ');
                fprintf(fp2, "%d%c", p2[il * 10 + k], last && k == 9 ? '\n' : ' ');
            }
        }
    };
    for (size_t p = 0; p < prompt.tokens.size(); p++) { t = eng.step(prompt.tokens[p]); dump(); }
    const auto st0 = eng.stream_stats();
    const auto t0 = clk::now();
    for (int i = 0; i < n_gen; i++) { t = eng.step(t); dump(); }
    const double secs = std::chrono::duration<double>(clk::now() - t0).count();
    const auto st1 = eng.stream_stats();
    if (n_gen) {
        printf("generate %d tokens: %.2f tok/s", n_gen, n_gen / secs);
        if (st1.n_slots)
            printf(", hit %.1f%%, %.1f MB/token SSD, stall %.2f ms/token",
                   100.0 * (st1.hits - st0.hits) / double(st1.hits - st0.hits + st1.misses - st0.misses),
                   (st1.bytes - st0.bytes) / 1e6 / n_gen, (st1.load_ms - st0.load_ms) / n_gen);
        printf("\n");
    }
    fclose(f);
    fclose(fp);
    fclose(fp2);
    printf("traced %zu prompt + %d generated tokens to %s\n", prompt.tokens.size(), n_gen, out_path);
    return 0;
}

// 批量矩阵乘单元测试：每种量化类型取一个真实权重，随机 X（T=200），GPU vs CPU（逐行解码 + double 点积）
static int test_mm(const GgufFile &f, Engine &eng) {
    std::map<GgmlType, int> seen;
    int bad = 0;
    const int T = 200;
    for (const auto &t : f.tensors()) {
        if (t.dims.size() < 2 || t.type == GgmlType::F32 || t.dims[1] % 64 || seen[t.type]++) continue;
        const bool is_exp = t.dims.size() == 3;
        const int K = t.dims[0], M = t.dims[1];
        std::vector<float> X(size_t(T) * K);
        srand(7);
        for (auto &v : X) v = (rand() % 2001 - 1000) / 1000.0f;
        // 专家张量：前 70 个 token 用专家 3，后 130 个用专家 200
        const std::vector<int> ex = is_exp ? std::vector<int>{3, 200} : std::vector<int>{};
        const std::vector<int> sg = is_exp ? std::vector<int>{70, 130} : std::vector<int>{};
        auto Y = eng.debug_mm(t, X, T, ex, sg);
        double err = 0, ref_rms = 0;
        std::vector<float> row(K);
        for (int m = 0; m < M; m += 7) {
            for (int tok : {0, 69, 70, 127, 128, 199}) {
                const int e = is_exp ? (tok < 70 ? 3 : 200) : 0;
                ref::dequant_row(t, uint64_t(e) * M + m, row.data());
                double r = 0;
                for (int k = 0; k < K; k++) r += double(row[k]) * X[size_t(tok) * K + k];
                err = std::max(err, std::fabs(Y[size_t(tok) * M + m] - r));
                ref_rms += r * r;
            }
        }
        ref_rms = std::sqrt(ref_rms / ((M + 6) / 7 * 6));
        const double rel = err / ref_rms;
        const bool ok = rel < 5e-3;
        bad += !ok;
        printf("%-8s %-34s %s K=%-5d M=%-6d max|err|/rms = %.2e %s\n", type_traits(t.type).name, t.name.c_str(),
               is_exp ? "mm_id" : "mm   ", K, M, rel, ok ? "ok" : "MISMATCH");
    }
    return bad != 0;
}

// 批量矩阵乘测速：TFLOPS 与有效权重带宽
static int bench_mm(const GgufFile &f, Engine &eng) {
    for (const char *name : {"blk.3.attn_q.weight", "blk.3.ffn_gate_exps.weight", "output.weight"}) {
        const Tensor &t = f.tensor(name);
        const int K = t.dims[0], M = t.dims[1];
        for (int T : {128, 512, 2048}) {
            std::vector<float> X(size_t(T) * K, 0.01f);
            const bool is_exp = t.dims.size() == 3;
            // 专家：把 T 个 token 平均分给 8 个专家
            std::vector<int> ex, sg;
            if (is_exp) for (int i = 0; i < 8; i++) { ex.push_back(i * 17); sg.push_back(T / 8); }
            eng.debug_mm(t, X, T, ex, sg);
            const double ms = eng.last_gpu_ms;
            const double flop = 2.0 * M * K * T, bytes = double(t.row_bytes()) * M * (is_exp ? 8 : 1);
            printf("%-30s %-8s T=%-5d %8.3f ms  %6.2f TFLOPS  weights %6.1f GB/s\n", name, type_traits(t.type).name, T, ms,
                   flop / ms / 1e9, bytes / ms / 1e6);
        }
    }
    return 0;
}

// prefill 注意力单元测试：随机 qkv，GPU（分块 qk_prefill + flash attention）vs CPU（double，K/V 按 f16 舍入）
static int test_attn(const GgufFile &f, Engine &eng) {
    const int n = 1300, QS = Q_DIM + 2 * KV_DIM;
    int bad = 0;
    for (int il : {0, 2}) {  // 层 0 为全注意力（NoPE），层 2 为 SWA + RoPE
        const bool full = layer_is_full(il);
        std::vector<float> qkv(size_t(n) * QS);
        srand(11 + il);
        for (auto &v : qkv) v = (rand() % 2001 - 1000) / 500.0f;
        auto gpu = eng.debug_attn(il, qkv, n, 512);
        const double gms = eng.last_gpu_ms;
        auto L = [&](const char *nm) { return (const float *)f.tensor("blk." + std::to_string(il) + "." + nm).data; };
        const float *qn = L("attn_q_norm.weight"), *kn = L("attn_k_norm.weight"), *sinks = L("attn_sinks.weight");
        auto norm_rope = [&](const float *in, const float *w, int pos, double *o) {
            double ss = 0;
            for (int d = 0; d < HEAD_DIM; d++) ss += double(in[d]) * in[d];
            const double sc = 1.0 / std::sqrt(ss / HEAD_DIM + RMS_EPS);
            double t[HEAD_DIM];
            for (int d = 0; d < HEAD_DIM; d++) t[d] = in[d] * sc * w[d];
            for (int d = 0; d < HEAD_DIM; d++) o[d] = t[d];
            if (full) return;
            for (int i = 0; i < HEAD_DIM / 2; i++) {
                const double a = pos * std::pow(double(ROPE_THETA), -2.0 * i / HEAD_DIM), c = std::cos(a), s = std::sin(a);
                o[i] = t[i] * c - t[i + 64] * s;
                o[i + 64] = t[i] * s + t[i + 64] * c;
            }
        };
        // 预先算好所有 key/value（f16 舍入）
        std::vector<double> K(size_t(n) * KV_DIM), V(size_t(n) * KV_DIM);
        for (int p = 0; p < n; p++)
            for (int hk = 0; hk < N_HEAD_KV; hk++) {
                double tmp[HEAD_DIM];
                norm_rope(&qkv[size_t(p) * QS + Q_DIM + hk * HEAD_DIM], kn, p, tmp);
                for (int d = 0; d < HEAD_DIM; d++) {
                    K[size_t(p) * KV_DIM + hk * HEAD_DIM + d] = double(_Float16(tmp[d]));
                    V[size_t(p) * KV_DIM + hk * HEAD_DIM + d] = double(_Float16(qkv[size_t(p) * QS + Q_DIM + KV_DIM + hk * HEAD_DIM + d]));
                }
            }
        double max_err = 0, rms = 0;
        int cnt = 0;
        for (int p : {0, 1, 31, 32, 200, 511, 512, 513, 600, 1023, 1024, 1100, 1299}) {
            for (int h : {0, 7, 8, 31}) {
                double q[HEAD_DIM];
                norm_rope(&qkv[size_t(p) * QS + h * HEAD_DIM], qn, p, q);
                const int kvh = h / GQA, lo = full ? 0 : std::max(0, p - (N_SWA - 1));
                double m = sinks[h], l = 0, acc[HEAD_DIM] = {0};
                std::vector<double> sc(p + 1);
                for (int j = lo; j <= p; j++) {
                    double d = 0;
                    for (int i = 0; i < HEAD_DIM; i++) d += q[i] * K[size_t(j) * KV_DIM + kvh * HEAD_DIM + i];
                    sc[j] = d / std::sqrt(double(HEAD_DIM));
                    m = std::max(m, sc[j]);
                }
                l = std::exp(sinks[h] - m);
                for (int j = lo; j <= p; j++) {
                    const double w = std::exp(sc[j] - m);
                    l += w;
                    for (int i = 0; i < HEAD_DIM; i++) acc[i] += w * V[size_t(j) * KV_DIM + kvh * HEAD_DIM + i];
                }
                for (int i = 0; i < HEAD_DIM; i++) {
                    const double r = acc[i] / l, g = gpu[size_t(p) * Q_DIM + h * HEAD_DIM + i];
                    max_err = std::max(max_err, std::fabs(r - g));
                    rms += r * r;
                    cnt++;
                }
            }
        }
        rms = std::sqrt(rms / cnt);
        const bool ok = max_err / rms < 2e-2;
        bad += !ok;
        printf("layer %d (%s): max|err| = %.3e, rms = %.3e, rel = %.2e %s  (GPU %.2f ms for %d tokens)\n", il,
               full ? "full/NoPE" : "SWA/RoPE", max_err, rms, max_err / rms, ok ? "ok" : "MISMATCH", gms, n);
    }
    return bad != 0;
}

// prefill 正确性：与同一引擎的逐 token decode 对比（最后一个位置的 logits），并测 prefill 速度
static int check_prefill(Engine &eng, const Oracle &oracle, int n_max) {
    std::vector<int32_t> toks(oracle.tokens.begin(), oracle.tokens.begin() + std::min<size_t>(n_max, oracle.tokens.size()));
    const int n = int(toks.size());
    std::vector<float> ld, lp, lm;
    eng.reset();
    auto t0 = clk::now();
    for (int p = 0; p < n; p++) eng.step(toks[p], p == n - 1 ? &ld : nullptr);
    fprintf(stderr, "[check-prefill] decode done\n");
    const double t_dec = std::chrono::duration<double>(clk::now() - t0).count();
    eng.reset();
    t0 = clk::now();
    eng.prefill(toks, &lp);
    fprintf(stderr, "[check-prefill] prefill done\n");
    const double t_pf = std::chrono::duration<double>(clk::now() - t0).count();
    // 先 prefill 前一半，再逐 token decode 后一半
    eng.reset();
    const int k = n / 2;
    eng.prefill(std::vector<int32_t>(toks.begin(), toks.begin() + k));
    fprintf(stderr, "[check-prefill] half prefill done\n");
    for (int p = k; p < n; p++) eng.step(toks[p], p == n - 1 ? &lm : nullptr);
    auto d1 = compare_logits(ld.data(), lp.data(), N_VOCAB), d2 = compare_logits(ld.data(), lm.data(), N_VOCAB);
    printf("%d tokens: decode %.2f s (%.1f tok/s), prefill %.2f s (%.1f tok/s)\n", n, t_dec, n / t_dec, t_pf, n / t_pf);
    printf("prefill vs decode:        top1 %d/%d  KL %.2e  max|d| %.3f\n", d1.top1_a, d1.top1_b, d1.kl, d1.max_abs);
    printf("prefill(%d)+decode vs decode: top1 %d/%d  KL %.2e  max|d| %.3f\n", k, d2.top1_a, d2.top1_b, d2.kl, d2.max_abs);
    if (oracle.has(n - 1)) {
        auto d3 = compare_logits(oracle.at(n - 1), lp.data(), N_VOCAB);
        printf("prefill vs llama.cpp:     top1 %d/%d  KL %.2e\n", d3.top1_a, d3.top1_b, d3.kl);
    }
    return 0;
}

// prefill 测速：先预热一次，再对各长度计时（token 取自 oracle 文件，不足则循环）
static int bench_prefill(Engine &eng, const Oracle &src, const std::vector<int> &lens) {
    auto make = [&](int n) {
        std::vector<int32_t> t(n);
        for (int i = 0; i < n; i++) t[i] = src.tokens[i % src.tokens.size()];
        return t;
    };
    eng.reset();
    eng.prefill(make(600));
    for (int n : lens) {
        eng.reset();
        auto toks = make(n);
        auto t0 = clk::now();
        eng.prefill(toks);
        const double secs = std::chrono::duration<double>(clk::now() - t0).count();
        printf("pp%-6d %8.1f tok/s  (%.2f s)\n", n, n / secs, secs);
        fflush(stdout);
    }
    return 0;
}

static int bench(Engine &eng, int n_gen, int n_prompt) {
    int32_t t = 150003;  // <|user|>
    auto t0 = clk::now();
    for (int i = 0; i < n_prompt; i++) t = eng.step(i == 0 ? 150003 : 1000 + i);
    auto t1 = clk::now();
    // 预热后计时生成
    for (int i = 0; i < 4; i++) t = eng.step(t);
    auto t2 = clk::now();
    double gpu = 0, enc = 0;
    for (int i = 0; i < n_gen; i++) { t = eng.step(t); gpu += eng.last_gpu_ms; enc += eng.last_encode_ms; }
    auto t3 = clk::now();
    printf("avg GPU %.2f ms, CPU encode %.2f ms per token\n", gpu / n_gen, enc / n_gen);
    auto ss = eng.stream_stats();
    if (ss.n_slots) {
        const double n = double(ss.hits + ss.misses);
        printf("stream: hit rate %.1f%% (prefetched %.1f/token), %.1f MB/token from SSD, CPU stall %.2f ms/token (over all %d steps)\n",
               100.0 * ss.hits / n, ss.prefetched / (n / (N_USED * (N_LAYER - N_DENSE))), ss.bytes / 1e6 / (n / (N_USED * (N_LAYER - N_DENSE))), ss.load_ms / (n / (N_USED * (N_LAYER - N_DENSE))),
               int(n / (N_USED * (N_LAYER - N_DENSE))));
    }
    const double pp = std::chrono::duration<double>(t1 - t0).count();
    const double tg = std::chrono::duration<double>(t3 - t2).count();
    if (n_prompt) printf("prompt (token-by-token) %d tok: %.1f tok/s\n", n_prompt, n_prompt / pp);
    printf("tg%d @ depth %d: %.2f tok/s (%.2f ms/tok)\n", n_gen, n_prompt + 4, n_gen / tg, tg * 1e3 / n_gen);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s check-deq|check|bench <model.gguf> [...]\n", argv[0]);
        return 1;
    }
    const std::string cmd = argv[1];
    GgufFile f(argv[2]);
    EngineOptions opt;
    opt.metallib_path = metallib_path(argv[0]);
    opt.model_path = argv[2];
    if (const char *pg = getenv("FLINT_POOL_GB"))  // 专家流式加载：GB 数，或 auto
        opt.pool_gb = std::string(pg) == "auto" ? -1 : atof(pg);
    opt.fused_metallib_path = opt.metallib_path.substr(0, opt.metallib_path.rfind('/')) + "/fused.metallib";
    opt.prefill_metallib_path = opt.metallib_path.substr(0, opt.metallib_path.rfind('/')) + "/prefill.metallib";
    Engine eng(f, opt);
    if (cmd == "check-deq") return check_deq(f, eng);
    if (cmd == "check") return check(eng, Oracle(argv[3]));
    if (cmd == "check-layers") return check_layers(f, eng, Oracle(argv[3]), argc > 4 ? atoi(argv[4]) : 1);
    if (cmd == "trace") return trace(eng, Oracle(argv[3]), atoi(argv[4]), argv[5]);
    if (cmd == "bench-mm") return bench_mm(f, eng);
    if (cmd == "bench-prefill") {
        std::vector<int> lens;
        for (int i = 4; i < argc; i++) lens.push_back(atoi(argv[i]));
        return bench_prefill(eng, Oracle(argv[3]), lens);
    }
    if (cmd == "check-prefill") return check_prefill(eng, Oracle(argv[3]), argc > 4 ? atoi(argv[4]) : 1 << 30);
    if (cmd == "test-attn") return test_attn(f, eng);
    if (cmd == "test-mm") return test_mm(f, eng);
    if (cmd == "profile") return profile(eng, argc > 3 ? atoi(argv[3]) : 0);
    if (cmd == "bench") return bench(eng, argc > 3 ? atoi(argv[3]) : 128, argc > 4 ? atoi(argv[4]) : 0);
    fprintf(stderr, "unknown command %s\n", cmd.c_str());
    return 1;
}
