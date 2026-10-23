// 用 CPU 参考实现跑 oracle 的 token 序列，逐位置对比 llama.cpp 的 logits。
// 用法: ref_check <model.gguf> <oracle.bin> [max_positions]
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#include "../oracle_io.h"
#include "ref_model.h"

int main(int argc, char **argv) {
    flint::GgufFile f(argv[1]);
    flint::Oracle oracle(argv[2]);
    const size_t n = argc > 3 ? std::min<size_t>(atoi(argv[3]), oracle.tokens.size()) : oracle.tokens.size();
    flint::ref::RefModel model(f);
    int agree = 0;
    double nll_ref = 0, nll_llama = 0;
    int n_nll = 0;
    double kl_sum = 0;
    for (size_t p = 0; p < n; p++) {
        auto t0 = std::chrono::steady_clock::now();
        auto logits = model.forward(oracle.tokens[p]);
        if (p + 1 < oracle.tokens.size() && p < oracle.nll.size()) {
            const double mine = flint::nll_of(logits.data(), oracle.n_vocab, oracle.tokens[p + 1]);
            if (std::fabs(mine - oracle.nll[p]) > 0.1) printf("  nll pos %4zu: llama %.3f ref %.3f\n", p, oracle.nll[p], mine);
            nll_ref += mine;
            nll_llama += oracle.nll[p];
            n_nll++;
        }
        if (!oracle.has(p)) continue;
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        auto d = flint::compare_logits(oracle.at(p), logits.data(), oracle.n_vocab);
        agree += d.top1_a == d.top1_b;
        kl_sum += d.kl;
        printf("pos %2zu tok %6d  top1 llama=%6d ref=%6d %s  KL=%.2e  max|d|=%.3f  (%.0f ms)\n", p, oracle.tokens[p],
               d.top1_a, d.top1_b, d.top1_a == d.top1_b ? "ok " : "MISMATCH", d.kl, d.max_abs, ms);
        fflush(stdout);
    }
    printf("top1 agreement %d/%zu, mean KL %.2e\n", agree, n, kl_sum / n);
    printf("mean NLL over %d tokens: llama.cpp %.4f, ref %.4f\n", n_nll, nll_llama / n_nll, nll_ref / n_nll);
}
