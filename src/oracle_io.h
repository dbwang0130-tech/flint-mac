// 读取 tools/oracle 生成的对照文件，并提供 logits 比较指标。
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace flint {

struct Oracle {
    int n_vocab = 0;
    int stride = 1;  // 只有 pos % stride == 0 的位置有 logits
    std::vector<int32_t> tokens;
    std::vector<float> logits;  // [n_tokens / stride][n_vocab]
    std::vector<float> nll;     // 每个位置对真实下一 token 的 -log p（可能为空）

    explicit Oracle(const std::string &path) {
        FILE *f = fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open " + path);
        int32_t hdr[3];
        if (fread(hdr, 4, 3, f) != 3) throw std::runtime_error("bad oracle file");
        n_vocab = hdr[1];
        stride = hdr[2];
        tokens.resize(hdr[0]);
        logits.resize(size_t((hdr[0] + stride - 1) / stride) * n_vocab);
        if (fread(tokens.data(), 4, tokens.size(), f) != tokens.size() ||
            fread(logits.data(), 4, logits.size(), f) != logits.size())
            throw std::runtime_error("truncated oracle file");
        int32_t n_nll = 0;
        if (fread(&n_nll, 4, 1, f) == 1) {
            nll.resize(n_nll);
            if (fread(nll.data(), 4, n_nll, f) != size_t(n_nll)) throw std::runtime_error("truncated nll");
        }
        fclose(f);
    }

    bool has(size_t pos) const { return pos % stride == 0; }
    const float *at(size_t pos) const { return &logits[pos / stride * n_vocab]; }
};

struct LogitDiff {
    int top1_a, top1_b;
    double kl;       // KL(softmax(a) || softmax(b))
    double max_abs;  // max |a - b|
};

// -log softmax(l)[target]
inline double nll_of(const float *l, int n, int target) {
    double m = l[0], z = 0;
    for (int i = 1; i < n; i++) m = std::max(m, double(l[i]));
    for (int i = 0; i < n; i++) z += std::exp(l[i] - m);
    return m + std::log(z) - l[target];
}

inline LogitDiff compare_logits(const float *a, const float *b, int n) {
    LogitDiff d{};
    d.top1_a = int(std::max_element(a, a + n) - a);
    d.top1_b = int(std::max_element(b, b + n) - b);
    const double ma = a[d.top1_a], mb = b[d.top1_b];
    double za = 0, zb = 0;
    for (int i = 0; i < n; i++) {
        za += std::exp(a[i] - ma);
        zb += std::exp(b[i] - mb);
        d.max_abs = std::max(d.max_abs, double(std::fabs(a[i] - b[i])));
    }
    for (int i = 0; i < n; i++) {
        const double la = a[i] - ma - std::log(za), lb = b[i] - mb - std::log(zb);
        d.kl += std::exp(la) * (la - lb);
    }
    return d;
}

}  // namespace flint
