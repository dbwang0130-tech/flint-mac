// 用 llama.cpp 生成对照数据：把 prompt 分词后逐 token decode，保存每个位置的 logits。
// 用法: oracle <model.gguf> <prompt|@file> <out.bin> [n_extra_greedy] [stride]
// 输出格式: int32 n_tokens, int32 n_vocab, int32 stride, int32 tokens[n], float logits[ceil(n/stride)][n_vocab]
// （只保存 pos % stride == 0 的位置），之后是 int32 n_nll, float nll[n_nll]：每个位置对真实下一 token 的 -log p
// n_extra_greedy > 0 时在 prompt 后继续贪心生成若干 token（同样记录）。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.gguf prompt out.bin [n_extra_greedy]\n", argv[0]);
        return 1;
    }
    std::string prompt = argv[2];
    if (!prompt.empty() && prompt[0] == '@') {
        FILE *pf = fopen(prompt.c_str() + 1, "rb");
        if (!pf) return 1;
        prompt.clear();
        char buf[4096];
        for (size_t r; (r = fread(buf, 1, sizeof(buf), pf)) > 0;) prompt.append(buf, r);
        fclose(pf);
    }
    const int stride = argc > 5 ? atoi(argv[5]) : 1;
    const int n_extra = argc > 4 ? atoi(argv[4]) : 0;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = getenv("ORACLE_CPU") ? 0 : 999;  // ORACLE_CPU=1：纯 CPU 后端
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> tokens(prompt.size() + 16);
    int n = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), false, true);
    if (n < 0) return 1;
    tokens.resize(n);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (tokens.size() + n_extra + 64 + 255) / 256 * 256;  // 按需分配，避免显存不足
    cp.n_batch = 1;
    // 数值敏感性实验：ORACLE_KV_F32=1 用 f32 KV；ORACLE_NO_FA=1 关闭 flash attention
    if (getenv("ORACLE_KV_F32")) cp.type_k = cp.type_v = GGML_TYPE_F32;
    if (getenv("ORACLE_NO_FA")) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.n_ubatch = 1;
    llama_context *ctx = llama_init_from_model(model, cp);

    std::vector<float> logits, nll;
    for (size_t i = 0; i < tokens.size(); i++) {
        llama_token t = tokens[i];
        if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) return 1;
        const float *l = llama_get_logits_ith(ctx, -1);
        if (i % stride == 0) logits.insert(logits.end(), l, l + n_vocab);
        if (i + 1 < tokens.size()) {
            double m = l[0], z = 0;
            for (int v = 1; v < n_vocab; v++) m = l[v] > m ? l[v] : m;
            for (int v = 0; v < n_vocab; v++) z += exp(l[v] - m);
            nll.push_back(float(m + log(z) - l[tokens[i + 1]]));
        }
        if (i + 1 == tokens.size() && (int)(tokens.size() - n) < n_extra) {
            int best = 0;
            for (int v = 1; v < n_vocab; v++)
                if (l[v] > l[best]) best = v;
            tokens.push_back(best);
        }
    }

    FILE *f = fopen(argv[3], "wb");
    int32_t hdr[3] = {(int32_t)tokens.size(), n_vocab, stride};
    fwrite(hdr, 4, 3, f);
    fwrite(tokens.data(), 4, tokens.size(), f);
    fwrite(logits.data(), 4, logits.size(), f);
    const int32_t n_nll = nll.size();
    fwrite(&n_nll, 4, 1, f);
    fwrite(nll.data(), 4, nll.size(), f);
    fclose(f);

    fprintf(stderr, "tokens:");
    if (tokens.size() > 64) fprintf(stderr, " (%zu tokens, showing first 64)", tokens.size());
    for (size_t ti = 0; ti < tokens.size() && ti < 64; ti++) {
        auto t = tokens[ti];
        char buf[64];
        int len = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
        fprintf(stderr, " %d'%.*s'", t, len > 0 ? len : 0, buf);
    }
    fprintf(stderr, "\nsaved %zu positions x %d vocab to %s\n", tokens.size(), n_vocab, argv[3]);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
