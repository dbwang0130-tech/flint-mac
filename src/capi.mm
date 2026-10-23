#include "capi.h"

#include <cstring>
#include <exception>
#include <memory>
#include <string>

#include "engine.h"
#include "gguf.h"
#include "metal/flint_config.h"

struct flint_engine {
    std::unique_ptr<flint::GgufFile> file;
    std::unique_ptr<flint::Engine> eng;
};

static void set_err(char *err, int32_t len, const char *msg) {
    if (err && len > 0) {
        strncpy(err, msg, len - 1);
        err[len - 1] = 0;
    }
}

extern "C" {

flint_engine *flint_open(const flint_config *cfg, char *err, int32_t err_len) {
    try {
        auto h = std::make_unique<flint_engine>();
        h->file = std::make_unique<flint::GgufFile>(cfg->model_path);
        flint::EngineOptions opt;
        const std::string dir = cfg->metallib_dir;
        opt.metallib_path = dir + "/flint.metallib";
        opt.fused_metallib_path = dir + "/fused.metallib";
        opt.prefill_metallib_path = dir + "/prefill.metallib";
        opt.model_path = cfg->model_path;
        opt.pool_gb = cfg->pool_gb;
        if (cfg->n_ctx > 0) opt.n_ctx = cfg->n_ctx;
        h->eng = std::make_unique<flint::Engine>(*h->file, opt);
        return h.release();
    } catch (const std::exception &ex) {
        set_err(err, err_len, ex.what());
        return nullptr;
    }
}

void flint_close(flint_engine *e) { delete e; }

int32_t flint_n_ctx(const flint_engine *e) { return e->eng->n_ctx(); }
int32_t flint_n_vocab(void) { return N_VOCAB; }
int32_t flint_pos(const flint_engine *e) { return e->eng->pos(); }

int32_t flint_truncate(flint_engine *e, int32_t pos) {
    if (pos < 0 || pos > e->eng->pos()) return -1;
    e->eng->truncate(pos);
    return 0;
}

int32_t flint_prefill(flint_engine *e, const int32_t *tokens, int32_t n, char *err, int32_t err_len) {
    try {
        return e->eng->prefill(std::vector<int32_t>(tokens, tokens + n));
    } catch (const std::exception &ex) {
        set_err(err, err_len, ex.what());
        return -1;
    }
}

int32_t flint_decode(flint_engine *e, int32_t token, char *err, int32_t err_len) {
    try {
        return e->eng->step(token);
    } catch (const std::exception &ex) {
        set_err(err, err_len, ex.what());
        return -1;
    }
}

const float *flint_logits(const flint_engine *e) { return e->eng->logits_ptr(); }

}  // extern "C"
