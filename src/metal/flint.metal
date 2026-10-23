// Flint decode kernels（v1：先求正确，结构为后续融合留好接口）。
// 量化块定义与 16 元素解码函数复用 ggml 的 kernels/dequantize.h（MIT, ggml authors）。
#include "dequantize.h"
#include "args.h"

// ---------------------------------------------------------------------------
// BF16：包装成 16 元素一块，套进同一个模板
struct block_bf16x16 {
    ushort v[16];
};

template <typename type4x4>
void dequantize_bf16x16(device const block_bf16x16 *xb, short il, thread type4x4 &reg) {
    for (int i = 0; i < 16; i++) reg[i / 4][i % 4] = as_type<float>(uint(xb->v[i]) << 16);
}

// ggml 的 dequantize_q4_K 用 half 算 d/16，d 很小时落入次正规数丢精度；这里改用 float。
template <typename type4x4>
void dequantize_q4_K_f32(device const block_q4_K *xb, short il, thread type4x4 &reg) {
    device const uchar *q = xb->qs;
    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il / 2, xb->scales);
    const float d = il < 2 ? float(xb->d) : float(xb->d) / 16.0f;
    const float dl = d * sc[0];
    const float ml = float(xb->dmin) * sc[1];
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) reg[i / 4][i % 4] = dl * (q[i] & mask) - ml;
}

// QK: 每块元素数；NL = QK / 16
#define DEQ_ARGS device const B *, short, thread float4x4 &
template <typename B, short NL, void (*deq)(DEQ_ARGS)>
inline float dot_chunk(device const uchar *row, uint c, device const float *x) {
    float4x4 w;
    deq((device const B *)row + c / NL, short(c % NL), w);
    device const float4 *xv = (device const float4 *)(x + c * 16);
    return dot(w[0], xv[0]) + dot(w[1], xv[1]) + dot(w[2], xv[2]) + dot(w[3], xv[3]);
}

// ---------------------------------------------------------------------------
// matvec：每个 simdgroup 算 MV_R 行，lane 按 16 元素 chunk 交错步进。
// grid = (ceil(N / (MV_R * MV_SG)), n_mat)，threadgroup = 32 * MV_SG
#define MV_R 2
#define MV_SG 2

template <typename B, short NL, void (*deq)(DEQ_ARGS)>
kernel void k_mv(constant MvArgs &a [[buffer(0)]], device const uchar *W [[buffer(1)]],
                 device const float *x [[buffer(2)]], device float *y [[buffer(3)]],
                 device const int *ids [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                 ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint m = tg.y;
    device const uchar *Wm = W + (a.use_ids ? FL_U64(ids[m]) : FL_U64(m)) * a.mat_stride;
    device const float *xm = x + m * a.x_stride;
    device float *ym = y + m * a.y_stride;
    const uint row0 = (tg.x * MV_SG + sg) * MV_R;
    const uint n_chunks = a.K / 16;

    float acc[MV_R] = {0};
    for (uint c = lane; c < n_chunks; c += 32) {
        for (short r = 0; r < MV_R; r++) {
            if (row0 + r < a.N) acc[r] += dot_chunk<B, NL, deq>(Wm + (row0 + r) * a.row_bytes, c, xm);
        }
    }
    for (short r = 0; r < MV_R; r++) {
        const float v = simd_sum(acc[r]) * a.scale;
        if (lane == 0 && row0 + r < a.N) ym[row0 + r] = a.add ? ym[row0 + r] + v : v;
    }
}

// threadgroup（64 线程）内从 256 个路由 logits 算 sigmoid top-6 并归一化，结果放 threadgroup 内存。
// 每线程负责 4 个专家；并列时取小下标（与 CPU 参考、k_topk 一致）。
inline void tg_topk6(device const float *logits, threadgroup int *t_ids, threadgroup float *t_w,
                     threadgroup float *bv, threadgroup int *bi, ushort tid, ushort sg, ushort lane) {
    float p[4];
    for (short k = 0; k < 4; k++) p[k] = 1.0f / (1.0f + exp(-logits[tid * 4 + k]));
    for (short r = 0; r < 6; r++) {
        float v = p[0];
        int i = tid * 4;
        for (short k = 1; k < 4; k++)
            if (p[k] > v) { v = p[k]; i = tid * 4 + k; }
        for (ushort off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_down(v, off);
            const int oi = simd_shuffle_down(i, off);
            if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
        }
        if (lane == 0) { bv[sg] = v; bi[sg] = i; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const bool second = bv[1] > bv[0] || (bv[1] == bv[0] && bi[1] < bi[0]);
        const int w = second ? bi[1] : bi[0];
        if (tid == 0) { t_ids[r] = w; t_w[r] = second ? bv[1] : bv[0]; }
        if (w / 4 == int(tid))
            for (short k = 0; k < 4; k++)
                if (w % 4 == k) p[k] = -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float sum = 0;
        for (short r = 0; r < 6; r++) sum += t_w[r];
        sum = max(sum, 6.103515625e-5f);
        for (short r = 0; r < 6; r++) t_w[r] /= sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// 专家 gate/up 矩阵乘，topk 融合在开头（省掉单独的 topk 阶段）；顺带把 ids/wts 写回给 down 用
// （gate、up 两个 dispatch 写入相同的值）。grid = (ceil(N / rows_per_tg), 6)，64 线程。
template <typename B, short NL, void (*deq)(DEQ_ARGS)>
kernel void k_mv_topk(constant MvArgs &a [[buffer(0)]], device const uchar *W [[buffer(1)]],
                      device const float *x [[buffer(2)]], device float *y [[buffer(3)]],
                      device const float *logits [[buffer(4)]], device int *ids_out [[buffer(5)]],
                      device float *wts_out [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                      ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup int t_ids[6];
    threadgroup float t_w[6], bv[2];
    threadgroup int bi[2];
    tg_topk6(logits, t_ids, t_w, bv, bi, tid, sg, lane);
    if (tg.x == 0 && tg.y == 0 && tid < 6) { ids_out[tid] = t_ids[tid]; wts_out[tid] = t_w[tid]; }

    const uint m = tg.y;
    device const uchar *Wm = W + FL_U64(t_ids[m]) * a.mat_stride;
    device float *ym = y + m * a.y_stride;
    const uint row0 = (tg.x * MV_SG + sg) * MV_R;
    const uint n_chunks = a.K / 16;
    float acc[MV_R] = {0};
    for (uint c = lane; c < n_chunks; c += 32)
        for (short r = 0; r < MV_R; r++) acc[r] += dot_chunk<B, NL, deq>(Wm + (row0 + r) * a.row_bytes, c, x);
    for (short r = 0; r < MV_R; r++) {
        const float v = simd_sum(acc[r]);
        if (lane == 0) ym[row0 + r] = v;
    }
}

// MoE / SwiGLU down：输入激活 silu(g)*u 在线计算，n_mat 个专家加权求和后写一次。
template <typename B, short NL, void (*deq)(DEQ_ARGS)>
kernel void k_mv_down(constant MvDownArgs &a [[buffer(0)]], device const uchar *W [[buffer(1)]],
                      device const float *g [[buffer(2)]], device const float *u [[buffer(3)]],
                      device float *y [[buffer(4)]], device const int *ids [[buffer(5)]],
                      device const float *wts [[buffer(6)]], uint tg [[threadgroup_position_in_grid]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg * MV_SG + sg) * MV_R;
    const uint n_chunks = a.K / 16;
    float acc[MV_R] = {0};
    for (uint m = 0; m < a.n_mat; m++) {
        device const uchar *Wm = W + (a.use_ids ? FL_U64(ids[m]) : FL_U64(m)) * a.mat_stride;
        device const float *gm = g + m * a.K, *um = u + m * a.K;
        const float wm = a.use_ids ? wts[m] : 1.0f;
        float accm[MV_R] = {0};
        for (uint c = lane; c < n_chunks; c += 32) {
            float4x4 act;
            for (short i = 0; i < 16; i++) {
                const float gv = gm[c * 16 + i];
                act[i / 4][i % 4] = gv / (1.0f + exp(-gv)) * um[c * 16 + i];
            }
            for (short r = 0; r < MV_R; r++) {
                if (row0 + r >= a.N) continue;
                float4x4 w;
                deq((device const B *)(Wm + (row0 + r) * a.row_bytes) + c / NL, short(c % NL), w);
                accm[r] += dot(w[0], act[0]) + dot(w[1], act[1]) + dot(w[2], act[2]) + dot(w[3], act[3]);
            }
        }
        for (short r = 0; r < MV_R; r++) acc[r] += wm * accm[r];
    }
    for (short r = 0; r < MV_R; r++) {
        const float v = simd_sum(acc[r]);
        if (lane == 0 && row0 + r < a.N) y[row0 + r] = a.add ? y[row0 + r] + v : v;
    }
}

// 取一行并解码（embedding 查表；也用来做 dequant 单元测试）。grid = K/16 线程
template <typename B, short NL, void (*deq)(DEQ_ARGS)>
kernel void k_get_row(constant RowArgs &a [[buffer(0)]], device const uchar *W [[buffer(1)]],
                      device float *y [[buffer(2)]], device const int *row_ptr [[buffer(3)]],
                      uint c [[thread_position_in_grid]]) {
    if (c >= a.K / 16) return;
    const uint row = a.row == 0xFFFFFFFFu ? uint(row_ptr[0]) : a.row;  // 0xFFFFFFFF：行号从 GPU buffer 读
    float4x4 w;
    deq((device const B *)(W + FL_U64(row) * a.row_bytes) + c / NL, short(c % NL), w);
    device float4 *yv = (device float4 *)(y + c * 16);
    yv[0] = w[0]; yv[1] = w[1]; yv[2] = w[2]; yv[3] = w[3];
}

#define INSTANTIATE(tname, B, NL, deq)                                                     \
    template [[host_name("mv_" tname)]] kernel void k_mv<B, NL, deq>(                      \
        constant MvArgs &, device const uchar *, device const float *, device float *,     \
        device const int *, uint2, ushort, ushort);                                        \
    template [[host_name("mv_down_" tname)]] kernel void k_mv_down<B, NL, deq>(            \
        constant MvDownArgs &, device const uchar *, device const float *,                 \
        device const float *, device float *, device const int *, device const float *,    \
        uint, ushort, ushort);                                                             \
    template [[host_name("get_row_" tname)]] kernel void k_get_row<B, NL, deq>(            \
        constant RowArgs &, device const uchar *, device float *, device const int *, uint);

#define INST_TOPK(tname, B, NL, deq)                                                       \
    template [[host_name("mv_topk_" tname)]] kernel void k_mv_topk<B, NL, deq>(              \
        constant MvArgs &, device const uchar *, device const float *, device float *,     \
        device const float *, device int *, device float *, uint2, ushort, ushort, ushort);
INST_TOPK("iq2_s", block_iq2_s, 16, dequantize_iq2_s)
INST_TOPK("iq3_xxs", block_iq3_xxs, 16, dequantize_iq3_xxs)
INST_TOPK("iq3_s", block_iq3_s, 16, dequantize_iq3_s)
INST_TOPK("iq4_xs", block_iq4_xs, 16, dequantize_iq4_xs)

INSTANTIATE("q8_0", block_q8_0, 2, dequantize_q8_0)
INSTANTIATE("q4_K", block_q4_K, 16, dequantize_q4_K_f32)
INSTANTIATE("q5_K", block_q5_K, 16, dequantize_q5_K)
INSTANTIATE("q6_K", block_q6_K, 16, dequantize_q6_K)
INSTANTIATE("iq2_s", block_iq2_s, 16, dequantize_iq2_s)
INSTANTIATE("iq3_xxs", block_iq3_xxs, 16, dequantize_iq3_xxs)
INSTANTIATE("iq3_s", block_iq3_s, 16, dequantize_iq3_s)
INSTANTIATE("iq4_xs", block_iq4_xs, 16, dequantize_iq4_xs)
INSTANTIATE("bf16", block_bf16x16, 1, dequantize_bf16x16)

// ---------------------------------------------------------------------------
// RMSNorm：单个 threadgroup（256 线程）
kernel void k_rmsnorm(constant NormArgs &a [[buffer(0)]], device const float *x [[buffer(1)]],
                      device const float *w [[buffer(2)]], device float *y [[buffer(3)]],
                      ushort tid [[thread_position_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]], ushort ntg [[threads_per_threadgroup]]) {
    threadgroup float partial[32];
    float ss = 0;
    for (uint i = tid; i < a.n; i += ntg) ss += x[i] * x[i];
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = lane < ntg / 32 ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = rsqrt(partial[0] / a.n + a.eps);
    for (uint i = tid; i < a.n; i += ntg) y[i] = x[i] * scale * w[i];
}

// ---------------------------------------------------------------------------
// QK RMSNorm + NEOX RoPE + 写 KV cache（f16）。
// grid = n_head + 2 * n_head_kv 个 threadgroup，每个 head_dim(128) 线程：
//   [0, n_head)               q 头：norm + rope，原地写回
//   [n_head, n_head+n_kv)     k 头：norm + rope，写 k_cache[slot]
//   [n_head+n_kv, ...)        v 头：直接写 v_cache[slot]
kernel void k_qk_norm_rope(constant QkArgs &a [[buffer(0)]], device float *q [[buffer(1)]],
                           device const float *k [[buffer(2)]], device const float *v [[buffer(3)]],
                           device const float *q_norm [[buffer(4)]], device const float *k_norm [[buffer(5)]],
                           device half *k_cache [[buffer(6)]], device half *v_cache [[buffer(7)]],
                           uint h [[threadgroup_position_in_grid]], ushort d [[thread_position_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint D = a.head_dim, kv_dim = a.n_head_kv * D;
    if (h >= a.n_head + a.n_head_kv) {
        const uint hv = h - a.n_head - a.n_head_kv;
        v_cache[a.slot * kv_dim + hv * D + d] = half(v[hv * D + d]);
        return;
    }
    const bool is_q = h < a.n_head;
    const uint hh = is_q ? h : h - a.n_head;
    const float val = is_q ? q[hh * D + d] : k[hh * D + d];

    threadgroup float partial[4];
    threadgroup float normed[128];
    float ss = simd_sum(val * val);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float tot = partial[0] + partial[1] + partial[2] + partial[3];
    const float nv = val * rsqrt(tot / D + a.eps) * (is_q ? q_norm[d] : k_norm[d]);

    float out = nv;
    if (a.has_rope) {
        normed[d] = nv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint half_d = D / 2, i = d % half_d;
        const float freq = pow(a.rope_theta, -2.0f * float(i) / float(D));
        const float ang = float(a.pos) * freq, c = cos(ang), s = sin(ang);
        out = d < half_d ? normed[d] * c - normed[d + half_d] * s : normed[d - half_d] * s + normed[d] * c;
    }
    if (is_q) q[hh * D + d] = out;
    else k_cache[a.slot * kv_dim + hh * D + d] = half(out);
}

// ---------------------------------------------------------------------------
// 单 token 注意力（带 learned sink）。grid = n_head 个 threadgroup，每个 8 simdgroup。
// 每个 simdgroup 轮流处理 key；lane 负责 head_dim 的 4 维；在线 softmax 后跨 simdgroup 合并。
#define ATT_SG 8
kernel void k_attn(constant AttnArgs &a [[buffer(0)]], device const float *q [[buffer(1)]],
                   device const half *k_cache [[buffer(2)]], device const half *v_cache [[buffer(3)]],
                   device const float *sinks [[buffer(4)]], device float *out [[buffer(5)]],
                   uint h [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]]) {
    const uint D = a.head_dim, kv_dim = a.n_head_kv * D;
    const uint kvh = h / (a.n_head / a.n_head_kv);
    const float4 qv = ((device const float4 *)(q + h * D))[lane] * a.scale;

    float m = -INFINITY, l = 0;
    float4 acc = 0;
    for (uint j = sg; j < a.n_keys; j += ATT_SG) {
        const uint slot = (a.first_slot + j) % a.n_slots;
        const float4 kv = float4(((device const half4 *)(k_cache + slot * kv_dim + kvh * D))[lane]);
        const float s = simd_sum(dot(qv, kv));
        const float m_new = max(m, s);
        const float corr = exp(m - m_new), p = exp(s - m_new);
        const float4 vv = float4(((device const half4 *)(v_cache + slot * kv_dim + kvh * D))[lane]);
        acc = acc * corr + p * vv;
        l = l * corr + p;
        m = m_new;
    }

    threadgroup float tm[ATT_SG], tl[ATT_SG];
    threadgroup float4 tacc[ATT_SG][32];
    if (lane == 0) { tm[sg] = m; tl[sg] = l; }
    tacc[sg][lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float sink = sinks[h];
        float M = sink;
        for (int i = 0; i < ATT_SG; i++) M = max(M, tm[i]);
        float L = exp(sink - M);
        float4 A = 0;
        for (int i = 0; i < ATT_SG; i++) {
            if (tm[i] == -INFINITY) continue;
            const float c = exp(tm[i] - M);
            L += tl[i] * c;
            A += tacc[i][lane] * c;
        }
        ((device float4 *)(out + h * D))[lane] = A / L;
    }
}

// ---------------------------------------------------------------------------
// 路由：sigmoid → top-k → 选中权重归一化。单 threadgroup，n_expert(256) 线程。
kernel void k_topk(constant TopkArgs &a [[buffer(0)]], device const float *logits [[buffer(1)]],
                   device int *ids [[buffer(2)]], device float *wts [[buffer(3)]],
                   ushort tid [[thread_position_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float best_v[8];
    threadgroup int best_i[8];
    threadgroup float sel_w[16];
    float p = tid < a.n_expert ? 1.0f / (1.0f + exp(-logits[tid])) : -INFINITY;
    const float prob = p;
    for (uint r = 0; r < a.k; r++) {
        // simdgroup 内 argmax（并列时取小下标，与 CPU 参考一致）
        float v = p;
        int i = tid;
        for (ushort off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_down(v, off);
            const int oi = simd_shuffle_down(i, off);
            if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
        }
        if (lane == 0) { best_v[sg] = v; best_i[sg] = i; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float bv = best_v[0];
            int bi = best_i[0];
            for (int s = 1; s < 8; s++)
                if (best_v[s] > bv || (best_v[s] == bv && best_i[s] < bi)) { bv = best_v[s]; bi = best_i[s]; }
            best_i[0] = bi;
            sel_w[r] = bv;
            ids[r] = bi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == uint(best_i[0])) p = -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    (void)prob;
    if (tid == 0) {
        float sum = 0;
        for (uint r = 0; r < a.k; r++) sum += sel_w[r];
        sum = max(sum, 6.103515625e-5f);
        for (uint r = 0; r < a.k; r++) wts[r] = sel_w[r] / sum;
    }
}

// ---------------------------------------------------------------------------
// argmax（贪心采样）：单 threadgroup 1024 线程
kernel void k_argmax(constant uint &n [[buffer(0)]], device const float *x [[buffer(1)]],
                     device int *out [[buffer(2)]], ushort tid [[thread_position_in_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float bv[32];
    threadgroup int bi[32];
    float v = -INFINITY;
    int i = 0;
    for (uint j = tid; j < n; j += 1024)
        if (x[j] > v) { v = x[j]; i = j; }
    for (ushort off = 16; off > 0; off >>= 1) {
        const float ov = simd_shuffle_down(v, off);
        const int oi = simd_shuffle_down(i, off);
        if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    if (lane == 0) { bv[sg] = v; bi[sg] = i; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float best = bv[0];
        int besti = bi[0];
        for (int s = 1; s < 32; s++)
            if (bv[s] > best || (bv[s] == best && bi[s] < besti)) { best = bv[s]; besti = bi[s]; }
        out[0] = besti;
    }
}
