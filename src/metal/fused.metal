// Flint 专用 decode kernels。形状全部来自 flint_config.h（编译期常量），按 Flint 的结构融合：
//   qkv(+attn_norm) | attn(+qk_norm/rope/写 KV) | o_proj(+残差) | router(+ffn_norm) ‖ 共享 gate/up(+norm)
//   | topk | 专家 gate/up(+norm, 输出 silu(g)*u) | 专家 down + 共享 down(+残差)
// 量化块定义与 16 元素解码函数复用 ggml 的 kernels/dequantize.h（MIT, ggml authors）。
#include "dequantize.h"
#include "flint_config.h"
#include "fused_args.h"

// ---------------------------------------------------------------------------
// 解码辅助

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

struct block_bf16x16 {
    ushort v[16];
};

template <typename type4x4>
void dequantize_bf16x16(device const block_bf16x16 *xb, short il, thread type4x4 &reg) {
    for (int i = 0; i < 16; i++) reg[i / 4][i % 4] = as_type<float>(uint(xb->v[i]) << 16);
}

#define DEQ_T(B) void (*deq)(device const B *, short, thread float4x4 &)

// 一行 K 个元素按 16 元素 chunk 编号；chunk c 位于第 c/NL 个块的第 c%NL 段
template <typename B, short NL, DEQ_T(B)>
inline float4x4 deq_chunk(device const uchar *row, uint c) {
    float4x4 w;
    deq((device const B *)row + c / NL, short(c % NL), w);
    return w;
}

template <typename B, short NL, DEQ_T(B), int K>
constexpr uint row_bytes() { return uint(K / (16 * NL) * sizeof(B)); }

inline float dot16(float4x4 a, float4x4 b) {
    return dot(a[0], b[0]) + dot(a[1], b[1]) + dot(a[2], b[2]) + dot(a[3], b[3]);
}

inline float4x4 load16(device const float *p) {
    device const float4 *v = (device const float4 *)p;
    return float4x4(v[0], v[1], v[2], v[3]);
}

inline float silu(float x) { return x / (1.0f + exp(-x)); }

constant constexpr int XC = N_EMBD / 16 / 32;  // 每 lane 负责的 chunk 数（4）

// RMSNorm 融合：写 x 的 kernel 已把 Σx² 原子累加到 ss 槽，这里只读一个标量；
// 各 lane 直接读自己负责的 x 块（与权重读取重叠，无需 barrier、无需整行预扫）。
inline float norm_scale(device const float *ss) { return rsqrt(ss[0] / N_EMBD + RMS_EPS); }

// 16 元素 chunk 布局：lane 负责 c = lane + 32i
inline void x_chunks_normed(device const float *x, device const float *nw, float scale, ushort lane,
                            thread float4x4 *xr) {
    for (short i = 0; i < XC; i++) {
        const uint c = lane + 32 * i;
        const float4x4 xv = load16(x + c * 16), w = load16(nw + c * 16);
        for (short j = 0; j < 4; j++) xr[i][j] = xv[j] * w[j] * scale;
    }
}

// 把 threadgroup 内各线程的值求和后一次原子加到 dst（4 个 simdgroup）
inline void tg_atomic_add(device atomic_float *dst, float v, threadgroup float *red, ushort tid, ushort sg,
                          ushort lane) {
    v = simd_sum(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) atomic_fetch_add_explicit(dst, red[0] + red[1] + red[2] + red[3], memory_order_relaxed);
}

// 对已在寄存器里的 normed x（K = N_EMBD）算 R 行点积
template <typename B, short NL, DEQ_T(B), short R>
inline void dot_rows_x(device const uchar *W, uint row0, thread const float4x4 *xr, ushort lane, thread float *acc) {
    constexpr uint RB = row_bytes<B, NL, deq, N_EMBD>();
    for (short i = 0; i < XC; i++) {
        const uint c = lane + 32 * i;
        for (short r = 0; r < R; r++) acc[r] += dot16(deq_chunk<B, NL, deq>(W + (row0 + r) * RB, c), xr[i]);
    }
}


// ---------------------------------------------------------------------------
// 32 元素子块点积策略：dot32(row, b, y) = Σ_{i<32} W[row][32b+i] * y[i]
// IQ 格式把码本装进 threadgroup memory（constant 内存下标发散时会串行化），每子块只乘一次 scale。
// init 由整个 threadgroup（128 线程）调用，之后需 threadgroup_barrier。
template <typename B, short NL, DEQ_T(B)>
struct DotGeneric {
    static constant constexpr int TBL = 1;
    static void init(threadgroup uchar *, ushort) {}
    static float dot32(device const uchar *row, uint b, thread const float *y, threadgroup const uchar *) {
        const float4x4 w0 = deq_chunk<B, NL, deq>(row, 2 * b), w1 = deq_chunk<B, NL, deq>(row, 2 * b + 1);
        const thread float4 *y4 = (const thread float4 *)y;
        return dot(w0[0], y4[0]) + dot(w0[1], y4[1]) + dot(w0[2], y4[2]) + dot(w0[3], y4[3]) +
               dot(w1[0], y4[4]) + dot(w1[1], y4[5]) + dot(w1[2], y4[6]) + dot(w1[3], y4[7]);
    }
};

inline float sgn(uint bits, short j) { return (bits >> j) & 1 ? -1.0f : 1.0f; }

struct DotIQ3XXS {
    static constant constexpr int TBL = 256 * 4 + 128;  // grid + ksigns
    static void init(threadgroup uchar *t, ushort tid) {
        for (ushort i = tid; i < 256; i += 128) ((threadgroup uint *)t)[i] = iq3xxs_grid[i];
        for (ushort i = tid; i < 128; i += 128) t[1024 + i] = ksigns_iq2xs[i];
    }
    static float dot32(device const uchar *row, uint b, thread const float *y, threadgroup const uchar *t) {
        device const block_iq3_xxs *xb = (device const block_iq3_xxs *)row + b / 8;
        const short ib = b % 8;
        device const uchar *q3 = xb->qs + 8 * ib;
        device const ushort *gas = (device const ushort *)(xb->qs + QK_K / 4) + 2 * ib;
        const uint aux = uint(gas[0]) | (uint(gas[1]) << 16);
        threadgroup const uint *grid = (threadgroup const uint *)t;
        float sum = 0;
        FOR_UNROLL (short l = 0; l < 4; l++) {
            const uchar4 g1 = as_type<uchar4>(grid[q3[2 * l]]), g2 = as_type<uchar4>(grid[q3[2 * l + 1]]);
            const uint sg = t[1024 + ((aux >> (7 * l)) & 127)];
            FOR_UNROLL (short j = 0; j < 4; j++)
                sum += y[8 * l + j] * float(g1[j]) * sgn(sg, j) + y[8 * l + 4 + j] * float(g2[j]) * sgn(sg, j + 4);
        }
        return float(xb->d) * (0.5f + float(aux >> 28)) * 0.5f * sum;
    }
};

struct DotIQ3S {
    static constant constexpr int TBL = 512 * 4;
    static void init(threadgroup uchar *t, ushort tid) {
        for (ushort i = tid; i < 512; i += 128) ((threadgroup uint *)t)[i] = iq3s_grid[i];
    }
    static float dot32(device const uchar *row, uint b, thread const float *y, threadgroup const uchar *t) {
        device const block_iq3_s *xb = (device const block_iq3_s *)row + b / 8;
        const short ib = b % 8;
        device const uchar *qs = xb->qs + 8 * ib;
        device const uchar *signs = xb->signs + 4 * ib;
        const uint qh = xb->qh[ib];
        threadgroup const uint *grid = (threadgroup const uint *)t;
        float sum = 0;
        FOR_UNROLL (short k = 0; k < 8; k++) {
            const uchar4 g = as_type<uchar4>(grid[qs[k] | (((qh >> k) & 1) << 8)]);
            const uint sg = signs[k / 2] >> (4 * (k % 2));
            FOR_UNROLL (short j = 0; j < 4; j++) sum += y[4 * k + j] * float(g[j]) * sgn(sg, j);
        }
        const float dl = float(xb->d) * (1 + 2 * ((xb->scales[ib / 2] >> (4 * (ib % 2))) & 0xf));
        return dl * sum;
    }
};

struct DotIQ2S {
    static constant constexpr int TBL = 1024 * 8;
    static void init(threadgroup uchar *t, ushort tid) {
        for (ushort i = tid; i < 1024; i += 128) ((threadgroup ulong *)t)[i] = iq2s_grid[i];
    }
    static float dot32(device const uchar *row, uint b, thread const float *y, threadgroup const uchar *t) {
        device const block_iq2_s *xb = (device const block_iq2_s *)row + b / 8;
        const short ib = b % 8;
        device const uchar *qs = xb->qs + 4 * ib;
        device const uchar *signs = xb->qs + QK_K / 8 + 4 * ib;
        const uint qh = xb->qh[ib];
        threadgroup const ulong *grid = (threadgroup const ulong *)t;
        float sum[2] = {0, 0};
        FOR_UNROLL (short k = 0; k < 4; k++) {
            const ulong gv = grid[qs[k] | (((qh >> (2 * k)) & 3) << 8)];
            const uint sg = signs[k];
            float s = 0;
            FOR_UNROLL (short j = 0; j < 8; j++) s += y[8 * k + j] * float((gv >> (8 * j)) & 0xff) * sgn(sg, j);
            sum[k / 2] += s;
        }
        const float d = float(xb->d) * 0.25f;
        const uint sc = xb->scales[ib];
        return d * ((0.5f + float(sc & 0xf)) * sum[0] + (0.5f + float(sc >> 4)) * sum[1]);
    }
};

struct DotIQ4XS {
    static constant constexpr int TBL = 16 * 4;
    static void init(threadgroup uchar *t, ushort tid) {
        if (tid < 16) ((threadgroup float *)t)[tid] = kvalues_iq4nl_f[tid];
    }
    static float dot32(device const uchar *row, uint b, thread const float *y, threadgroup const uchar *t) {
        device const block_iq4_xs *xb = (device const block_iq4_xs *)row + b / 8;
        const short ib = b % 8;
        device const uint *q4 = (device const uint *)xb->qs + 4 * ib;
        threadgroup const float *kv = (threadgroup const float *)t;
        float sum = 0;
        FOR_UNROLL (short i = 0; i < 4; i++) {
            const uint q = q4[i];
            FOR_UNROLL (short k = 0; k < 4; k++) {
                sum += y[4 * i + k] * kv[(q >> (8 * k)) & 0xf];
                sum += y[16 + 4 * i + k] * kv[(q >> (8 * k + 4)) & 0xf];
            }
        }
        const int ls = ((xb->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((xb->scales_h >> (2 * ib)) & 3) << 4);
        return float(xb->d) * float(ls - 32) * sum;
    }
};

// ---------------------------------------------------------------------------
// ① QKV：rows [0,4096) q，[4096,4608) k，[4608,5120) v，输出到连续的 qkv buffer
#define QKV_R 4
#define QKV_SG 4
template <typename B, short NL, DEQ_T(B)>
kernel void k_qkv(device const uchar *Wq [[buffer(0)]], device const uchar *Wk [[buffer(1)]],
                  device const uchar *Wv [[buffer(2)]], device const float *x [[buffer(3)]],
                  device const float *nw [[buffer(4)]], device float *qkv [[buffer(5)]],
                  device const float *ss [[buffer(6)]], uint tg [[threadgroup_position_in_grid]],
                  ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg * QKV_SG + sg) * QKV_R;
    device const uchar *W = row0 < Q_DIM ? Wq : row0 < Q_DIM + KV_DIM ? Wk : Wv;
    const uint r0 = row0 < Q_DIM ? row0 : row0 < Q_DIM + KV_DIM ? row0 - Q_DIM : row0 - Q_DIM - KV_DIM;
    float4x4 xr[XC];
    x_chunks_normed(x, nw, norm_scale(ss), lane, xr);
    float acc[QKV_R] = {0};
    dot_rows_x<B, NL, deq, QKV_R>(W, r0, xr, lane, acc);
    for (short r = 0; r < QKV_R; r++) {
        const float v = simd_sum(acc[r]);
        if (lane == 0) qkv[row0 + r] = v;
    }
}

// ---------------------------------------------------------------------------
// ② 单 token 注意力：融合 QK RMSNorm、NEOX RoPE（SWA 层）、learned sink、写 KV cache。
// grid = (N_HEAD, n_split)，每个 threadgroup ATT_SG 个 simdgroup；lane 负责 head_dim 的 4 维。
// key 位置 [k_begin, pos) 从 cache 读，当前 token 的 k/v 由最后一个 split 用寄存器里的值处理。
#define ATT_SG 4

// lane 持有维度 4*lane..4*lane+3；NEOX 配对维度 i 与 i+64 在 lane 与 lane^16 之间
inline float4 norm_rope(float4 v, device const float *nw, ushort lane, uint pos, bool rope) {
    v *= rsqrt(simd_sum(dot(v, v)) / HEAD_DIM + RMS_EPS) * ((device const float4 *)nw)[lane];
    if (!rope) return v;
    const float4 partner = simd_shuffle_xor(v, ushort(16));
    float4 out;
    for (short j = 0; j < 4; j++) {
        const float i = float(4 * (lane % 16) + j);
        const float ang = float(pos) * precise::pow(ROPE_THETA, -2.0f * i / HEAD_DIM);
        const float c = precise::cos(ang), s = precise::sin(ang);
        out[j] = lane < 16 ? v[j] * c - partner[j] * s : partner[j] * s + v[j] * c;
    }
    return out;
}

kernel void k_attn_dec(constant AttnDecArgs &a [[buffer(0)]], device const float *qkv [[buffer(1)]],
                       device const float *q_norm [[buffer(2)]], device const float *k_norm [[buffer(3)]],
                       device const float *sinks [[buffer(4)]], device half *k_cache [[buffer(5)]],
                       device half *v_cache [[buffer(6)]], device float *out [[buffer(7)]],
                       device float *partial [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint h = tg.x, split = tg.y, kvh = h / GQA;
    const bool rope = !a.is_full;
    const float4 qv = norm_rope(((device const float4 *)(qkv + h * HEAD_DIM))[lane], q_norm, lane, a.pos, rope) *
                      rsqrt(float(HEAD_DIM));

    const uint k_begin = a.is_full ? 0 : uint(max(int(a.pos) - (N_SWA - 1), 0));
    const uint j0 = k_begin + split * a.keys_per_split;
    const uint j1 = min(a.pos, j0 + a.keys_per_split);
    const bool last_split = split == a.n_split - 1;
    const uint slot_mask = a.is_full ? 0xFFFFFFFFu : uint(SWA_RING - 1);

    float m = -INFINITY, l = 0;
    float4 acc = 0;
    for (uint j = j0 + sg; j < j1; j += ATT_SG) {
        const uint slot = j & slot_mask;
        const float4 kv = float4(((device const half4 *)(k_cache + slot * KV_DIM + kvh * HEAD_DIM))[lane]);
        const float s = simd_sum(dot(qv, kv));
        const float m_new = max(m, s);
        const float corr = exp(m - m_new), p = exp(s - m_new);
        const float4 vv = float4(((device const half4 *)(v_cache + slot * KV_DIM + kvh * HEAD_DIM))[lane]);
        acc = acc * corr + p * vv;
        l = l * corr + p;
        m = m_new;
    }
    if (last_split && sg == ATT_SG - 1) {
        // 当前 token：k/v 先按 f16 舍入，与写入 cache、llama.cpp 的行为一致
        const float4 kraw = ((device const float4 *)(qkv + Q_DIM + kvh * HEAD_DIM))[lane];
        const half4 kh = half4(norm_rope(kraw, k_norm, lane, a.pos, rope));
        const half4 vh = half4(((device const float4 *)(qkv + Q_DIM + KV_DIM + kvh * HEAD_DIM))[lane]);
        const float s = simd_sum(dot(qv, float4(kh)));
        const float m_new = max(m, s);
        const float corr = exp(m - m_new), p = exp(s - m_new);
        acc = acc * corr + p * float4(vh);
        l = l * corr + p;
        m = m_new;
        if (h % GQA == 0) {
            const uint slot = a.pos & slot_mask;
            ((device half4 *)(k_cache + slot * KV_DIM + kvh * HEAD_DIM))[lane] = kh;
            ((device half4 *)(v_cache + slot * KV_DIM + kvh * HEAD_DIM))[lane] = vh;
        }
    }

    threadgroup float tm[ATT_SG], tl[ATT_SG];
    threadgroup float4 tacc[ATT_SG][32];
    if (lane == 0) { tm[sg] = m; tl[sg] = l; }
    tacc[sg][lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;

    float M = -INFINITY;
    for (int i = 0; i < ATT_SG; i++) M = max(M, tm[i]);
    float L = 0;
    float4 A = 0;
    for (int i = 0; i < ATT_SG; i++) {
        if (tl[i] == 0) continue;
        const float c = exp(tm[i] - M);
        L += tl[i] * c;
        A += tacc[i][lane] * c;
    }
    if (a.n_split == 1) {
        const float sink = sinks[h];
        const float Mf = max(M, sink), cs = exp(M - Mf);
        ((device float4 *)(out + h * HEAD_DIM))[lane] = A * cs / (L * cs + exp(sink - Mf));
    } else {
        // 部分结果：[h][split] = { acc[128], m, l }
        device float *p = partial + (h * a.n_split + split) * (HEAD_DIM + 2);
        ((device float4 *)p)[lane] = A;
        if (lane == 0) { p[HEAD_DIM] = M; p[HEAD_DIM + 1] = L; }
    }
}

// 合并各 split 的部分结果并加 sink。grid = N_HEAD，每个 32 线程
kernel void k_attn_combine(constant AttnDecArgs &a [[buffer(0)]], device const float *partial [[buffer(1)]],
                           device const float *sinks [[buffer(2)]], device float *out [[buffer(3)]],
                           uint h [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    device const float *p0 = partial + h * a.n_split * (HEAD_DIM + 2);
    const float sink = sinks[h];
    float M = sink;
    for (uint s = 0; s < a.n_split; s++) M = max(M, p0[s * (HEAD_DIM + 2) + HEAD_DIM]);
    float L = exp(sink - M);
    float4 A = 0;
    for (uint s = 0; s < a.n_split; s++) {
        device const float *p = p0 + s * (HEAD_DIM + 2);
        const float ls = p[HEAD_DIM + 1];
        if (ls == 0) continue;
        const float c = exp(p[HEAD_DIM] - M);
        L += ls * c;
        A += ((device const float4 *)p)[lane] * c;
    }
    ((device float4 *)(out + h * HEAD_DIM))[lane] = A / L;
}

// ---------------------------------------------------------------------------
// ③ 矩阵乘 + 残差：x[r] += dot(W[r], in)，K 为编译期常量（o_proj 4096 / dense down 6144）
#define RES_R 2
#define RES_SG 4
template <typename B, short NL, DEQ_T(B), int K>
kernel void k_mv_res(device const uchar *W [[buffer(0)]], device const float *in [[buffer(1)]],
                     device float *x [[buffer(2)]], device atomic_float *ss_out [[buffer(3)]],
                     uint tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr uint RB = row_bytes<B, NL, deq, K>();
    const uint row0 = (tg * RES_SG + sg) * RES_R;
    float acc[RES_R] = {0};
    for (uint c = lane; c < K / 16; c += 32) {
        const float4x4 xv = load16(in + c * 16);
        for (short r = 0; r < RES_R; r++) acc[r] += dot16(deq_chunk<B, NL, deq>(W + (row0 + r) * RB, c), xv);
    }
    float sq = 0;
    for (short r = 0; r < RES_R; r++) {
        const float v = simd_sum(acc[r]);
        if (lane == 0) {
            const float nx = x[row0 + r] + v;
            x[row0 + r] = nx;
            sq += nx * nx;
        }
    }
    threadgroup float red[RES_SG];
    tg_atomic_add(ss_out, sq, red, tid, sg, lane);
}

// ---------------------------------------------------------------------------
// ④ 路由 logits（BF16，融合 ffn_norm）
#define ROUTER_R 2
#define ROUTER_SG 4
kernel void k_router(device const uchar *W [[buffer(0)]], device const float *x [[buffer(1)]],
                     device const float *nw [[buffer(2)]], device float *logits [[buffer(3)]],
                     device const float *ss [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg * ROUTER_SG + sg) * ROUTER_R;
    float4x4 xr[XC];
    x_chunks_normed(x, nw, norm_scale(ss), lane, xr);
    float acc[ROUTER_R] = {0};
    dot_rows_x<block_bf16x16, 1, dequantize_bf16x16, ROUTER_R>(W, row0, xr, lane, acc);
    for (short r = 0; r < ROUTER_R; r++) {
        const float v = simd_sum(acc[r]);
        if (lane == 0) logits[row0 + r] = v;
    }
}

// ⑤ sigmoid → top-6 → 选中权重归一化。单 threadgroup，N_EXPERT 线程
kernel void k_topk(device const float *logits [[buffer(0)]], device int *ids [[buffer(1)]],
                   device float *wts [[buffer(2)]], ushort tid [[thread_position_in_threadgroup]],
                   ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float best_v[N_EXPERT / 32];
    threadgroup int best_i[N_EXPERT / 32];
    threadgroup float sel_w[N_USED];
    threadgroup int winner;
    float p = 1.0f / (1.0f + exp(-logits[tid]));
    for (short r = 0; r < N_USED; r++) {
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
            for (short s = 1; s < N_EXPERT / 32; s++)
                if (best_v[s] > bv || (best_v[s] == bv && best_i[s] < bi)) { bv = best_v[s]; bi = best_i[s]; }
            winner = bi;
            sel_w[r] = bv;
            ids[r] = bi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == uint(winner)) p = -INFINITY;
    }
    if (tid == 0) {
        float sum = 0;
        for (short r = 0; r < N_USED; r++) sum += sel_w[r];
        sum = max(sum, 6.103515625e-5f);
        for (short r = 0; r < N_USED; r++) wts[r] = sel_w[r] / sum;
    }
}

// ---------------------------------------------------------------------------
// ⑥ gate/up 成对计算（融合 ffn_norm），直接输出 act = silu(gate·x) * (up·x)
// grid = (n_rows / (GU_R*GU_SG), n_mat)；专家：n_mat=6、use_ids=1；共享/dense：n_mat=1
// lane 负责 32 元素子块 b = lane + 32i（K=2048 → 每 lane 2 个子块）
#define GU_R 2
#define GU_SG 4
constant constexpr int XB = N_EMBD / 32 / 32;  // 每 lane 的子块数（2）
template <typename P, int RB>
kernel void k_gateup(constant GateUpArgs &a [[buffer(0)]], device const uchar *Wg [[buffer(1)]],
                     device const uchar *Wu [[buffer(2)]], device const float *x [[buffer(3)]],
                     device const float *nw [[buffer(4)]], device float *act [[buffer(5)]],
                     device const int *ids [[buffer(6)]], device const float *ss [[buffer(7)]],
                     uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup uchar tbl[P::TBL];
    P::init(tbl, tid);
    const float scale = norm_scale(ss);
    float y[XB][32];
    FOR_UNROLL (short i = 0; i < XB; i++)
        FOR_UNROLL (short j = 0; j < 32; j++) {
            const uint e = (lane + 32 * i) * 32 + j;
            y[i][j] = x[e] * nw[e] * scale;
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // 码本就绪

    const uint m = tg.y;
    const ulong mat = ulong(a.use_ids ? ids[m] : 0) * a.n_rows * RB;
    const uint row0 = (tg.x * GU_SG + sg) * GU_R;
    float g[GU_R] = {0}, u[GU_R] = {0};
    FOR_UNROLL (short i = 0; i < XB; i++) {
        const uint b = lane + 32 * i;
        FOR_UNROLL (short r = 0; r < GU_R; r++) {
            g[r] += P::dot32(Wg + mat + (row0 + r) * RB, b, y[i], tbl);
            u[r] += P::dot32(Wu + mat + (row0 + r) * RB, b, y[i], tbl);
        }
    }
    FOR_UNROLL (short r = 0; r < GU_R; r++) {
        const float gv = simd_sum(g[r]), uv = simd_sum(u[r]);
        if (lane == 0) act[m * a.n_rows + row0 + r] = silu(gv) * uv;
    }
}

// ---------------------------------------------------------------------------
// ⑦ 专家 down（6 个，加权）+ 共享专家 down + 残差，一次写回 x
// 路由专家 K=512 = 16 个子块：lane 0-15 算第 row0 行，16-31 算 row0+1 行
#define DN_SG 4
template <typename PE, int RBE, typename PS, int RBS>
kernel void k_down_moe(device const uchar *We [[buffer(0)]], device const uchar *Ws [[buffer(1)]],
                       device const float *act [[buffer(2)]], device const float *act_sh [[buffer(3)]],
                       device const int *ids [[buffer(4)]], device const float *wts [[buffer(5)]],
                       device float *x [[buffer(6)]], device atomic_float *ss_out [[buffer(7)]],
                       uint tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup uchar tbl_e[PE::TBL], tbl_s[PS::TBL];
    PE::init(tbl_e, tid);
    PS::init(tbl_s, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row0 = (tg * DN_SG + sg) * 2;
    const uint my_row = row0 + lane / 16, b = lane % 16;
    float acc_e = 0;
    FOR_UNROLL (short m = 0; m < N_USED; m++) {
        device const uchar *Wm = We + ulong(ids[m]) * N_EMBD * RBE;
        float y[32];
        FOR_UNROLL (short j = 0; j < 32; j++) y[j] = act[m * N_FF_EXP + b * 32 + j];
        acc_e += wts[m] * PE::dot32(Wm + my_row * RBE, b, y, tbl_e);
    }
    // 共享专家 K=1024 = 32 个子块：每 lane 一个，两行都算
    float y[32];
    FOR_UNROLL (short j = 0; j < 32; j++) y[j] = act_sh[lane * 32 + j];
    const float s0 = PS::dot32(Ws + row0 * RBS, lane, y, tbl_s);
    const float s1 = PS::dot32(Ws + (row0 + 1) * RBS, lane, y, tbl_s);
    const float v0 = simd_sum(lane < 16 ? acc_e : 0.0f) + simd_sum(s0);
    const float v1 = simd_sum(lane >= 16 ? acc_e : 0.0f) + simd_sum(s1);
    float sq = 0;
    if (lane == 0) {
        const float n0 = x[row0] + v0, n1 = x[row0 + 1] + v1;
        x[row0] = n0;
        x[row0 + 1] = n1;
        sq = n0 * n0 + n1 * n1;
    }
    threadgroup float red[DN_SG];
    tg_atomic_add(ss_out, sq, red, tid, sg, lane);
}

// ---------------------------------------------------------------------------
// lm_head（Q8_0，融合 output_norm）+ argmax
#define LM_R 4
#define LM_SG 4
kernel void k_lmhead(device const uchar *W [[buffer(0)]], device const float *x [[buffer(1)]],
                     device const float *nw [[buffer(2)]], device float *logits [[buffer(3)]],
                     device const float *ss [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg * LM_SG + sg) * LM_R;
    float4x4 xr[XC];
    x_chunks_normed(x, nw, norm_scale(ss), lane, xr);
    float acc[LM_R] = {0};
    dot_rows_x<block_q8_0, 2, dequantize_q8_0, LM_R>(W, row0, xr, lane, acc);
    for (short r = 0; r < LM_R; r++) {
        const float v = simd_sum(acc[r]);
        if (lane == 0) logits[row0 + r] = v;
    }
}

kernel void k_argmax(device const float *x [[buffer(0)]], device int *out [[buffer(1)]],
                     ushort tid [[thread_position_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                     ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float bv[32];
    threadgroup int bi[32];
    float v = -INFINITY;
    int i = 0;
    for (uint j = tid; j < N_VOCAB; j += 1024)
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
        for (short s = 1; s < 32; s++)
            if (bv[s] > best || (bv[s] == best && bi[s] < besti)) { best = bv[s]; besti = bi[s]; }
        out[0] = besti;
    }
}

// ---------------------------------------------------------------------------
// embedding（Q8_0 查表）。单 threadgroup 128 线程：写 x 与本 token 的 attn_norm 平方和（槽 0），
// 并清零另一奇偶的全部槽（供下一个 token 原子累加）。
kernel void k_embed(device const block_q8_0 *W [[buffer(0)]], device const int *tok [[buffer(1)]],
                    device float *x [[buffer(2)]], device float *ss_cur [[buffer(3)]],
                    device float *ss_next [[buffer(4)]], ushort tid [[thread_index_in_threadgroup]],
                    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    float4x4 w;
    dequantize_q8_0(W + ulong(tok[0]) * (N_EMBD / 32) + tid / 2, short(tid % 2), w);
    device float4 *xv = (device float4 *)(x + tid * 16);
    xv[0] = w[0]; xv[1] = w[1]; xv[2] = w[2]; xv[3] = w[3];
    float sq = simd_sum(dot16(w, w));
    threadgroup float red[4];
    if (lane == 0) red[sg] = sq;
    for (uint i = tid; i < SS_SLOTS; i += 128) ss_next[i] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) ss_cur[0] = red[0] + red[1] + red[2] + red[3];
}

// ---------------------------------------------------------------------------
// 取一行并解码（embedding 查表；check-deq 单元测试）。grid = K/16 线程
template <typename B, short NL, DEQ_T(B)>
kernel void k_get_row(constant FRowArgs &a [[buffer(0)]], device const uchar *W [[buffer(1)]],
                      device float *y [[buffer(2)]], device const int *row_ptr [[buffer(3)]],
                      uint c [[thread_position_in_grid]]) {
    if (c >= a.K / 16) return;
    const uint row = a.row == 0xFFFFFFFFu ? uint(row_ptr[0]) : a.row;
    const float4x4 w = deq_chunk<B, NL, deq>(W + ulong(row) * a.row_bytes, c);
    device float4 *yv = (device float4 *)(y + c * 16);
    yv[0] = w[0]; yv[1] = w[1]; yv[2] = w[2]; yv[3] = w[3];
}

// ---------------------------------------------------------------------------
// 实例化：只生成这个 GGUF 里实际出现的 (角色, 类型) 组合
#define T_q8_0 block_q8_0, 2, dequantize_q8_0
#define T_q4_K block_q4_K, 16, dequantize_q4_K_f32
#define T_q5_K block_q5_K, 16, dequantize_q5_K
#define T_q6_K block_q6_K, 16, dequantize_q6_K
#define T_iq2_s block_iq2_s, 16, dequantize_iq2_s
#define T_iq3_xxs block_iq3_xxs, 16, dequantize_iq3_xxs
#define T_iq3_s block_iq3_s, 16, dequantize_iq3_s
#define T_iq4_xs block_iq4_xs, 16, dequantize_iq4_xs
#define T_bf16 block_bf16x16, 1, dequantize_bf16x16

#define INST_QKV(t)                                                                                          \
    template [[host_name("qkv_" #t)]] kernel void k_qkv<T_##t>(device const uchar *, device const uchar *, \
        device const uchar *, device const float *, device const float *, device float *, device const float *, \
        uint, ushort, ushort);
#define INST_RES(t, K)                                                                                        \
    template [[host_name("mv_res_" #t "_" #K)]] kernel void k_mv_res<T_##t, K>(device const uchar *,       \
        device const float *, device float *, device atomic_float *, uint, ushort, ushort, ushort);
#define P_q8_0 DotGeneric<T_q8_0>
#define P_q4_K DotGeneric<T_q4_K>
#define P_q5_K DotGeneric<T_q5_K>
#define P_q6_K DotGeneric<T_q6_K>
#define P_iq2_s DotIQ2S
#define P_iq3_xxs DotIQ3XXS
#define P_iq3_s DotIQ3S
#define P_iq4_xs DotIQ4XS
#define RB_(t, K) (row_bytes<T_##t, K>())
#define INST_GU(t)                                                                                            \
    template [[host_name("gateup_" #t)]] kernel void k_gateup<P_##t, RB_(t, N_EMBD)>(constant GateUpArgs &, \
        device const uchar *, device const uchar *, device const float *, device const float *,            \
        device float *, device const int *, device const float *, uint2, ushort, ushort, ushort);
#define INST_DN(te, ts)                                                                                       \
    template [[host_name("down_" #te "_" #ts)]] kernel void                                                \
        k_down_moe<P_##te, RB_(te, N_FF_EXP), P_##ts, RB_(ts, N_FF_SH)>(device const uchar *,              \
        device const uchar *, device const float *, device const float *, device const int *,              \
        device const float *, device float *, device atomic_float *, uint, ushort, ushort, ushort);
#define INST_ROW(t)                                                                                           \
    template [[host_name("get_row_" #t)]] kernel void k_get_row<T_##t>(constant FRowArgs &,                \
        device const uchar *, device float *, device const int *, uint);

// 注意力 / 共享专家权重类型
#define FOR_ATTN_TYPES(X) X(q8_0) X(q5_K) X(q6_K) X(iq4_xs)
FOR_ATTN_TYPES(INST_QKV)
INST_RES(q8_0, 4096) INST_RES(q5_K, 4096) INST_RES(q6_K, 4096) INST_RES(iq4_xs, 4096)
INST_RES(q8_0, 6144)
// gate/up：共享专家 + dense（同注意力类型）与路由专家类型的并集
FOR_ATTN_TYPES(INST_GU) INST_GU(iq2_s) INST_GU(iq3_xxs) INST_GU(iq3_s)
// down：路由专家类型 × 共享专家类型
#define INST_DN_E(ts) INST_DN(iq4_xs, ts) INST_DN(iq3_s, ts) INST_DN(q4_K, ts)
FOR_ATTN_TYPES(INST_DN_E)
// get_row：所有类型
FOR_ATTN_TYPES(INST_ROW) INST_ROW(q4_K) INST_ROW(iq2_s) INST_ROW(iq3_xxs) INST_ROW(iq3_s) INST_ROW(bf16)
