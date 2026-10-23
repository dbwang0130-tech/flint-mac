// Flint prefill kernels：批量矩阵乘（M5 tensor op / Neural Accelerator）等。
// 矩阵乘按 ggml kernel_mul_mm 的 tensor-op 路径改写（MIT, ggml authors）：权重按 16 元素块解码成 half 放进
// threadgroup memory，激活直接作为 device tensor 交给 matmul2d。分块 64 行 × 128 token，K 步长 32。
#define GGML_METAL_HAS_TENSOR
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "dequantize.h"
#include "flint_config.h"
#include "prefill_args.h"

using namespace metal;

// ggml 的 dequantize_q4_K 用 half 算 d/16 会丢精度（见 flint.metal）；这里同样改用 float
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

constant constexpr int MM_ROWS = 64, MM_TOK = 128, MM_TOK_ID = 32, MM_K = 32;

// 一个 64×TOK 输出块：W 从 Wbase 起的第 ra 行开始，X/Y 从各自指针起的第 rb 个 token 开始
// 普通矩阵乘 TOK=128；专家版每个专家分到的 token 少，用 TOK=32 避免空算
template <typename B, short NL, void (*deq)(device const B *, short, thread half4x4 &), int TOK = MM_TOK>
inline void mm_tile(device const uchar *W, uint row_bytes, uint M, uint K, int n_tok, device const float *X,
                    uint x_stride, device float *Y, uint y_stride, int ra, int rb, threadgroup half *sa,
                    ushort tid) {
    auto tB = tensor((device float *)X, dextents<int32_t, 2>(K, n_tok), array<int, 2>({1, int(x_stride)}));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(TOK, MM_ROWS, static_cast<int>(dynamic_extent), false, true, true,
                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tensor(sa, dextents<int32_t, 2>(MM_K, MM_ROWS))), float>();

    for (uint k0 = 0; k0 < K; k0 += MM_K) {
        // 解码 64 行 × 32 列权重：每个线程 1 个 16 元素块（64 行 × 2 块 = 128 个工作项）
        {
            const int row = tid / 2, kc = tid % 2;
            const uint kp = k0 + kc * 16;
            half4x4 w;
            deq((device const B *)(W + uint64_t(ra + row) * row_bytes) + kp / (16 * NL), short((kp / 16) % NL), w);
            for (short i = 0; i < 16; i++) sa[row * MM_K + kc * 16 + i] = w[i / 4][i % 4];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto tAv = tensor(sa, dextents<int32_t, 2>(MM_K, MM_ROWS), array<int, 2>({1, MM_K}));
        auto tBv = tensor((device float *)X + k0 + rb * x_stride, dextents<int32_t, 2>(MM_K, n_tok - rb),
                          array<int, 2>({1, int(x_stride)}));
        mm.run(tBv, tAv, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int tileM = min(MM_ROWS, int(M) - ra), tileN = min(TOK, n_tok - rb);
    auto tD = tensor(Y + uint64_t(rb) * y_stride + ra, dextents<int32_t, 2>(tileM, tileN), array<int, 2>({1, int(y_stride)}));
    cT.store(tD);
}

// grid = (ceil(N / 128), M / 64)
template <typename B, short NL, void (*deq)(device const B *, short, thread half4x4 &)>
kernel void k_mm(constant MmArgs &a [[buffer(0)]], device const uchar *W [[buffer(1)]],
                 device const float *X [[buffer(2)]], device float *Y [[buffer(3)]],
                 threadgroup half *sa [[threadgroup(0)]], uint3 tg [[threadgroup_position_in_grid]],
                 ushort tid [[thread_index_in_threadgroup]]) {
    mm_tile<B, NL, deq>(W, a.row_bytes, a.M, a.K, int(a.N), X, a.x_stride, Y, a.y_stride, int(tg.y) * MM_ROWS,
                        int(tg.x) * MM_TOK, sa, tid);
}

// 专家版：grid = (n_tiles_max, M / 64)。tiles[0] = 实际块数，tiles[1 + 3i ..] = {矩阵号, 段起点, 段长}
// 一个块覆盖某专家段内最多 MM_TOK_ID(32) 个 token；段内 token 在 X / Y 中连续（已按专家排好序）
template <typename B, short NL, void (*deq)(device const B *, short, thread half4x4 &)>
kernel void k_mm_id(constant MmIdArgs &a [[buffer(0)]], device const uchar *W [[buffer(1)]],
                    device const float *X [[buffer(2)]], device float *Y [[buffer(3)]],
                    device const int *tiles [[buffer(4)]], threadgroup half *sa [[threadgroup(0)]],
                    uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    if (int(tg.x) >= tiles[0]) return;
    device const int *t = tiles + 1 + 3 * tg.x;
    const int mat = t[0], seg = t[1], cnt = t[2];
    mm_tile<B, NL, deq, MM_TOK_ID>(W + ulong(mat) * a.mat_stride, a.mm.row_bytes, a.mm.M, a.mm.K, cnt,
                        X + uint64_t(seg) * a.mm.x_stride, a.mm.x_stride, Y + uint64_t(seg) * a.mm.y_stride,
                        a.mm.y_stride, int(tg.y) * MM_ROWS, 0, sa, tid);
}

// ---------------------------------------------------------------------------
// 批量 QK RMSNorm + NEOX RoPE + 写 KV cache（f16）。grid = (T, N_HEAD + 2*N_HEAD_KV)，每个 128 线程（一维一个线程）
kernel void k_qk_prefill(constant QkPfArgs &a [[buffer(0)]], device float *qkv [[buffer(1)]],
                         device const float *q_norm [[buffer(2)]], device const float *k_norm [[buffer(3)]],
                         device half *k_cache [[buffer(4)]], device half *v_cache [[buffer(5)]],
                         uint2 tg [[threadgroup_position_in_grid]], ushort d [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint t = tg.x, h = tg.y, pos = a.p0 + t;
    const uint slot = a.is_full ? pos : (pos & (SWA_RING - 1));
    device float *row = qkv + uint64_t(t) * (Q_DIM + 2 * KV_DIM);
    if (h >= N_HEAD + N_HEAD_KV) {
        const uint hv = h - N_HEAD - N_HEAD_KV;
        v_cache[uint64_t(slot) * KV_DIM + hv * HEAD_DIM + d] = half(row[Q_DIM + KV_DIM + hv * HEAD_DIM + d]);
        return;
    }
    const bool is_q = h < N_HEAD;
    const uint off = is_q ? h * HEAD_DIM : Q_DIM + (h - N_HEAD) * HEAD_DIM;
    const float val = row[off + d];
    threadgroup float partial[4], normed[HEAD_DIM];
    const float ss = simd_sum(val * val);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float nv = val * rsqrt((partial[0] + partial[1] + partial[2] + partial[3]) / HEAD_DIM + RMS_EPS) *
                     (is_q ? q_norm[d] : k_norm[d]);
    float out = nv;
    if (!a.is_full) {
        normed[d] = nv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint hd = HEAD_DIM / 2, i = d % hd;
        const float ang = float(pos) * precise::pow(ROPE_THETA, -2.0f * float(i) / HEAD_DIM);
        const float c = precise::cos(ang), sn = precise::sin(ang);
        out = d < hd ? normed[d] * c - normed[d + hd] * sn : normed[d - hd] * sn + normed[d] * c;
    }
    if (is_q) row[off + d] = out;
    else k_cache[uint64_t(slot) * KV_DIM + (h - N_HEAD) * HEAD_DIM + d] = half(out);
}

// ---------------------------------------------------------------------------
// prefill flash attention（因果 + SWA + learned sink，GQA 8，D=128，f16 KV）。
// grid = (ceil(T/32), N_HEAD)，128 线程 = 4 个 simdgroup，每个负责 8 个 query。
// key 按 32 个一块装进 threadgroup memory；S = Q·Kᵀ 与 O += P·V 用 8×8 simdgroup 矩阵。
// 在线 softmax 初值 m = sink、l = 1：sink 只进分母、不贡献 value。
constant constexpr int AQ = 32, AK = 32, D = HEAD_DIM;
kernel void k_attn_prefill(constant AttnPfArgs &a [[buffer(0)]], device const float *Q [[buffer(1)]],
                           device const half *k_cache [[buffer(2)]], device const half *v_cache [[buffer(3)]],
                           device const float *sinks [[buffer(4)]], device float *out [[buffer(5)]],
                           uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup half Qs[AQ * D], Ks[AK * D], Vs[AK * D];
    threadgroup float Ss[4][8 * AK];
    threadgroup half Ps[4][8 * AK];
    threadgroup float Ds[4][64];  // 每个 simdgroup 的对角缩放矩阵
    threadgroup float Ms[4][8], Ls[4][8];

    const uint h = tg.y, kvh = h / GQA;
    const int q0 = int(tg.x) * AQ;                      // 本块第一个 query 在 chunk 内的下标
    const int nq = min(AQ, int(a.T) - q0);
    const float scale = rsqrt(float(D));

    // 装入 32 个 query（乘上 1/sqrt(D)），不足补 0
    for (int i = tid; i < AQ * D; i += 128) {
        const int r = i / D, c = i % D;
        Qs[i] = r < nq ? half(Q[uint64_t(q0 + r) * a.q_stride + h * D + c] * scale) : half(0);
    }
    if (lane < 8) { Ms[sg][lane] = sinks[h]; Ls[sg][lane] = 1.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 qm[D / 8];
    for (int k = 0; k < D / 8; k++) simdgroup_load(qm[k], Qs + sg * 8 * D + k * 8, D);
    simdgroup_float8x8 om[D / 8];
    for (int k = 0; k < D / 8; k++) om[k] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    // 本块 query 的位置范围 → 需要的 key 位置范围
    const int pq_first = int(a.p0) + q0, pq_last = int(a.p0) + q0 + nq - 1;
    const int k_lo = a.is_full ? 0 : max(0, pq_first - (N_SWA - 1));
    const int k_hi = pq_last;  // 含
    const uint slot_mask = a.is_full ? 0xFFFFFFFFu : uint(SWA_RING - 1);

    for (int kb = k_lo; kb <= k_hi; kb += AK) {
        const int nk = min(AK, k_hi - kb + 1);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < AK * D / 4; i += 128) {
            const int r = (i * 4) / D, c = (i * 4) % D;
            half4 kv = half4(0), vv = half4(0);
            if (r < nk) {
                const uint64_t base = uint64_t(uint(kb + r) & slot_mask) * KV_DIM + kvh * D + c;
                kv = *(device const half4 *)(k_cache + base);
                vv = *(device const half4 *)(v_cache + base);
            }
            *(threadgroup half4 *)(Ks + r * D + c) = kv;
            *(threadgroup half4 *)(Vs + r * D + c) = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S(8×32) = Q(8×128) · Kᵀ(128×32)
        for (int j = 0; j < AK / 8; j++) {
            simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8>(0.0f);
            for (int k = 0; k < D / 8; k++) {
                simdgroup_half8x8 kt;
                simdgroup_load(kt, Ks + j * 8 * D + k * 8, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s, qm[k], kt, s);
            }
            simdgroup_store(s, Ss[sg] + j * 8, AK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // 在线 softmax：lane 负责第 r = lane/4 行的 8 个元素
        {
            const int r = lane / 4, c0 = (lane % 4) * 8;
            const int qpos = pq_first + sg * 8 + r;
            const bool row_valid = sg * 8 + r < nq;
            float v[8], mx = -INFINITY;
            for (int c = 0; c < 8; c++) {
                const int kpos = kb + c0 + c;
                const bool ok = row_valid && c0 + c < nk && kpos <= qpos && (a.is_full || qpos - kpos < N_SWA);
                v[c] = ok ? Ss[sg][r * AK + c0 + c] : -INFINITY;
                mx = max(mx, v[c]);
            }
            mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
            mx = max(mx, simd_shuffle_xor(mx, ushort(2)));
            const float m_old = Ms[sg][r], m_new = max(m_old, mx);
            float sum = 0;
            for (int c = 0; c < 8; c++) {
                const float p = v[c] == -INFINITY ? 0.0f : exp(v[c] - m_new);
                sum += p;
                Ps[sg][r * AK + c0 + c] = half(p);
            }
            sum += simd_shuffle_xor(sum, ushort(1));
            sum += simd_shuffle_xor(sum, ushort(2));
            const float corr = exp(m_old - m_new);
            if (lane % 4 == 0) {
                Ms[sg][r] = m_new;
                Ls[sg][r] = Ls[sg][r] * corr + sum;
            }
            // 对角矩阵 diag(corr)
            for (int c = lane; c < 64; c += 32) Ds[sg][c] = 0.0f;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (lane % 4 == 0) Ds[sg][r * 8 + r] = corr;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(corr)·O + P(8×32)·V(32×128)
        simdgroup_float8x8 dm;
        simdgroup_load(dm, Ds[sg], 8);
        for (int k = 0; k < D / 8; k++) simdgroup_multiply(om[k], dm, om[k]);
        for (int j = 0; j < AK / 8; j++) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, Ps[sg] + j * 8, AK);
            for (int k = 0; k < D / 8; k++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, Vs + j * 8 * D + k * 8, D);
                simdgroup_multiply_accumulate(om[k], pm, vm, om[k]);
            }
        }
    }

    // 输出 O / l：每次把 4 个 8×8 块存进本 simdgroup 的 Ss（8×32），按行除以 l 写回 [T][Q_DIM]
    for (int kk = 0; kk < D / 8; kk += 4) {
        for (int j = 0; j < 4; j++) simdgroup_store(om[kk + j], Ss[sg] + j * 8, AK);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const int r = lane / 4, c0 = (lane % 4) * 8;
        if (sg * 8 + r < nq) {
            const float inv_l = 1.0f / Ls[sg][r];
            device float *o = out + uint64_t(q0 + sg * 8 + r) * Q_DIM + h * D + kk * 8 + c0;
            for (int c = 0; c < 8; c++) o[c] = Ss[sg][r * AK + c0 + c] * inv_l;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------------------
// 按行的小 kernel

// embedding：X[t] = token_embd[tok[t]]（Q8_0）。grid = (N, 1)，128 线程（每线程 16 元素）
kernel void k_embed_rows(device const block_q8_0 *W [[buffer(0)]], device const int *tok [[buffer(1)]],
                         device float *X [[buffer(2)]], uint t [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]]) {
    half4x4 w;
    dequantize_q8_0(W + ulong(tok[t]) * (N_EMBD / 32) + tid / 2, short(tid % 2), w);
    device float4 *o = (device float4 *)(X + uint64_t(t) * N_EMBD + tid * 16);
    for (short i = 0; i < 4; i++) o[i] = float4(w[i]);
}

// RMSNorm：Y[t] = X[t] / rms * w。grid = (N, 1)，256 线程
kernel void k_rmsnorm_rows(device const float *X [[buffer(0)]], device const float *w [[buffer(1)]],
                           device float *Y [[buffer(2)]], uint t [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    device const float4 *x = (device const float4 *)(X + uint64_t(t) * N_EMBD);
    float4 v[2] = {x[tid], x[tid + 256]};
    threadgroup float red[8];
    const float ss = simd_sum(dot(v[0], v[0]) + dot(v[1], v[1]));
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0;
    for (short i = 0; i < 8; i++) tot += red[i];
    const float sc = rsqrt(tot / N_EMBD + RMS_EPS);
    device float4 *y = (device float4 *)(Y + uint64_t(t) * N_EMBD);
    device const float4 *w4 = (device const float4 *)w;
    y[tid] = v[0] * sc * w4[tid];
    y[tid + 256] = v[1] * sc * w4[tid + 256];
}

// X += Y（逐元素，float4）。grid 覆盖 n_rows * width / 4
kernel void k_add_rows(constant RowsArgs &a [[buffer(0)]], device float4 *X [[buffer(1)]],
                       device const float4 *Y [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < a.n_rows * a.width / 4) X[i] += Y[i];
}

// A = silu(G) * U（逐元素，float4）
kernel void k_swiglu(constant RowsArgs &a [[buffer(0)]], device const float4 *G [[buffer(1)]],
                     device const float4 *U [[buffer(2)]], device float4 *A [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= a.n_rows * a.width / 4) return;
    const float4 g = G[i];
    A[i] = g / (1.0f + exp(-g)) * U[i];
}

// 逐 token 路由：sigmoid → top-6 → 归一化。grid = (N, 1)，256 线程（并列取小下标，与 decode 一致）
kernel void k_topk_rows(device const float *logits [[buffer(0)]], device int *ids [[buffer(1)]],
                        device float *wts [[buffer(2)]], uint t [[threadgroup_position_in_grid]],
                        ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float bv[8], sel[N_USED];
    threadgroup int bi[8], winner;
    float p = 1.0f / (1.0f + exp(-logits[uint64_t(t) * N_EXPERT + tid]));
    for (short r = 0; r < N_USED; r++) {
        float v = p;
        int i = tid;
        for (ushort off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_down(v, off);
            const int oi = simd_shuffle_down(i, off);
            if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
        }
        if (lane == 0) { bv[sg] = v; bi[sg] = i; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float b = bv[0];
            int bj = bi[0];
            for (short s2 = 1; s2 < 8; s2++)
                if (bv[s2] > b || (bv[s2] == b && bi[s2] < bj)) { b = bv[s2]; bj = bi[s2]; }
            winner = bj;
            sel[r] = b;
            ids[t * N_USED + r] = bj;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == uint(winner)) p = -INFINITY;
    }
    if (tid == 0) {
        float sum = 0;
        for (short r = 0; r < N_USED; r++) sum += sel[r];
        sum = max(sum, 6.103515625e-5f);
        for (short r = 0; r < N_USED; r++) wts[t * N_USED + r] = sel[r] / sum;
    }
}

// 按专家重排：XG[i] = Xn[src[i]]。grid = (n_pairs, 1)，128 线程
kernel void k_gather_rows(device const float *X [[buffer(0)]], device const int *src [[buffer(1)]],
                          device float *XG [[buffer(2)]], uint i [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]]) {
    device const float4 *x = (device const float4 *)(X + uint64_t(src[i]) * N_EMBD);
    device float4 *y = (device float4 *)(XG + uint64_t(i) * N_EMBD);
    for (uint j = tid; j < N_EMBD / 4; j += 128) y[j] = x[j];
}

// MoE 合并：X[t] += SH[t] + Σ_k w[t][k] · DG[pos[t][k]]。grid = (N, 1)，128 线程
kernel void k_moe_combine(device float *X [[buffer(0)]], device const float *SH [[buffer(1)]],
                          device const float *DG [[buffer(2)]], device const int *pos [[buffer(3)]],
                          device const float *wts [[buffer(4)]], uint t [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]]) {
    device float4 *x = (device float4 *)(X + uint64_t(t) * N_EMBD);
    device const float4 *sh = (device const float4 *)(SH + uint64_t(t) * N_EMBD);
    for (uint j = tid; j < N_EMBD / 4; j += 128) {
        float4 acc = sh[j];
        for (short k = 0; k < N_USED; k++)
            acc += wts[t * N_USED + k] * ((device const float4 *)(DG + uint64_t(pos[t * N_USED + k]) * N_EMBD))[j];
        x[j] += acc;
    }
}

#define T_q8_0 block_q8_0, 2, dequantize_q8_0
#define T_q4_K block_q4_K, 16, dequantize_q4_K_f32
#define T_q5_K block_q5_K, 16, dequantize_q5_K
#define T_q6_K block_q6_K, 16, dequantize_q6_K
#define T_iq2_s block_iq2_s, 16, dequantize_iq2_s
#define T_iq3_xxs block_iq3_xxs, 16, dequantize_iq3_xxs
#define T_iq3_s block_iq3_s, 16, dequantize_iq3_s
#define T_iq4_xs block_iq4_xs, 16, dequantize_iq4_xs
#define T_bf16 block_bf16x16, 1, dequantize_bf16x16

#define INST_MM(t)                                                                                          \
    template [[host_name("mm_" #t)]] kernel void k_mm<T_##t>(constant MmArgs &, device const uchar *,     \
        device const float *, device float *, threadgroup half *, uint3, ushort);                          \
    template [[host_name("mm_id_" #t)]] kernel void k_mm_id<T_##t>(constant MmIdArgs &,                 \
        device const uchar *, device const float *, device float *, device const int *, threadgroup half *, \
        uint3, ushort);

INST_MM(q8_0) INST_MM(q4_K) INST_MM(q5_K) INST_MM(q6_K) INST_MM(iq2_s) INST_MM(iq3_xxs) INST_MM(iq3_s)
INST_MM(iq4_xs) INST_MM(bf16)
