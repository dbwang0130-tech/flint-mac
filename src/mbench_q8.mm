// Q8_0 matvec 带宽 microbenchmark：原始 GGUF 块布局 vs. SoA 重排布局，对照纯读带宽。
// 用法: mbench_q8 [N rows] [K cols] [iters] [copies]
// copies>1 时轮流读互不相同的矩阵副本，避开 SLC 缓存，模拟真实 decode。
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const char *kSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct block_q8_0 { half d; char qs[32]; };

// 纯读带宽：每线程累加若干 float4，防止被优化掉。
kernel void read_bw(device const float4 *src [[buffer(0)]], device float *dst [[buffer(1)]],
                    constant uint &n4 [[buffer(2)]], uint gid [[thread_position_in_grid]],
                    uint gsize [[threads_per_grid]]) {
    float4 acc = 0;
    for (uint i = gid; i < n4; i += gsize) acc += src[i];
    if (acc.x == 12345.0f) dst[gid] = acc.y;
}

// 原始布局：每个 simdgroup 算 ROWS 行，lane 按块步进。
#define ROWS 4
kernel void mv_q8_aos(device const block_q8_0 *w [[buffer(0)]], device const float *x [[buffer(1)]],
                      device float *y [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
                      uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const uint nb = K / 32;
    const uint row0 = (tg * 2 + sg) * ROWS;
    // 4 lane 合作一个块：每 lane 8 个 int8
    const uint sub = lane % 4, bl = lane / 4;
    float acc[ROWS] = {0};
    for (uint b = bl; b < nb; b += 8) {
        device const float4 *xv = (device const float4 *)(x + b * 32 + sub * 8);
        float4 x0 = xv[0], x1 = xv[1];
        for (uint r = 0; r < ROWS; r++) {
            device const block_q8_0 &blk = w[(row0 + r) * nb + b];
            device const char *q = blk.qs + sub * 8;
            float s = dot(float4(q[0], q[1], q[2], q[3]), x0) + dot(float4(q[4], q[5], q[6], q[7]), x1);
            acc[r] += s * float(blk.d);
        }
    }
    for (uint r = 0; r < ROWS; r++) {
        float v = simd_sum(acc[r]);
        if (lane == 0 && row0 + r < N) y[row0 + r] = v;
    }
}

// SoA 布局：qs 为 [N][K] int8 连续、16B 对齐；d 为 [N][K/32] half。
kernel void mv_q8_soa(device const char4 *qs [[buffer(0)]], device const half *d [[buffer(1)]],
                      device const float4 *x [[buffer(2)]], device float *y [[buffer(3)]],
                      constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
                      uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const uint nb = K / 32;
    const uint row0 = (tg * 2 + sg) * ROWS;
    // 每 lane 一次读 16 个 int8（半个块）：lane%2 选半块，lane/2 选块
    const uint half_ = lane % 2, bl = lane / 2;
    float acc[ROWS] = {0};
    for (uint b = bl; b < nb; b += 16) {
        const uint xo = (b * 32 + half_ * 16) / 4;
        float4 x0 = x[xo], x1 = x[xo + 1], x2 = x[xo + 2], x3 = x[xo + 3];
        for (uint r = 0; r < ROWS; r++) {
            const uint row = row0 + r;
            device const char4 *q = qs + (row * K + b * 32 + half_ * 16) / 4;
            char4 q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
            float s = dot(float4(q0), x0) + dot(float4(q1), x1) + dot(float4(q2), x2) + dot(float4(q3), x3);
            acc[r] += s * float(d[row * nb + b]);
        }
    }
    for (uint r = 0; r < ROWS; r++) {
        float v = simd_sum(acc[r]);
        if (lane == 0 && row0 + r < N) y[row0 + r] = v;
    }
}
)METAL";

struct BlockQ8 {
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 34);

static uint16_t f32_to_f16(float f) {
    __fp16 h = (__fp16)f;
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
}

static float f16_to_f32(uint16_t u) {
    __fp16 h;
    memcpy(&h, &u, 2);
    return (float)h;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const uint32_t N = argc > 1 ? atoi(argv[1]) : 150144;
        const uint32_t K = argc > 2 ? atoi(argv[2]) : 2048;
        const uint32_t nb = K / 32;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        printf("device: %s, N=%u K=%u\n", dev.name.UTF8String, N, K);

        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSource) options:nil error:&err];
        if (!lib) { printf("compile error: %s\n", err.localizedDescription.UTF8String); return 1; }
        auto pso = [&](const char *name) {
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(name)] error:&err];
            if (!p) { printf("pso %s: %s\n", name, err.localizedDescription.UTF8String); exit(1); }
            return p;
        };
        id<MTLComputePipelineState> p_read = pso("read_bw"), p_aos = pso("mv_q8_aos"), p_soa = pso("mv_q8_soa");

        // 随机权重
        std::vector<BlockQ8> w((size_t)N * nb);
        std::vector<float> x(K);
        srand(1);
        for (auto &b : w) {
            b.d = f32_to_f16((rand() % 1000) / 100000.0f + 0.001f);
            for (auto &q : b.qs) q = (int8_t)(rand() % 255 - 127);
        }
        for (auto &v : x) v = (rand() % 2000) / 1000.0f - 1.0f;

        const uint32_t copies = argc > 4 ? atoi(argv[4]) : 1;
        size_t aos_bytes = w.size() * sizeof(BlockQ8);
        const size_t qs_bytes = (size_t)N * K, d_bytes = (size_t)N * nb * 2;
        id<MTLBuffer> b_aos = [dev newBufferWithLength:aos_bytes * copies options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_qs = [dev newBufferWithLength:qs_bytes * copies options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_d = [dev newBufferWithLength:d_bytes * copies options:MTLResourceStorageModeShared];
        memcpy(b_aos.contents, w.data(), aos_bytes);
        int8_t *qs = (int8_t *)b_qs.contents;
        uint16_t *dd = (uint16_t *)b_d.contents;
        for (size_t r = 0; r < N; r++)
            for (size_t b = 0; b < nb; b++) {
                const BlockQ8 &blk = w[r * nb + b];
                memcpy(qs + r * K + b * 32, blk.qs, 32);
                dd[r * nb + b] = blk.d;
            }
        for (uint32_t c = 1; c < copies; c++) {
            memcpy((char *)b_aos.contents + c * aos_bytes, b_aos.contents, aos_bytes);
            memcpy((char *)b_qs.contents + c * qs_bytes, b_qs.contents, qs_bytes);
            memcpy((char *)b_d.contents + c * d_bytes, b_d.contents, d_bytes);
        }
        printf("copies=%u, working set %.0f MB\n", copies, aos_bytes * copies / 1e6);
        id<MTLBuffer> b_x = [dev newBufferWithBytes:x.data() length:K * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_y1 = [dev newBufferWithLength:N * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_y2 = [dev newBufferWithLength:N * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_sink = [dev newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> q = [dev newCommandQueue];

        auto run = [&](const char *label, size_t bytes, auto encode) {
            const int iters = argc > 3 ? atoi(argv[3]) : 50;
            for (int warm = 0; warm < 3; warm++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                encode(enc, 0);
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (int i = 0; i < iters; i++) encode(enc, i % copies);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            double sec = (cb.GPUEndTime - cb.GPUStartTime) / iters;
            printf("  %-28s %8.1f us  %6.1f GB/s\n", label, sec * 1e6, bytes / sec / 1e9);
        };

        uint32_t n4 = (uint32_t)(aos_bytes / 16);
        run("read_bw (float4 stream)", (size_t)n4 * 16, [&](id<MTLComputeCommandEncoder> e, uint32_t c) {
            [e setComputePipelineState:p_read];
            [e setBuffer:b_aos offset:c * aos_bytes atIndex:0];
            [e setBuffer:b_sink offset:0 atIndex:1];
            [e setBytes:&n4 length:4 atIndex:2];
            [e dispatchThreads:MTLSizeMake(1 << 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        uint32_t tgs = (N + 2 * 4 - 1) / (2 * 4);
        run("mv_q8_aos (GGUF layout)", aos_bytes, [&](id<MTLComputeCommandEncoder> e, uint32_t c) {
            [e setComputePipelineState:p_aos];
            [e setBuffer:b_aos offset:c * aos_bytes atIndex:0];
            [e setBuffer:b_x offset:0 atIndex:1];
            [e setBuffer:b_y1 offset:0 atIndex:2];
            [e setBytes:&N length:4 atIndex:3];
            [e setBytes:&K length:4 atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        });
        run("mv_q8_soa (repacked)", (size_t)N * K + (size_t)N * nb * 2, [&](id<MTLComputeCommandEncoder> e, uint32_t c) {
            [e setComputePipelineState:p_soa];
            [e setBuffer:b_qs offset:c * qs_bytes atIndex:0];
            [e setBuffer:b_d offset:c * d_bytes atIndex:1];
            [e setBuffer:b_x offset:0 atIndex:2];
            [e setBuffer:b_y2 offset:0 atIndex:3];
            [e setBytes:&N length:4 atIndex:4];
            [e setBytes:&K length:4 atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        });

        // 正确性：CPU 参考
        const float *y1 = (const float *)b_y1.contents, *y2 = (const float *)b_y2.contents;
        double max_err1 = 0, max_err2 = 0;
        for (uint32_t r = 0; r < N; r += 97) {
            double ref = 0;
            for (uint32_t b = 0; b < nb; b++) {
                const BlockQ8 &blk = w[(size_t)r * nb + b];
                double s = 0;
                for (int i = 0; i < 32; i++) s += blk.qs[i] * (double)x[b * 32 + i];
                ref += s * f16_to_f32(blk.d);
            }
            max_err1 = fmax(max_err1, fabs(y1[r] - ref) / (fabs(ref) + 1e-3));
            max_err2 = fmax(max_err2, fabs(y2[r] - ref) / (fabs(ref) + 1e-3));
        }
        printf("  max rel err: aos %.2e, soa %.2e\n", max_err1, max_err2);
    }
    return 0;
}
