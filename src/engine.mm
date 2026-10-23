#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <sys/mman.h>
#include <cerrno>

#include "engine.h"

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>

#include "metal/args.h"
#include "metal/flint_config.h"
#include "metal/fused_args.h"
#include "metal/prefill_args.h"

#include <set>

namespace flint {

namespace {

constexpr int MV_R = 2, MV_SG = 2;  // 与 flint.metal 一致

const char *kernel_type_name(GgmlType t) {
    switch (t) {
        case GgmlType::Q8_0: return "q8_0";
        case GgmlType::Q4_K: return "q4_K";
        case GgmlType::Q5_K: return "q5_K";
        case GgmlType::Q6_K: return "q6_K";
        case GgmlType::IQ2_S: return "iq2_s";
        case GgmlType::IQ3_XXS: return "iq3_xxs";
        case GgmlType::IQ3_S: return "iq3_s";
        case GgmlType::IQ4_XS: return "iq4_xs";
        case GgmlType::BF16: return "bf16";
        default: throw std::runtime_error(std::string("no kernel for type ") + type_traits(t).name);
    }
}

}  // namespace

struct Engine::Impl {
    // 权重在 Metal buffer 里的位置
    struct W {
        id<MTLBuffer> buf;
        uint64_t off;
        const Tensor *t;
        uint32_t K() const { return uint32_t(t->dims[0]); }
        uint32_t N() const { return uint32_t(t->dims[1]); }
        uint32_t row_bytes() const { return uint32_t(t->row_bytes()); }
        uint64_t mstride = 0;  // 流式模式：专家槽大小
        uint64_t mat_stride() const { return mstride ? mstride : t->dims[1] * t->row_bytes(); }
    };

    struct Layer {
        W attn_norm, q, k, v, o, q_norm, k_norm, sinks, ffn_norm;
        W gate, up, down;                        // dense 层
        W router, gate_exps, up_exps, down_exps;  // MoE 层
        W gate_sh, up_sh, down_sh;
        id<MTLBuffer> k_cache, v_cache;
        uint32_t n_slots;
    };

    const GgufFile &f;
    const HParams &hp;
    EngineOptions opt;

    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> lib, lib_fused, lib_prefill;
    // prefill 中间缓冲区（按调用位置复用）及其常驻集合
    id<MTLBuffer> pf_bufs[19];
    id<MTLResidencySet> pf_res;
    id<MTLResidencySet> residency;
    std::map<std::string, id<MTLComputePipelineState>> psos;

    std::vector<id<MTLBuffer>> weight_bufs;
    std::vector<std::pair<uint64_t, uint64_t>> weight_ranges;  // 文件内 [start, end)
    W tok_embd, out_norm, output;
    std::vector<Layer> layers;

    // ---- 专家流式加载 ----
    bool streaming = false;
    id<MTLBuffer> resident, sids;  // 常驻的非专家权重 / 专家槽池 / 每层槽号 [N_LAYER][6]
    std::map<std::string, uint64_t> resident_off;
    struct ExpertSrc {
        uint64_t file_off[3];  // gate/up/down 张量在文件中的起点（专家 0）
        uint32_t bytes[3];     // 每个专家各部分字节数
    };
    std::vector<ExpertSrc> esrc;  // [N_LAYER]
    // 槽池按专家尺寸分级：同尺寸的层共用一级，每级固定槽大小、各自 LRU（避免按最大专家分槽浪费 ~25%）
    struct SizeClass {
        uint64_t bytes = 0;  // 槽大小（16 KB 对齐）
        uint64_t base = 0;   // 在 pool 中的字节偏移（每级独立 buffer，恒为 0）
        id<MTLBuffer> buf;   // 每级一个 buffer（单个 MTLBuffer 有 maxBufferLength 上限）
        int first = 0, count = 0;  // 全局槽号区间 [first, first+count)
        int head = -1, tail = -1;  // LRU（头 = 最近）
        int n_layers = 0;
    };
    std::vector<SizeClass> cls;
    std::vector<int> layer_cls = std::vector<int>(N_LAYER, -1);
    int n_slots = 0;
    std::vector<int32_t> slot_of, owner, lru_prev, lru_next, slot_cls;  // slot_of[layer*256+e] = 全局槽号
    int fd_direct = -1;
    std::vector<dispatch_group_t> slot_group;  // [n_slots] 该槽最近一次异步写入；nil 表示就绪
    id<MTLSharedEvent> ev;
    uint64_t ev_base = 0;

    Engine::StreamStats st;

    static bool is_expert_tensor(const std::string &n) { return n.find("_exps.") != std::string::npos; }

    void lru_unlink(int s) {
        SizeClass &c = cls[slot_cls[s]];
        if (lru_prev[s] >= 0) lru_next[lru_prev[s]] = lru_next[s]; else c.head = lru_next[s];
        if (lru_next[s] >= 0) lru_prev[lru_next[s]] = lru_prev[s]; else c.tail = lru_prev[s];
    }
    void lru_push_front(int s) {
        SizeClass &c = cls[slot_cls[s]];
        lru_prev[s] = -1;
        lru_next[s] = c.head;
        if (c.head >= 0) lru_prev[c.head] = s;
        c.head = s;
        if (c.tail < 0) c.tail = s;
    }
    void lru_touch(int s) { lru_unlink(s); lru_push_front(s); }

    void wait_slot(int sl) {
        if (!slot_group[sl]) return;
        dispatch_group_wait(slot_group[sl], DISPATCH_TIME_FOREVER);
        slot_group[sl] = nil;
    }

    // prefill 期间按层顺序扫过全部专家，是缓存放不下的循环访问：LRU 每次都会全部重读。
    // 此时改为淘汰"最近用过"的（MRU），跳过受保护的层（当前层、下一层），让缓存稳定保留其余层
    bool pf_mru = false;
    int pf_protect[2] = {-1, -1};

    int pick_victim(SizeClass &c) {
        if (!pf_mru || owner[c.tail] < 0) return c.tail;  // 空槽优先
        for (int s = c.head; s >= 0; s = lru_next[s]) {
            const int ol = owner[s] < 0 ? -1 : owner[s] / N_EXPERT;
            if (ol != pf_protect[0] && ol != pf_protect[1]) return s;
        }
        return c.tail;
    }

    // 在该层所属尺寸级选一个槽（见 pick_victim），异步装入 (il, e)：若该槽还有未完成的写入，先等它。返回全局槽号
    int load_expert(int il, int e) {
        SizeClass &c = cls[layer_cls[il]];
        const int sl = pick_victim(c);
        wait_slot(sl);
        dispatch_group_t group = dispatch_group_create();
        slot_group[sl] = group;
        if (owner[sl] >= 0) slot_of[owner[sl]] = -1;
        const int key = il * N_EXPERT + e;
        owner[sl] = key;
        slot_of[key] = sl;
        lru_touch(sl);
        uint8_t *dst = (uint8_t *)c.buf.contents + uint64_t(sl - c.first) * c.bytes;
        uint64_t part_off = 0;
        const int fd = fd_direct;
        for (int p = 0; p < 3; p++) {
            const uint32_t n = esrc[il].bytes[p];
            const uint64_t src = esrc[il].file_off[p] + uint64_t(e) * n;
            uint8_t *d = dst + part_off;
            dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^{
                uint64_t done = 0;
                while (done < n) {
                    const ssize_t r = pread(fd, d + done, n - done, off_t(src + done));
                    if (r <= 0) break;
                    done += uint64_t(r);
                }
            });
            part_off += n;
            st.bytes += n;
        }
        return sl;
    }

    // prefill 用：后台装入第 l 层全部未缓存的专家（prompt 稍长几乎每个专家都会用到）。
    // 若要淘汰的槽属于 protect 层（GPU 可能正在用），停止，以免覆盖正在计算的权重
    int prefetch_layer(int l, int protect) {
        if (l >= N_LAYER || l < N_DENSE) return 0;
        int n = 0;
        for (int ex = 0; ex < N_EXPERT; ex++) {
            const int key = l * N_EXPERT + ex;
            if (slot_of[key] >= 0) continue;
            const int v = pick_victim(cls[layer_cls[l]]);
            if (owner[v] >= 0 && (owner[v] / N_EXPERT == protect || owner[v] / N_EXPERT == l)) break;
            load_expert(l, ex);
            n++;
        }
        return n;
    }

    // 非专家权重拷进常驻 buffer；专家只记录文件位置，按尺寸分级分配槽池
    void setup_streaming() {
        uint64_t total = 0;
        for (const auto &t : f.tensors())
            if (!is_expert_tensor(t.name)) total += (t.nbytes + 255) / 256 * 256;
        resident = [dev newBufferWithLength:total options:MTLResourceStorageModeShared];
        uint64_t off = 0;
        for (const auto &t : f.tensors()) {
            if (is_expert_tensor(t.name)) continue;
            memcpy((uint8_t *)resident.contents + off, t.data, t.nbytes);
            resident_off[t.name] = off;
            off += (t.nbytes + 255) / 256 * 256;
        }
        esrc.assign(N_LAYER, ExpertSrc{});
        uint64_t all_expert_bytes = 0;
        std::vector<uint64_t> layer_sz(N_LAYER, 0);
        for (int il = N_DENSE; il < N_LAYER; il++) {
            const char *parts[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
            uint64_t sum = 0;
            for (int p = 0; p < 3; p++) {
                const Tensor &t = f.tensor("blk." + std::to_string(il) + "." + parts[p]);
                esrc[il].file_off[p] = t.offset;
                esrc[il].bytes[p] = uint32_t(t.nbytes / N_EXPERT);
                sum += esrc[il].bytes[p];
            }
            layer_sz[il] = (sum + 16383) / 16384 * 16384;
            all_expert_bytes += layer_sz[il] * N_EXPERT;
        }
        // 每级缓存同样比例的专家；同尺寸的层再按单个 buffer 上限分组（每组一个 buffer、一套 LRU）
        if (opt.pool_gb < 0) {
            // auto：取 ① GPU 推荐工作集 − 常驻 − KV − 0.6 GB 与 ② 系统当前可用内存 − 1.5 GB 的较小值。
            // 只看 ① 会把其他进程（虚拟机等）挤进 swap：实测 16.5 GB 池使 swap 用到 19 GB、速度暴跌。
            const double kv = 2.0 * 2 * KV_DIM * (14.0 * opt.n_ctx + 35.0 * SWA_RING);
            const double by_gpu = double(dev.recommendedMaxWorkingSetSize) - double(total) - kv - 1.0e9;  // 余量含 prefill 中间缓冲区
            vm_statistics64_data_t vm{};
            mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
            host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &cnt);
            const double page = double(vm_page_size);
            const double avail = (double(vm.free_count) + vm.inactive_count + vm.purgeable_count + vm.speculative_count) * page;
            const double by_mem = avail - 1.5e9;
            opt.pool_gb = std::max(1.0, std::min(by_gpu, by_mem) / 1e9);
            fprintf(stderr, "auto pool: GPU 上限 %.1f GB，可用内存 %.1f GB → 槽池 %.1f GB\n", by_gpu / 1e9, avail / 1e9, opt.pool_gb);
        }
        const double frac = std::min(1.0, opt.pool_gb * 1e9 / double(all_expert_bytes));
        const double max_buf = double(dev.maxBufferLength) * 0.95;
        for (int il = N_DENSE; il < N_LAYER; il++) {
            const uint64_t sz = layer_sz[il];
            int c = 0;
            while (c < int(cls.size()) && !(cls[c].bytes == sz && frac * (cls[c].n_layers + 1) * N_EXPERT * sz <= max_buf)) c++;
            if (c == int(cls.size())) cls.push_back(SizeClass{sz});
            cls[c].n_layers++;
            layer_cls[il] = c;
        }
        // 小尺寸级（层数少）给满容量：容量不足一层（256）时 prefill 要分波、每批重读；
        // 这些级合计只占几百 MB，剩余预算再按比例分给大级
        const double budget = std::min(double(all_expert_bytes), opt.pool_gb * 1e9);
        double small_bytes = 0, big_bytes = 0;
        for (auto &c : cls) {
            const double full = double(c.n_layers) * N_EXPERT * c.bytes;
            (full <= 1.5e9 ? small_bytes : big_bytes) += full;
        }
        const double big_frac = big_bytes > 0 ? std::clamp((budget - small_bytes) / big_bytes, 0.0, 1.0) : 1.0;
        uint64_t pool_bytes = 0;
        bool mlock_warned = false;
        for (auto &c : cls) {
            c.first = n_slots;
            const double full = double(c.n_layers) * N_EXPERT * c.bytes;
            c.count = full <= 1.5e9 ? c.n_layers * N_EXPERT : std::max(2 * N_EXPERT, int(big_frac * c.n_layers * N_EXPERT));
            c.buf = [dev newBufferWithLength:uint64_t(c.count) * c.bytes options:MTLResourceStorageModeShared];
            if (!c.buf) throw std::runtime_error("cannot allocate expert pool");
            // 锁住物理页：槽池被系统压缩/换出后 GPU 访问会缺页，速度崩溃（实测 13.5 GB 池时 3 tok/s）
            if (mlock(c.buf.contents, c.buf.length) != 0) {
                if (!mlock_warned) fprintf(stderr, "warning: mlock expert pool failed (%s)\n", strerror(errno));
                mlock_warned = true;
            }
            n_slots += c.count;
            pool_bytes += uint64_t(c.count) * c.bytes;
        }
        slot_of.assign(N_LAYER * N_EXPERT, -1);
        owner.assign(n_slots, -1);
        lru_prev.assign(n_slots, -1);
        lru_next.assign(n_slots, -1);
        slot_cls.assign(n_slots, 0);
        for (int ci = 0; ci < int(cls.size()); ci++)
            for (int i = cls[ci].first; i < cls[ci].first + cls[ci].count; i++) {
                slot_cls[i] = ci;
                lru_push_front(i);
            }
        sids = [dev newBufferWithLength:N_LAYER * N_USED * 4 options:MTLResourceStorageModeShared];
        slot_group.assign(n_slots, nil);

        fd_direct = open(opt.model_path.c_str(), O_RDONLY);
        if (fd_direct < 0) throw std::runtime_error("open for pread: " + opt.model_path);
        fcntl(fd_direct, F_NOCACHE, 1);
        ev = [dev newSharedEvent];
        st.n_slots = n_slots;
        st.slot_mb = pool_bytes / double(n_slots) / 1048576.0;
        fprintf(stderr, "streaming: resident %.2f GB, pool %.2f GB, %d slots in %zu size classes (%.0f%% of experts)\n",
                total / 1e9, pool_bytes / 1e9, n_slots, cls.size(), 100.0 * n_slots / ((N_LAYER - N_DENSE) * N_EXPERT));
    }

    // 第 il 层 topk 完成后：查 LRU，未命中的专家并发 pread 进槽，写槽号
    void stream_layer(int il) {
        const double t0 = CACurrentMediaTime();
        static const bool fake = getenv("FLINT_STREAM_FAKE") != nullptr;  // 计时实验：不装载，只量握手开销
        if (fake) {
            int *sid = (int *)sids.contents + il * N_USED;
            for (int k = 0; k < N_USED; k++) sid[k] = k;
            static const int fake_us = getenv("FLINT_STREAM_FAKE_US") ? atoi(getenv("FLINT_STREAM_FAKE_US")) : 0;
            if (fake_us) {  // 忙等模拟装载耗时（不让出 CPU）
                const double until = CACurrentMediaTime() + fake_us * 1e-6;
                while (CACurrentMediaTime() < until) {}
            }
            return;
        }
        const int *eid = (const int *)ids.contents + il * N_USED;
        int *sid = (int *)sids.contents + il * N_USED;
        // 1) 本层实际路由：命中（含已预取）直接用；未命中同步读
        std::vector<int> need;  // 本层要用、且仍在预取中的槽
        for (int k = 0; k < N_USED; k++) {
            const int key = il * N_EXPERT + eid[k];
            if (slot_of[key] >= 0) {
                const int sl = slot_of[key];
                lru_touch(sl);
                sid[k] = sl - cls[layer_cls[il]].first;  // GPU 用级内编号
                if (slot_group[sl]) need.push_back(sl);
                st.hits++;
                continue;
            }
            const int sl = load_expert(il, eid[k]);
            sid[k] = sl - cls[layer_cls[il]].first;
            need.push_back(sl);
            st.misses++;
        }
        // 2) 预取下一层预测的专家（与加急读一起占用 SSD，不等待）
        auto prefetch = [&](int l, const int *pid, int k_n) {
            for (int k = 0; k < k_n; k++) {
                const int key = l * N_EXPERT + pid[k];
                if (slot_of[key] >= 0) continue;
                load_expert(l, pid[k]);
                st.prefetched++;
            }
        };
        if (il + 1 < N_LAYER) prefetch(il + 1, (const int *)pids.contents + (il + 1) * PK, pred_k);
        if (il + 2 < N_LAYER && pred2_k > 0) prefetch(il + 2, (const int *)pids2.contents + (il + 2) * PK, pred2_k);
        for (int sl : need) wait_slot(sl);
        st.load_ms += (CACurrentMediaTime() - t0) * 1e3;
    }

    // 专家张量在槽池中的视图：W = 所属尺寸级起点 + 部件偏移，矩阵步长 = 该级槽大小，ids 换成级内槽号
    W expert_view(int il, int part, const Tensor *t) {
        const SizeClass &c = cls[layer_cls[il]];
        uint64_t off = c.base;
        for (int p = 0; p < part; p++) off += esrc[il].bytes[p];
        W w{c.buf, off, t};
        w.mstride = c.bytes;
        return w;
    }

    // 激活
    // 专家预测：pids[l] = 用第 l 层 router 作用在第 l-1 层 ffn 输入上的 top-PK（按分数降序）；pids2[l] 来自第 l-2 层
    static constexpr int PK = 10;
    id<MTLBuffer> rlogits_pred, pids, pwts, rlogits_pred2, pids2, pwts2;
    id<MTLBuffer> x, xn, qkv, partial, att, rlogits, ids, wts, g, u, gs, us, logits, out_tok, tok;

    Impl(const GgufFile &f_, const HParams &hp_, const EngineOptions &o) : f(f_), hp(hp_), opt(o) {
        dev = MTLCreateSystemDefaultDevice();
        queue = [dev newCommandQueue];
        NSError *err = nil;
        lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(opt.metallib_path.c_str())] error:&err];
        if (!lib) throw std::runtime_error("load metallib: " + std::string(err.localizedDescription.UTF8String));
        lib_fused = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(opt.fused_metallib_path.c_str())] error:&err];
        if (!lib_fused) throw std::runtime_error("load fused metallib: " + std::string(err.localizedDescription.UTF8String));
        {
            MTLResidencySetDescriptor *d = [MTLResidencySetDescriptor new];
            NSError *e2 = nil;
            pf_res = [dev newResidencySetWithDescriptor:d error:&e2];  // 只挂在 prefill 的 command buffer 上
        }
        lib_prefill = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(opt.prefill_metallib_path.c_str())] error:&err];
        if (!lib_prefill) throw std::runtime_error("load prefill metallib: " + std::string(err.localizedDescription.UTF8String));
        streaming = opt.pool_gb != 0;  // <0 表示自动
        if (streaming) setup_streaming();
        else map_weights();
        bind_layers();
        alloc_activations();
        if (streaming) {
            MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
            NSError *e2 = nil;
            residency = [dev newResidencySetWithDescriptor:rd error:&e2];
            [residency addAllocation:resident];
            for (auto &c : cls) [residency addAllocation:c.buf];
            [residency commit];
            [residency requestResidency];
            [queue addResidencySet:residency];
        }
    }

    id<MTLComputePipelineState> pso(const std::string &name) {
        auto it = psos.find(name);
        if (it != psos.end()) return it->second;
        id<MTLFunction> fn = [lib newFunctionWithName:@(name.c_str())];
        if (!fn) fn = [lib_fused newFunctionWithName:@(name.c_str())];
        if (!fn) fn = [lib_prefill newFunctionWithName:@(name.c_str())];
        if (!fn) throw std::runtime_error("missing kernel " + name);
        NSError *err = nil;
        id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!p) throw std::runtime_error("pso " + name + ": " + err.localizedDescription.UTF8String);
        return psos[name] = p;
    }

    // 把 mmap 的文件按 maxBufferLength 切块，零拷贝包成 MTLBuffer。
    void map_weights() {
        const uint64_t page = getpagesize();
        const uint64_t max_len = dev.maxBufferLength;
        std::vector<const Tensor *> ts;
        for (const auto &t : f.tensors()) ts.push_back(&t);
        std::sort(ts.begin(), ts.end(), [](auto a, auto b) { return a->offset < b->offset; });

        uint64_t start = 0, end = 0;
        auto flush = [&]() {
            if (end <= start) return;
            const uint64_t len = std::min<uint64_t>((end - start + page - 1) / page * page, f.size() - start);
            const uint64_t len_aligned = (len + page - 1) / page * page;
            id<MTLBuffer> b = [dev newBufferWithBytesNoCopy:(void *)(f.base() + start)
                                                     length:len_aligned
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
            if (!b) throw std::runtime_error("newBufferWithBytesNoCopy failed");
            weight_bufs.push_back(b);
            weight_ranges.push_back({start, start + len_aligned});
        };
        bool first = true;
        for (const Tensor *t : ts) {
            const uint64_t t_start = t->offset / page * page, t_end = t->offset + t->nbytes;
            if (first) { start = t_start; end = t_end; first = false; continue; }
            if (t_end - start > max_len) {
                flush();
                start = t_start;
            }
            end = std::max(end, t_end);
        }
        flush();

        MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
        rd.label = @"flint-weights";
        NSError *err = nil;
        residency = [dev newResidencySetWithDescriptor:rd error:&err];
        for (id<MTLBuffer> b : weight_bufs) [residency addAllocation:b];
        [residency commit];
        [residency requestResidency];
        [queue addResidencySet:residency];
    }

    W weight(const std::string &name) {
        const Tensor &t = f.tensor(name);
        if (streaming) {
            auto it = resident_off.find(name);
            if (it == resident_off.end()) throw std::runtime_error("expert tensor has no resident copy: " + name);
            return W{resident, it->second, &t};
        }
        for (size_t i = 0; i < weight_bufs.size(); i++) {
            auto [s, e] = weight_ranges[i];
            if (t.offset >= s && t.offset + t.nbytes <= e) return W{weight_bufs[i], t.offset - s, &t};
        }
        throw std::runtime_error("tensor not mapped: " + name);
    }

    void bind_layers() {
        tok_embd = weight("token_embd.weight");
        out_norm = weight("output_norm.weight");
        output = weight("output.weight");
        const uint32_t kv_dim = hp.n_head_kv * hp.head_dim;
        for (int il = 0; il < hp.n_layer; il++) {
            auto w = [&](const char *n) { return weight("blk." + std::to_string(il) + "." + n); };
            Layer L{};
            L.attn_norm = w("attn_norm.weight");
            L.q = w("attn_q.weight");
            L.k = w("attn_k.weight");
            L.v = w("attn_v.weight");
            L.o = w("attn_output.weight");
            L.q_norm = w("attn_q_norm.weight");
            L.k_norm = w("attn_k_norm.weight");
            L.sinks = w("attn_sinks.weight");
            L.ffn_norm = w("ffn_norm.weight");
            if (il < hp.n_dense_lead) {
                L.gate = w("ffn_gate.weight");
                L.up = w("ffn_up.weight");
                L.down = w("ffn_down.weight");
            } else {
                L.router = w("ffn_gate_inp.weight");
                if (streaming) {
                    const std::string b = "blk." + std::to_string(il) + ".";
                    L.gate_exps = expert_view(il, 0, &f.tensor(b + "ffn_gate_exps.weight"));
                    L.up_exps = expert_view(il, 1, &f.tensor(b + "ffn_up_exps.weight"));
                    L.down_exps = expert_view(il, 2, &f.tensor(b + "ffn_down_exps.weight"));
                } else {
                    L.gate_exps = w("ffn_gate_exps.weight");
                    L.up_exps = w("ffn_up_exps.weight");
                    L.down_exps = w("ffn_down_exps.weight");
                }
                L.gate_sh = w("ffn_gate_shexp.weight");
                L.up_sh = w("ffn_up_shexp.weight");
                L.down_sh = w("ffn_down_shexp.weight");
            }
            L.n_slots = hp.is_swa[il] ? SWA_RING : opt.n_ctx;
            L.k_cache = [dev newBufferWithLength:size_t(L.n_slots) * kv_dim * 2 options:MTLResourceStorageModePrivate];
            L.v_cache = [dev newBufferWithLength:size_t(L.n_slots) * kv_dim * 2 options:MTLResourceStorageModePrivate];
            layers.push_back(L);
        }
    }

    void alloc_activations() {
        auto buf = [&](size_t floats) { return [dev newBufferWithLength:floats * 4 options:MTLResourceStorageModeShared]; };
        const int E = hp.n_embd, QD = hp.n_head * hp.head_dim, KVD = hp.n_head_kv * hp.head_dim;
        const int n_ff_max = std::max({hp.n_ff_dense, hp.n_ff_exp * hp.n_expert_used, hp.n_ff_shexp});
        x = buf(E); xn = buf(E); qkv = buf(QD + 2 * KVD); att = buf(QD);
        partial = buf(size_t(N_HEAD) * 32 * (HEAD_DIM + 2));
        rlogits = buf(hp.n_expert); ids = buf(N_LAYER * N_USED); wts = buf(N_LAYER * N_USED);
        rlogits_pred = buf(hp.n_expert); pids = buf(N_LAYER * PK); pwts = buf(N_LAYER * PK);
        rlogits_pred2 = buf(hp.n_expert); pids2 = buf(N_LAYER * PK); pwts2 = buf(N_LAYER * PK);  // 每层各自一段，便于记录路由
        g = buf(n_ff_max); u = buf(n_ff_max); gs = buf(hp.n_ff_shexp); us = buf(hp.n_ff_shexp);
        logits = buf(hp.n_vocab); out_tok = buf(1); tok = buf(1);
    }


    // ---- profile：每次调度单独一个 encoder，前后打 GPU 时间戳 ----
    bool profiling = false;
    id<MTLCommandBuffer> cur_cb;
    id<MTLCounterSampleBuffer> samples;
    std::vector<std::string> prof_labels;
    static constexpr int kMaxProf = 2048;

    static std::string role(const W &w) {
        std::string n = w.t->name;
        if (n.rfind("blk.", 0) == 0) n = n.substr(n.find('.', 4) + 1);
        return n.substr(0, n.rfind(".weight"));
    }

    // FLINT_OPT=attn,...：逐项打开优化，用于同一二进制内 A/B
    std::set<std::string> opts = [] {
        std::set<std::string> v;
        if (!getenv("FLINT_OPT")) v.insert("attn");  // 默认：融合注意力（实测更快、长上下文必需）
        if (const char *e = getenv("FLINT_OPT")) {
            std::string s = e;
            for (size_t i = 0, j; i <= s.size(); i = j + 1) {
                j = s.find(',', i);
                if (j == std::string::npos) j = s.size();
                if (j > i) v.insert(s.substr(i, j - i));
            }
        }
        return v;
    }();
    bool on(const char *o) const { return opts.count(o) != 0; }
    bool predict() const { return streaming || on("pred"); }
    // 预取个数：提前一层取预测的前 pred_k 个；提前两层取前 pred2_k 个（0 = 关闭，也不计算两层预测）
    const int pred_k = getenv("FLINT_PRED_K") ? atoi(getenv("FLINT_PRED_K")) : 8;  // 实测 6/8/10、加两层预取差别在噪声内，取 8
    const int pred2_k = getenv("FLINT_PRED2_K") ? atoi(getenv("FLINT_PRED2_K")) : (on("pred") ? PK : 0);

    void enc_topk(id<MTLComputeCommandEncoder> __strong &e, id<MTLBuffer> lg, id<MTLBuffer> out_ids, id<MTLBuffer> out_w, uint64_t off, uint32_t k = N_USED) {
        TopkArgs ta{uint32_t(hp.n_expert), k};
        prof(e, "k_topk");
        [e setComputePipelineState:pso("k_topk")];
        [e setBytes:&ta length:sizeof(ta) atIndex:0];
        [e setBuffer:lg offset:0 atIndex:1];
        [e setBuffer:out_ids offset:off atIndex:2];
        [e setBuffer:out_w offset:off atIndex:3];
        if (!skip_cur) [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(hp.n_expert, 1, 1)];
    }

    std::vector<std::string> skip_list = [] {
        std::vector<std::string> v;
        if (const char *e = getenv("FLINT_SKIP")) {
            std::string s = e;
            for (size_t i = 0, j; i <= s.size(); i = j + 1) {
                j = s.find(',', i);
                if (j == std::string::npos) j = s.size();
                if (j > i) v.push_back(s.substr(i, j - i));
            }
        }
        return v;
    }();
    bool skip_cur = false;
    uint64_t ids_off = 0;  // 当前层在 ids/wts 中的字节偏移

    void prof(id<MTLComputeCommandEncoder> __strong &e, const std::string &label) {
        skip_cur = false;
        for (const auto &p : skip_list)
            if (label.find(p) != std::string::npos) skip_cur = true;
        if (!profiling) return;
        [e endEncoding];
        const int i = int(prof_labels.size());
        if (i >= kMaxProf) throw std::runtime_error("too many profiled dispatches");
        prof_labels.push_back(label);
        MTLComputePassDescriptor *pd = [MTLComputePassDescriptor computePassDescriptor];
        pd.sampleBufferAttachments[0].sampleBuffer = samples;
        pd.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2 * i;
        pd.sampleBufferAttachments[0].endOfEncoderSampleIndex = 2 * i + 1;
        e = [cur_cb computeCommandEncoderWithDescriptor:pd];
    }

    void init_profiler() {
        id<MTLCounterSet> ts = nil;
        for (id<MTLCounterSet> cs in dev.counterSets)
            if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) ts = cs;
        if (!ts || ![dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])
            throw std::runtime_error("GPU timestamp counters not supported");
        MTLCounterSampleBufferDescriptor *d = [MTLCounterSampleBufferDescriptor new];
        d.counterSet = ts;
        d.storageMode = MTLStorageModeShared;
        d.sampleCount = 2 * kMaxProf;
        NSError *err = nil;
        samples = [dev newCounterSampleBufferWithDescriptor:d error:&err];
        if (!samples) throw std::runtime_error("counter sample buffer: " + std::string(err.localizedDescription.UTF8String));
    }

    // ---- 编码辅助 ----
    void enc_rmsnorm(id<MTLComputeCommandEncoder> __strong &e, id<MTLBuffer> in, const W &w, id<MTLBuffer> out, uint32_t n) {
        prof(e, "rmsnorm");
        NormArgs a{n, hp.rms_eps};
        [e setComputePipelineState:pso("k_rmsnorm")];
        [e setBytes:&a length:sizeof(a) atIndex:0];
        [e setBuffer:in offset:0 atIndex:1];
        [e setBuffer:w.buf offset:w.off atIndex:2];
        [e setBuffer:out offset:0 atIndex:3];
        if (!skip_cur) [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    void enc_mv(id<MTLComputeCommandEncoder> __strong &e, const W &w, id<MTLBuffer> in, id<MTLBuffer> out, bool add = false,
                uint32_t n_mat = 1, bool use_ids = false, uint32_t x_stride = 0, uint64_t out_off = 0) {
        prof(e, std::string("mv_") + kernel_type_name(w.t->type) + " " + role(w));
        MvArgs a{};
        a.mat_stride = w.mat_stride();
        a.K = w.K();
        a.N = w.N();
        a.row_bytes = w.row_bytes();
        a.x_stride = x_stride;
        a.y_stride = a.N;
        a.use_ids = use_ids;
        a.add = add;
        a.scale = 1.0f;
        [e setComputePipelineState:pso(std::string("mv_") + kernel_type_name(w.t->type))];
        [e setBytes:&a length:sizeof(a) atIndex:0];
        [e setBuffer:w.buf offset:w.off atIndex:1];
        [e setBuffer:in offset:0 atIndex:2];
        [e setBuffer:out offset:out_off atIndex:3];
        [e setBuffer:ids offset:ids_off atIndex:4];
        const uint32_t rows_per_tg = MV_R * MV_SG;
        if (!skip_cur) [e dispatchThreadgroups:MTLSizeMake((a.N + rows_per_tg - 1) / rows_per_tg, n_mat, 1)
            threadsPerThreadgroup:MTLSizeMake(32 * MV_SG, 1, 1)];
    }

    void enc_mv_topk(id<MTLComputeCommandEncoder> __strong &e, const W &w, id<MTLBuffer> out) {
        prof(e, std::string("mv_topk_") + kernel_type_name(w.t->type) + " " + role(w));
        MvArgs a{};
        a.mat_stride = w.mat_stride();
        a.K = w.K();
        a.N = w.N();
        a.row_bytes = w.row_bytes();
        a.y_stride = a.N;
        [e setComputePipelineState:pso(std::string("mv_topk_") + kernel_type_name(w.t->type))];
        [e setBytes:&a length:sizeof(a) atIndex:0];
        [e setBuffer:w.buf offset:w.off atIndex:1];
        [e setBuffer:xn offset:0 atIndex:2];
        [e setBuffer:out offset:0 atIndex:3];
        [e setBuffer:rlogits offset:0 atIndex:4];
        [e setBuffer:ids offset:ids_off atIndex:5];
        [e setBuffer:wts offset:ids_off atIndex:6];
        const uint32_t rows_per_tg = MV_R * MV_SG;
        if (!skip_cur) [e dispatchThreadgroups:MTLSizeMake(a.N / rows_per_tg, N_USED, 1)
            threadsPerThreadgroup:MTLSizeMake(32 * MV_SG, 1, 1)];
    }

    void enc_mv_down(id<MTLComputeCommandEncoder> __strong &e, const W &w, id<MTLBuffer> gin, id<MTLBuffer> uin,
                     id<MTLBuffer> out, uint32_t n_mat, bool use_ids) {
        prof(e, std::string("mv_down_") + kernel_type_name(w.t->type) + " " + role(w));
        MvDownArgs a{};
        a.mat_stride = w.mat_stride();
        a.K = w.K();
        a.N = w.N();
        a.row_bytes = w.row_bytes();
        a.n_mat = n_mat;
        a.use_ids = use_ids;
        a.add = 1;
        [e setComputePipelineState:pso(std::string("mv_down_") + kernel_type_name(w.t->type))];
        [e setBytes:&a length:sizeof(a) atIndex:0];
        [e setBuffer:w.buf offset:w.off atIndex:1];
        [e setBuffer:gin offset:0 atIndex:2];
        [e setBuffer:uin offset:0 atIndex:3];
        [e setBuffer:out offset:0 atIndex:4];
        [e setBuffer:ids offset:ids_off atIndex:5];
        [e setBuffer:wts offset:ids_off atIndex:6];
        const uint32_t rows_per_tg = MV_R * MV_SG;
        if (!skip_cur) [e dispatchThreadgroups:MTLSizeMake((a.N + rows_per_tg - 1) / rows_per_tg, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32 * MV_SG, 1, 1)];
    }

    void enc_get_row(id<MTLComputeCommandEncoder> __strong &e, const W &w, uint32_t row, id<MTLBuffer> out) {
        prof(e, "get_row");
        RowArgs a{w.K(), w.row_bytes(), row};
        [e setComputePipelineState:pso(std::string("get_row_") + kernel_type_name(w.t->type))];
        [e setBytes:&a length:sizeof(a) atIndex:0];
        [e setBuffer:w.buf offset:w.off atIndex:1];
        [e setBuffer:out offset:0 atIndex:2];
        [e setBuffer:tok offset:0 atIndex:3];
        if (!skip_cur) [e dispatchThreads:MTLSizeMake(a.K / 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    // 并发 encoder 内的依赖屏障（profile 模式每次调度单独 encoder，本身串行）
    void barrier(id<MTLComputeCommandEncoder> e) {
        if (!profiling) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    void encode_token(id<MTLComputeCommandEncoder> __strong &e, int pos, int n_run) {
        const uint32_t E = hp.n_embd, D = hp.head_dim;
        enc_get_row(e, tok_embd, 0xFFFFFFFFu, x);
        barrier(e);

        for (int il = 0; il < n_run; il++) {
            const Layer &L = layers[il];
            ids_off = uint64_t(il) * N_USED * 4;
            const bool swa = hp.is_swa[il];

            // ---- attention ----
            enc_rmsnorm(e, x, L.attn_norm, xn, E);
            barrier(e);
            enc_mv(e, L.q, xn, qkv);  // q/k/v 并发，写入连续的 qkv
            enc_mv(e, L.k, xn, qkv, false, 1, false, 0, Q_DIM * 4);
            enc_mv(e, L.v, xn, qkv, false, 1, false, 0, (Q_DIM + KV_DIM) * 4);
            barrier(e);

            if (on("attn")) {
                // 融合版：qk_norm/rope/写 KV 都在注意力 kernel 内（fused.metal: k_attn_dec）
                const uint32_t k_begin = swa ? uint32_t(std::max(pos - (N_SWA - 1), 0)) : 0;
                const uint32_t n_keys = pos - k_begin;
                AttnDecArgs da{};
                da.pos = pos;
                da.is_full = !swa;
                da.keys_per_split = std::max<uint32_t>(256, (n_keys + 31) / 32);
                da.n_split = std::max<uint32_t>(1, (n_keys + da.keys_per_split - 1) / da.keys_per_split);
                prof(e, "attn_dec");
                [e setComputePipelineState:pso("k_attn_dec")];
                [e setBytes:&da length:sizeof(da) atIndex:0];
                [e setBuffer:qkv offset:0 atIndex:1];
                [e setBuffer:L.q_norm.buf offset:L.q_norm.off atIndex:2];
                [e setBuffer:L.k_norm.buf offset:L.k_norm.off atIndex:3];
                [e setBuffer:L.sinks.buf offset:L.sinks.off atIndex:4];
                [e setBuffer:L.k_cache offset:0 atIndex:5];
                [e setBuffer:L.v_cache offset:0 atIndex:6];
                [e setBuffer:att offset:0 atIndex:7];
                [e setBuffer:partial offset:0 atIndex:8];
                if (!skip_cur) [e dispatchThreadgroups:MTLSizeMake(N_HEAD, da.n_split, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                barrier(e);
                if (da.n_split > 1) {
                    prof(e, "attn_combine");
                    [e setComputePipelineState:pso("k_attn_combine")];
                    [e setBytes:&da length:sizeof(da) atIndex:0];
                    [e setBuffer:partial offset:0 atIndex:1];
                    [e setBuffer:L.sinks.buf offset:L.sinks.off atIndex:2];
                    [e setBuffer:att offset:0 atIndex:3];
                    if (!skip_cur) [e dispatchThreadgroups:MTLSizeMake(N_HEAD, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    barrier(e);
                }
            } else {
                QkArgs qa{};
                qa.n_head = hp.n_head;
                qa.n_head_kv = hp.n_head_kv;
                qa.head_dim = D;
                qa.pos = pos;
                qa.slot = swa ? pos % L.n_slots : pos;
                qa.has_rope = hp.has_rope[il];
                qa.eps = hp.rms_eps;
                qa.rope_theta = hp.rope_theta;
                prof(e, "k_qk_norm_rope");
                [e setComputePipelineState:pso("k_qk_norm_rope")];
                [e setBytes:&qa length:sizeof(qa) atIndex:0];
                [e setBuffer:qkv offset:0 atIndex:1];
                [e setBuffer:qkv offset:Q_DIM * 4 atIndex:2];
                [e setBuffer:qkv offset:(Q_DIM + KV_DIM) * 4 atIndex:3];
                [e setBuffer:L.q_norm.buf offset:L.q_norm.off atIndex:4];
                [e setBuffer:L.k_norm.buf offset:L.k_norm.off atIndex:5];
                [e setBuffer:L.k_cache offset:0 atIndex:6];
                [e setBuffer:L.v_cache offset:0 atIndex:7];
                if (!skip_cur) [e dispatchThreadgroups:MTLSizeMake(hp.n_head + 2 * hp.n_head_kv, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(D, 1, 1)];
                barrier(e);

                AttnArgs aa{};
                aa.n_head = hp.n_head;
                aa.n_head_kv = hp.n_head_kv;
                aa.head_dim = D;
                aa.n_keys = swa ? std::min<uint32_t>(pos + 1, N_SWA) : pos + 1;
                aa.n_slots = L.n_slots;
                aa.first_slot = (pos + 1 - aa.n_keys) % L.n_slots;
                aa.scale = 1.0f / std::sqrt(float(D));
                prof(e, "k_attn");
                [e setComputePipelineState:pso("k_attn")];
                [e setBytes:&aa length:sizeof(aa) atIndex:0];
                [e setBuffer:qkv offset:0 atIndex:1];
                [e setBuffer:L.k_cache offset:0 atIndex:2];
                [e setBuffer:L.v_cache offset:0 atIndex:3];
                [e setBuffer:L.sinks.buf offset:L.sinks.off atIndex:4];
                [e setBuffer:att offset:0 atIndex:5];
                if (!skip_cur) [e dispatchThreadgroups:MTLSizeMake(hp.n_head, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * 8, 1, 1)];
                barrier(e);
            }

            enc_mv(e, L.o, att, x, /*add=*/true);
            barrier(e);

            // ---- FFN ----
            enc_rmsnorm(e, x, L.ffn_norm, xn, E);
            barrier(e);
            if (il < hp.n_dense_lead) {
                enc_mv(e, L.gate, xn, g);
                enc_mv(e, L.up, xn, u);
                barrier(e);
                enc_mv_down(e, L.down, g, u, x, 1, false);
                if (predict() && il + 1 == N_DENSE) {
                    enc_mv(e, layers[il + 1].router, xn, rlogits_pred);
                    enc_mv(e, layers[il + 2].router, xn, rlogits_pred2);
                    barrier(e);
                    enc_topk(e, rlogits_pred, pids, pwts, uint64_t(il + 1) * PK * 4, PK);
                    enc_topk(e, rlogits_pred2, pids2, pwts2, uint64_t(il + 2) * PK * 4, PK);
                }
                barrier(e);
            } else {
                // 路由与共享专家 gate/up 互不依赖，并发
                enc_mv(e, L.router, xn, rlogits);
                enc_mv(e, L.gate_sh, xn, gs);
                enc_mv(e, L.up_sh, xn, us);
                const bool pred_next = predict() && il + 1 < n_run;
                const bool pred_next2 = predict() && pred2_k > 0 && il + 2 < n_run;
                if (pred_next) enc_mv(e, layers[il + 1].router, xn, rlogits_pred);
                if (pred_next2) enc_mv(e, layers[il + 2].router, xn, rlogits_pred2);
                barrier(e);
                if (pred_next) enc_topk(e, rlogits_pred, pids, pwts, uint64_t(il + 1) * PK * 4, PK);
                if (pred_next2) enc_topk(e, rlogits_pred2, pids2, pwts2, uint64_t(il + 2) * PK * 4, PK);
                if (on("topk")) {
                    // topk 融合进专家 gate/up：省掉一个阶段；共享 down 与之并发（读 gs/us、写 x，互不冲突）
                    enc_mv_down(e, L.down_sh, gs, us, x, 1, false);
                    enc_mv_topk(e, L.gate_exps, g);
                    enc_mv_topk(e, L.up_exps, u);
                    barrier(e);
                    enc_mv_down(e, L.down_exps, g, u, x, hp.n_expert_used, true);
                    barrier(e);
                    continue;
                }
                TopkArgs ta{uint32_t(hp.n_expert), uint32_t(hp.n_expert_used)};
                prof(e, "k_topk");
                [e setComputePipelineState:pso("k_topk")];
                [e setBytes:&ta length:sizeof(ta) atIndex:0];
                [e setBuffer:rlogits offset:0 atIndex:1];
                [e setBuffer:ids offset:ids_off atIndex:2];
                [e setBuffer:wts offset:ids_off atIndex:3];
                if (!skip_cur) [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(hp.n_expert, 1, 1)];
                enc_mv_down(e, L.down_sh, gs, us, x, 1, false);  // 与 topk 并发（不读 ids）
                barrier(e);

                id<MTLBuffer> real_ids = ids;
                if (streaming) {
                    // topk 完成 → 通知 CPU 装载专家；等 CPU 写好槽号后再继续本层专家计算
                    // （试过 GPU 内自旋握手：执行中途 CPU 写的标志 GPU 迟迟看不到，结果错误且极慢，已放弃）
                    [e endEncoding];
                    [cur_cb encodeSignalEvent:ev value:ev_base + 2 * il + 1];
                    [cur_cb encodeWaitForEvent:ev value:ev_base + 2 * il + 2];
                    e = [cur_cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                    ids = sids;  // 专家 kernel 按槽号寻址（W = 槽池 + 部件偏移，步长 = 槽大小）
                }
                enc_mv(e, L.gate_exps, xn, g, false, hp.n_expert_used, true);
                enc_mv(e, L.up_exps, xn, u, false, hp.n_expert_used, true);
                barrier(e);
                enc_mv_down(e, L.down_exps, g, u, x, hp.n_expert_used, true);
                barrier(e);
                ids = real_ids;
            }
        }

        enc_rmsnorm(e, x, out_norm, xn, E);
        barrier(e);
        enc_mv(e, output, xn, logits);
        barrier(e);
        uint32_t nv = hp.n_vocab;
        prof(e, "k_argmax");
        [e setComputePipelineState:pso("k_argmax")];
        [e setBytes:&nv length:4 atIndex:0];
        [e setBuffer:logits offset:0 atIndex:1];
        [e setBuffer:out_tok offset:0 atIndex:2];
        if (!skip_cur) [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    }
};

Engine::Engine(const GgufFile &f, const EngineOptions &opt) : hp_(HParams::from_gguf(f)) {
    impl_ = std::make_unique<Impl>(f, hp_, opt);
}

Engine::~Engine() = default;

int32_t Engine::step(int32_t token, std::vector<float> *logits) {
    @autoreleasepool {
        if (pos_ >= impl_->opt.n_ctx) throw std::runtime_error("context full");
        *(int32_t *)impl_->tok.contents = token;
        const double t0 = CACurrentMediaTime();
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        impl_->cur_cb = cb;
        impl_->prof_labels.clear();
        static const bool serial = getenv("FLINT_SERIAL") != nullptr;  // A/B 对比用
        id<MTLComputeCommandEncoder> e = (impl_->profiling || serial) ? [cb computeCommandEncoder]
                                                          : [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        impl_->encode_token(e, pos_, debug_n_layers >= 0 ? debug_n_layers : hp_.n_layer);
        [e endEncoding];
        last_encode_ms = (CACurrentMediaTime() - t0) * 1e3;
        [cb commit];
        if (impl_->streaming) {
            // 逐层响应 GPU 的 topk 信号：装载未命中的专家后放行
            const int n_run = debug_n_layers >= 0 ? debug_n_layers : N_LAYER;
            for (int il = N_DENSE; il < n_run; il++) {
                const uint64_t want = impl_->ev_base + 2 * il + 1;
                while (impl_->ev.signaledValue < want) {
                    if (cb.status >= MTLCommandBufferStatusCompleted) break;
                }
                impl_->stream_layer(il);
                impl_->ev.signaledValue = want + 1;
            }
            impl_->ev_base += 2 * N_LAYER + 2;
        }
        [cb waitUntilCompleted];
        last_gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
        if (cb.error) throw std::runtime_error(std::string("GPU error: ") + cb.error.localizedDescription.UTF8String);
        pos_++;
        if (logits) {
            logits->resize(hp_.n_vocab);
            memcpy(logits->data(), impl_->logits.contents, hp_.n_vocab * 4);
        }
        return *(int32_t *)impl_->out_tok.contents;
    }
}

Engine::StreamStats Engine::stream_stats() const { return impl_->st; }

int Engine::n_ctx() const { return impl_->opt.n_ctx; }

const float *Engine::logits_ptr() const { return (const float *)impl_->logits.contents; }

void Engine::enable_profiling() {
    impl_->init_profiler();
    impl_->profiling = true;
}

std::vector<std::pair<std::string, double>> Engine::last_profile() {
    std::vector<std::pair<std::string, double>> out;
    const auto n = impl_->prof_labels.size();
    if (!n) return out;
    NSData *data = [impl_->samples resolveCounterRange:NSMakeRange(0, 2 * n)];
    const MTLCounterResultTimestamp *ts = (const MTLCounterResultTimestamp *)data.bytes;
    double total_ticks = 0;
    for (size_t i = 0; i < n; i++) total_ticks += double(ts[2 * i + 1].timestamp - ts[2 * i].timestamp);
    // 时间戳单位因设备而异：按本次 command buffer 的 GPU 时长等比换算成毫秒
    const double ms_per_tick = last_gpu_ms / total_ticks;
    for (size_t i = 0; i < n; i++)
        out.push_back({impl_->prof_labels[i], double(ts[2 * i + 1].timestamp - ts[2 * i].timestamp) * ms_per_tick});
    return out;
}

// ---------------------------------------------------------------------------
// 批量 prefill（常驻模式）：按层优先——整段 token 先全部过第 l 层再进第 l+1 层。
// 注意力按 ≤512 分块（因果依赖只在层内）；FFN 按 ≤2048 批整批算，MoE 由 CPU 按专家计数排序后分块。
int32_t Engine::prefill(const std::vector<int32_t> &tokens, std::vector<float> *logits) {
    @autoreleasepool {
        Impl &I = *impl_;
        const int N = int(tokens.size());
        if (N == 0) throw std::runtime_error("prefill: empty");
        if (pos_ + N > I.opt.n_ctx) throw std::runtime_error("prefill: context full");
        const int P0 = pos_, CH = MAX_CHUNK, FB = std::min(N, 1024), QS = Q_DIM + 2 * KV_DIM;  // FFN 批大小：常驻 Q3 时显存余量很小，中间缓冲区要控制在 ~0.3 GB
        // 中间缓冲区按名字复用（容量不够才重新分配），并放进常驻集合：
        // 每个 command buffer 引用新的大 buffer 时 Metal 要先把它换入，实测 2048 token 多花 ~1 s
        bool realloc = false;
        auto buf = [&](int k, size_t bytes) -> id<MTLBuffer> {
            bytes = std::max<size_t>(bytes, 16);
            id<MTLBuffer> __strong &b = I.pf_bufs[k];
            if (!b || b.length < bytes) {
                if (b) [I.pf_res removeAllocation:b];
                b = [I.dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                [I.pf_res addAllocation:b];
                realloc = true;
            }
            return b;
        };
        id<MTLBuffer> btok = buf(0, size_t(N) * 4);
        memcpy(btok.contents, tokens.data(), N * 4);
        id<MTLBuffer> X = buf(1, size_t(N) * N_EMBD * 4), Xn = buf(2, size_t(FB) * N_EMBD * 4), TMP = buf(3, size_t(FB) * N_EMBD * 4);
        id<MTLBuffer> QKV = buf(4, size_t(CH) * QS * 4), ATT = buf(5, size_t(CH) * Q_DIM * 4);
        id<MTLBuffer> RL = buf(6, size_t(FB) * N_EXPERT * 4), IDS = buf(7, size_t(FB) * N_USED * 4), WTS = buf(8, size_t(FB) * N_USED * 4);
        id<MTLBuffer> POS = buf(9, size_t(FB) * N_USED * 4), SRC = buf(10, size_t(FB) * N_USED * 4);
        id<MTLBuffer> TILES = buf(11, (1 + 3 * (size_t(FB) * N_USED / 32 + N_EXPERT)) * 4);
        id<MTLBuffer> XG = buf(12, size_t(FB) * N_USED * N_EMBD * 4), GG = buf(13, size_t(FB) * N_USED * N_FF_EXP * 4);
        id<MTLBuffer> UG = buf(14, size_t(FB) * N_USED * N_FF_EXP * 4), DG = buf(15, size_t(FB) * N_USED * N_EMBD * 4);
        id<MTLBuffer> GS = buf(16, size_t(FB) * std::max(N_FF_SH, N_FF_DENSE) * 4), US = buf(17, size_t(FB) * std::max(N_FF_SH, N_FF_DENSE) * 4);
        id<MTLBuffer> SH = buf(18, size_t(FB) * N_EMBD * 4);
        if (realloc) [I.pf_res commit];
        [I.pf_res requestResidency];  // 只在 prefill 期间常驻：常驻 Q3 时显存余量很小，decode 时要让出来

        id<MTLCommandBuffer> cb;
        id<MTLComputeCommandEncoder> e;
        // FLINT_PF_STATS=1：按阶段统计 GPU 时间（attn = 注意力+路由，moe = 专家/共享/合并，dense = 前两层 FFN）
        static const bool stats = getenv("FLINT_PF_STATS") != nullptr;
        std::map<std::string, double> gpu_ms;
        std::string phase = "embed";
        double cpu_sort_ms = 0, load_ms = 0;
        int prefetched = 0;
        const double t_wall0 = CACurrentMediaTime();
        auto begin = [&]() {
            cb = [I.queue commandBuffer];
            [cb useResidencySet:I.pf_res];
            e = [cb computeCommandEncoder];
        };
        auto finish = [&]() {
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) throw std::runtime_error(std::string("GPU error: ") + cb.error.localizedDescription.UTF8String);
            gpu_ms[phase] += (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
        };
        // Y[t][..] = W · X[t]：W 为 2D 权重；x_off / y_off 为字节偏移
        auto mm = [&](const Impl::W &w, id<MTLBuffer> x, uint64_t x_off, uint32_t x_stride, id<MTLBuffer> y, uint64_t y_off,
                      uint32_t y_stride, int n) {
            MmArgs a{uint32_t(w.t->dims[1]), uint32_t(w.t->dims[0]), uint32_t(n), uint32_t(w.t->row_bytes()), x_stride, y_stride};
            [e setComputePipelineState:I.pso(std::string("mm_") + kernel_type_name(w.t->type))];
            [e setBytes:&a length:sizeof(a) atIndex:0];
            [e setBuffer:w.buf offset:w.off atIndex:1];
            [e setBuffer:x offset:x_off atIndex:2];
            [e setBuffer:y offset:y_off atIndex:3];
            [e setThreadgroupMemoryLength:64 * 32 * 2 atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake((n + 127) / 128, a.M / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        };
        auto mm_id = [&](const Impl::W &w, id<MTLBuffer> x, uint32_t x_stride, id<MTLBuffer> y, uint32_t y_stride, int n_tiles) {
            MmIdArgs a{{uint32_t(w.t->dims[1]), uint32_t(w.t->dims[0]), 0, uint32_t(w.t->row_bytes()), x_stride, y_stride},
                       w.mat_stride()};
            [e setComputePipelineState:I.pso(std::string("mm_id_") + kernel_type_name(w.t->type))];
            [e setBytes:&a length:sizeof(a) atIndex:0];
            [e setBuffer:w.buf offset:w.off atIndex:1];
            [e setBuffer:x offset:0 atIndex:2];
            [e setBuffer:y offset:0 atIndex:3];
            [e setBuffer:TILES offset:0 atIndex:4];
            [e setThreadgroupMemoryLength:64 * 32 * 2 atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake(n_tiles, a.mm.M / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        };
        auto rows = [&](const char *k, int n_rows, int threads, std::initializer_list<std::pair<id<MTLBuffer>, uint64_t>> bufs) {
            [e setComputePipelineState:I.pso(k)];
            int i = 0;
            for (auto &[b, off] : bufs) [e setBuffer:b offset:off atIndex:i++];
            [e dispatchThreadgroups:MTLSizeMake(n_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        };
        auto elem = [&](const char *k, int n_rows, int width, std::initializer_list<id<MTLBuffer>> bufs) {
            RowsArgs ra{uint32_t(n_rows), uint32_t(width)};
            [e setComputePipelineState:I.pso(k)];
            [e setBytes:&ra length:sizeof(ra) atIndex:0];
            int i = 1;
            for (id<MTLBuffer> b : bufs) [e setBuffer:b offset:0 atIndex:i++];
            [e dispatchThreads:MTLSizeMake(size_t(n_rows) * width / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        };
        auto norm_w = [&](const Impl::W &w) { return std::make_pair(w.buf, w.off); };

        begin();
        rows("k_embed_rows", N, 128, {norm_w(I.tok_embd), {btok, 0}, {X, 0}});
        finish();

        I.pf_mru = I.streaming && N >= 64;  // 短 prefill 不是整层扫描，保持 LRU 以免冲掉 decode 的热专家
        for (int il = 0; il < N_LAYER; il++) {
            const Impl::Layer &L = I.layers[il];
            const bool full = layer_is_full(il);
            I.pf_protect[0] = il;
            I.pf_protect[1] = il + 1;
            // ---- 注意力：按块 ----
            phase = "attn+router";
            begin();
            for (int c0 = 0; c0 < N; c0 += CH) {
                const int T = std::min(CH, N - c0);
                const uint64_t xo = uint64_t(c0) * N_EMBD * 4;
                rows("k_rmsnorm_rows", T, 256, {{X, xo}, norm_w(L.attn_norm), {Xn, 0}});
                mm(L.q, Xn, 0, N_EMBD, QKV, 0, QS, T);
                mm(L.k, Xn, 0, N_EMBD, QKV, Q_DIM * 4, QS, T);
                mm(L.v, Xn, 0, N_EMBD, QKV, (Q_DIM + KV_DIM) * 4, QS, T);
                QkPfArgs qa{uint32_t(P0 + c0), uint32_t(T), full};
                [e setComputePipelineState:I.pso("k_qk_prefill")];
                [e setBytes:&qa length:sizeof(qa) atIndex:0];
                [e setBuffer:QKV offset:0 atIndex:1];
                [e setBuffer:L.q_norm.buf offset:L.q_norm.off atIndex:2];
                [e setBuffer:L.k_norm.buf offset:L.k_norm.off atIndex:3];
                [e setBuffer:L.k_cache offset:0 atIndex:4];
                [e setBuffer:L.v_cache offset:0 atIndex:5];
                [e dispatchThreadgroups:MTLSizeMake(T, N_HEAD + 2 * N_HEAD_KV, 1) threadsPerThreadgroup:MTLSizeMake(HEAD_DIM, 1, 1)];
                AttnPfArgs aa{uint32_t(P0 + c0), uint32_t(T), full, uint32_t(QS)};
                [e setComputePipelineState:I.pso("k_attn_prefill")];
                [e setBytes:&aa length:sizeof(aa) atIndex:0];
                [e setBuffer:QKV offset:0 atIndex:1];
                [e setBuffer:L.k_cache offset:0 atIndex:2];
                [e setBuffer:L.v_cache offset:0 atIndex:3];
                [e setBuffer:L.sinks.buf offset:L.sinks.off atIndex:4];
                [e setBuffer:ATT offset:0 atIndex:5];
                [e dispatchThreadgroups:MTLSizeMake((T + 31) / 32, N_HEAD, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                mm(L.o, ATT, 0, Q_DIM, TMP, 0, N_EMBD, T);
                RowsArgs ra{uint32_t(T), N_EMBD};
                [e setComputePipelineState:I.pso("k_add_rows")];
                [e setBytes:&ra length:sizeof(ra) atIndex:0];
                [e setBuffer:X offset:xo atIndex:1];
                [e setBuffer:TMP offset:0 atIndex:2];
                [e dispatchThreads:MTLSizeMake(size_t(T) * N_EMBD / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            // ---- FFN：按批 ----
            for (int b0 = 0; b0 < N; b0 += FB) {
                const int B = std::min(FB, N - b0);
                const uint64_t xo = uint64_t(b0) * N_EMBD * 4;
                rows("k_rmsnorm_rows", B, 256, {{X, xo}, norm_w(L.ffn_norm), {Xn, 0}});
                if (il < N_DENSE) {
                    phase = "dense";
                    if (I.streaming && il == N_DENSE - 1 && b0 == 0 && N >= 64) prefetched += I.prefetch_layer(N_DENSE, -1);
                    mm(L.gate, Xn, 0, N_EMBD, GS, 0, N_FF_DENSE, B);
                    mm(L.up, Xn, 0, N_EMBD, US, 0, N_FF_DENSE, B);
                    elem("k_swiglu", B, N_FF_DENSE, {GS, US, GS});
                    mm(L.down, GS, 0, N_FF_DENSE, TMP, 0, N_EMBD, B);
                    RowsArgs ra{uint32_t(B), N_EMBD};
                    [e setComputePipelineState:I.pso("k_add_rows")];
                    [e setBytes:&ra length:sizeof(ra) atIndex:0];
                    [e setBuffer:X offset:xo atIndex:1];
                    [e setBuffer:TMP offset:0 atIndex:2];
                    [e dispatchThreads:MTLSizeMake(size_t(B) * N_EMBD / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    continue;
                }
                mm(L.router, Xn, 0, N_EMBD, RL, 0, N_EXPERT, B);
                rows("k_topk_rows", B, N_EXPERT, {{RL, 0}, {IDS, 0}, {WTS, 0}});
                finish();

                // CPU：按专家计数排序（稳定），生成重排表与分块表
                const double t_sort0 = CACurrentMediaTime();
                const int *ids = (const int *)IDS.contents;
                int *src = (int *)SRC.contents, *posv = (int *)POS.contents, *tiles = (int *)TILES.contents;
                int cnt[N_EXPERT] = {0}, off[N_EXPERT], fill[N_EXPERT] = {0};
                for (int i = 0; i < B * N_USED; i++) cnt[ids[i]]++;
                for (int x2 = 0, acc = 0; x2 < N_EXPERT; x2++) { off[x2] = acc; acc += cnt[x2]; }
                for (int i = 0; i < B * N_USED; i++) {
                    const int ex = ids[i], d = off[ex] + fill[ex]++;
                    src[d] = i / N_USED;
                    posv[i] = d;
                }
                cpu_sort_ms += (CACurrentMediaTime() - t_sort0) * 1e3;

                // 专家按编号连续区间分波：常驻模式一波全做（矩阵号 = 专家号）；
                // 流式模式每波的不同专家数不超过该尺寸级的槽数，装入后矩阵号 = 级内槽号。
                // 同一区间的专家在排序后占连续的行 [off[lo], off[hi])，SwiGLU 只对这些行做
                phase = "moe";
                begin();
                rows("k_gather_rows", B * N_USED, 128, {{Xn, 0}, {SRC, 0}, {XG, 0}});
                const int cap = I.streaming ? I.cls[I.layer_cls[il]].count : N_EXPERT;
                for (int lo = 0; lo < N_EXPERT;) {
                    int hi = lo, distinct = 0;
                    while (hi < N_EXPERT && distinct + (cnt[hi] > 0) <= cap) distinct += cnt[hi++] > 0;
                    if (I.streaming) {
                        const double t_ld = CACurrentMediaTime();
                        std::vector<int> need;
                        for (int ex = lo; ex < hi; ex++) {
                            if (!cnt[ex]) continue;
                            const int key = il * N_EXPERT + ex;
                            if (I.slot_of[key] >= 0) {
                                I.lru_touch(I.slot_of[key]);
                                I.st.hits++;
                            } else {
                                I.load_expert(il, ex);
                                I.st.misses++;
                            }
                            need.push_back(I.slot_of[key]);
                        }
                        for (int sl : need) I.wait_slot(sl);
                        load_ms += (CACurrentMediaTime() - t_ld) * 1e3;
                        if (getenv("FLINT_PF_STATS") && atoi(getenv("FLINT_PF_STATS")) >= 2)
                            fprintf(stderr, "  layer %2d class %d (cap %d, %d layers) wave [%3d,%3d) need %3zu wait %6.1f ms\n", il,
                                    I.layer_cls[il], cap, I.cls[I.layer_cls[il]].n_layers, lo, hi, need.size(),
                                    (CACurrentMediaTime() - t_ld) * 1e3);
                    }
                    int nt = 0;
                    const int first = I.streaming ? I.cls[I.layer_cls[il]].first : 0;
                    for (int ex = lo; ex < hi; ex++)
                        for (int s0 = 0; s0 < cnt[ex]; s0 += 32) {  // 与 prefill.metal 的 MM_TOK_ID 一致
                            tiles[1 + 3 * nt] = I.streaming ? I.slot_of[il * N_EXPERT + ex] - first : ex;
                            tiles[2 + 3 * nt] = off[ex] + s0;
                            tiles[3 + 3 * nt] = std::min(32, cnt[ex] - s0);
                            nt++;
                        }
                    tiles[0] = nt;
                    const int r0 = lo < N_EXPERT ? off[lo] : B * N_USED;
                    const int r1 = hi < N_EXPERT ? off[hi] : B * N_USED;
                    if (nt > 0) {
                        mm_id(L.gate_exps, XG, N_EMBD, GG, N_FF_EXP, nt);
                        mm_id(L.up_exps, XG, N_EMBD, UG, N_FF_EXP, nt);
                        RowsArgs ra{uint32_t(r1 - r0), N_FF_EXP};
                        const uint64_t ro = uint64_t(r0) * N_FF_EXP * 4;
                        [e setComputePipelineState:I.pso("k_swiglu")];
                        [e setBytes:&ra length:sizeof(ra) atIndex:0];
                        [e setBuffer:GG offset:ro atIndex:1];
                        [e setBuffer:UG offset:ro atIndex:2];
                        [e setBuffer:GG offset:ro atIndex:3];
                        [e dispatchThreads:MTLSizeMake(size_t(r1 - r0) * N_FF_EXP / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        mm_id(L.down_exps, GG, N_FF_EXP, DG, N_EMBD, nt);
                    }
                    lo = hi;
                    // 下一波要复用分块表、并可能淘汰本波的槽：先等 GPU 做完
                    if (lo < N_EXPERT) { finish(); begin(); }
                }
                // 本层专家都已就位：后台预取下一层，与本层 FFN 及下一层注意力的 GPU 计算重叠
                // 首批后即发起整层预取（后续批次 + 下一层注意力都可重叠）。只在 token 足够多时做：
                // 64 个 token 时每层约 78% 的专家会被用到；短 prefill（如多轮对话的增量）只装真正用到的
                if (I.streaming && b0 == 0 && N >= 64) prefetched += I.prefetch_layer(il + 1, il);
                mm(L.gate_sh, Xn, 0, N_EMBD, GS, 0, N_FF_SH, B);
                mm(L.up_sh, Xn, 0, N_EMBD, US, 0, N_FF_SH, B);
                elem("k_swiglu", B, N_FF_SH, {GS, US, GS});
                mm(L.down_sh, GS, 0, N_FF_SH, SH, 0, N_EMBD, B);
                rows("k_moe_combine", B, 128, {{X, xo}, {SH, 0}, {DG, 0}, {POS, 0}, {WTS, 0}});
            }
            finish();
        }

        if (stats) {
            double tot = 0;
            for (auto &[k2, v] : gpu_ms) tot += v;
            fprintf(stderr, "prefill %d tokens: wall %.1f ms, GPU %.1f ms (", N, (CACurrentMediaTime() - t_wall0) * 1e3, tot);
            for (auto &[k2, v] : gpu_ms) fprintf(stderr, " %s %.1f", k2.c_str(), v);
            fprintf(stderr, "), CPU sort %.1f ms", cpu_sort_ms);
            if (I.streaming) fprintf(stderr, ", expert load wait %.1f ms, prefetched %d", load_ms, prefetched);
            fprintf(stderr, "\n");
        }
        // 最后一个 token：复用 decode 路径的 output_norm + lm_head + argmax
        memcpy(I.x.contents, (const float *)X.contents + size_t(N - 1) * N_EMBD, N_EMBD * 4);
        begin();
        I.enc_rmsnorm(e, I.x, I.out_norm, I.xn, N_EMBD);
        I.enc_mv(e, I.output, I.xn, I.logits);
        uint32_t nv = N_VOCAB;
        [e setComputePipelineState:I.pso("k_argmax")];
        [e setBytes:&nv length:4 atIndex:0];
        [e setBuffer:I.logits offset:0 atIndex:1];
        [e setBuffer:I.out_tok offset:0 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        finish();
        [I.pf_res endResidency];
        I.pf_mru = false;
        pos_ += N;
        if (logits) {
            logits->resize(N_VOCAB);
            memcpy(logits->data(), I.logits.contents, N_VOCAB * 4);
        }
        return *(int32_t *)I.out_tok.contents;
    }
}

// 测试批量矩阵乘：Y[t][m] = W·X[t]。experts 非空时走 mm_id：segs[i] 个 token 用 experts[i] 号专家
std::vector<float> Engine::debug_mm(const Tensor &t, const std::vector<float> &X, int T, const std::vector<int> &experts,
                                    const std::vector<int> &segs) {
    @autoreleasepool {
        Impl &I = *impl_;
        Impl::W w = I.weight(t.name);
        const uint32_t K = t.dims[0], M = t.dims[1];
        id<MTLBuffer> bx = [I.dev newBufferWithBytes:X.data() length:X.size() * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [I.dev newBufferWithLength:size_t(T) * M * 4 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [I.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        MmArgs a{M, K, uint32_t(T), uint32_t(t.row_bytes()), K, M};
        if (experts.empty()) {
            [e setComputePipelineState:I.pso(std::string("mm_") + kernel_type_name(t.type))];
            [e setBytes:&a length:sizeof(a) atIndex:0];
        } else {
            std::vector<int> tiles{0};
            int seg = 0;
            for (size_t i = 0; i < experts.size(); i++) {
                for (int s0 = 0; s0 < segs[i]; s0 += 32) {
                    tiles.insert(tiles.end(), {experts[i], seg + s0, std::min(32, segs[i] - s0)});
                    tiles[0]++;
                }
                seg += segs[i];
            }
            MmIdArgs ia{a, uint64_t(M) * t.row_bytes()};
            [e setComputePipelineState:I.pso(std::string("mm_id_") + kernel_type_name(t.type))];
            [e setBytes:&ia length:sizeof(ia) atIndex:0];
            [e setBytes:tiles.data() length:tiles.size() * 4 atIndex:4];
            a.N = uint32_t(tiles[0]);  // 仅用于下面的 grid.x（分块数）
        }
        [e setBuffer:w.buf offset:w.off atIndex:1];
        [e setBuffer:bx offset:0 atIndex:2];
        [e setBuffer:by offset:0 atIndex:3];
        [e setThreadgroupMemoryLength:64 * 32 * 2 atIndex:0];
        static const int reps = getenv("FLINT_MM_REPS") ? atoi(getenv("FLINT_MM_REPS")) : 1;  // 测速：重复 dispatch
        for (int r = 0; r < reps; r++)
            [e dispatchThreadgroups:MTLSizeMake(experts.empty() ? (T + 127) / 128 : a.N, M / 64, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) throw std::runtime_error(std::string("GPU error: ") + cb.error.localizedDescription.UTF8String);
        last_gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3 / reps;
        std::vector<float> r(size_t(T) * M);
        memcpy(r.data(), by.contents, r.size() * 4);
        return r;
    }
}

// 测试 prefill 注意力：qkv 为 [n_tok][5120]（未 norm/rope），按 chunk 分块依次 qk_prefill + attn_prefill，返回 [n_tok][4096]
std::vector<float> Engine::debug_attn(int il, const std::vector<float> &qkv_in, int n_tok, int chunk) {
    @autoreleasepool {
        Impl &I = *impl_;
        const Impl::Layer &L = I.layers[il];
        const bool full = layer_is_full(il);
        const int QS = Q_DIM + 2 * KV_DIM;
        id<MTLBuffer> bq = [I.dev newBufferWithBytes:qkv_in.data() length:qkv_in.size() * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bo = [I.dev newBufferWithLength:size_t(n_tok) * Q_DIM * 4 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [I.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        for (int p0 = 0; p0 < n_tok; p0 += chunk) {
            const int T = std::min(chunk, n_tok - p0);
            QkPfArgs qa{uint32_t(p0), uint32_t(T), full};
            [e setComputePipelineState:I.pso("k_qk_prefill")];
            [e setBytes:&qa length:sizeof(qa) atIndex:0];
            [e setBuffer:bq offset:uint64_t(p0) * QS * 4 atIndex:1];
            [e setBuffer:L.q_norm.buf offset:L.q_norm.off atIndex:2];
            [e setBuffer:L.k_norm.buf offset:L.k_norm.off atIndex:3];
            [e setBuffer:L.k_cache offset:0 atIndex:4];
            [e setBuffer:L.v_cache offset:0 atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake(T, N_HEAD + 2 * N_HEAD_KV, 1) threadsPerThreadgroup:MTLSizeMake(HEAD_DIM, 1, 1)];
            AttnPfArgs aa{uint32_t(p0), uint32_t(T), full, uint32_t(QS)};
            [e setComputePipelineState:I.pso("k_attn_prefill")];
            [e setBytes:&aa length:sizeof(aa) atIndex:0];
            [e setBuffer:bq offset:uint64_t(p0) * QS * 4 atIndex:1];
            [e setBuffer:L.k_cache offset:0 atIndex:2];
            [e setBuffer:L.v_cache offset:0 atIndex:3];
            [e setBuffer:L.sinks.buf offset:L.sinks.off atIndex:4];
            [e setBuffer:bo offset:uint64_t(p0) * Q_DIM * 4 atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake((T + 31) / 32, N_HEAD, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) throw std::runtime_error(std::string("GPU error: ") + cb.error.localizedDescription.UTF8String);
        last_gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
        std::vector<float> r(size_t(n_tok) * Q_DIM);
        memcpy(r.data(), bo.contents, r.size() * 4);
        return r;
    }
}

std::vector<int> Engine::debug_experts() const {
    const int *p = (const int *)impl_->ids.contents + impl_->ids_off / 4;
    return std::vector<int>(p, p + hp_.n_expert_used);
}

std::vector<int> Engine::routing_pred(int ahead) const {
    const int *p = (const int *)(ahead == 2 ? impl_->pids2 : impl_->pids).contents;
    return std::vector<int>(p, p + N_LAYER * Impl::PK);
}

std::vector<int> Engine::routing() const {
    const int *p = (const int *)impl_->ids.contents;
    return std::vector<int>(p, p + N_LAYER * N_USED);
}

std::vector<float> Engine::debug_hidden() const {
    const float *p = (const float *)impl_->x.contents;
    return std::vector<float>(p, p + hp_.n_embd);
}

std::vector<float> Engine::debug_get_row(const Tensor &t, uint32_t row) {
    @autoreleasepool {
        Impl::W w = impl_->weight(t.name);
        id<MTLBuffer> out = [impl_->dev newBufferWithLength:t.dims[0] * 4 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        impl_->enc_get_row(e, w, row, out);
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        std::vector<float> r(t.dims[0]);
        memcpy(r.data(), out.contents, r.size() * 4);
        return r;
    }
}

}  // namespace flint
