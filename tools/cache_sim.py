#!/usr/bin/env python3
"""用路由轨迹模拟专家缓存命中率（LRU / 静态热门集 / Belady 上界）。

轨迹文件：每行一个 token，47 层 × 6 个专家 id（层 2..48）。
缓存粒度为 (层, 专家)，全局共享一个容量（按"可缓存专家数占总数的比例"给出）。
静态热门集的频率来自 --profile 指定的另一份轨迹（避免拿自己测自己）。
用法: cache_sim.py <trace> <profile_trace> [warmup_tokens]
"""
import heapq
import sys
from collections import OrderedDict, defaultdict

N_MOE_LAYERS, N_EXPERT = 47, 256
TOTAL = N_MOE_LAYERS * N_EXPERT


def load(path):
    seq = []
    for line in open(path):
        ids = list(map(int, line.split()))
        seq.append([(l, ids[l * 6 + k]) for l in range(N_MOE_LAYERS) for k in range(6)])
    return seq


def lru(seq, cap, warm):
    c, hit, n = OrderedDict(), 0, 0
    for t, acc in enumerate(seq):
        for key in acc:
            if key in c:
                c.move_to_end(key)
                hit += t >= warm
            else:
                c[key] = 1
                if len(c) > cap:
                    c.popitem(last=False)
            n += t >= warm
    return hit / n


def static(seq, cap, warm, freq):
    # 按另一份轨迹统计的频率，常驻最常用的 cap 个专家（不做替换）
    hot = set(sorted(freq, key=freq.get, reverse=True)[:cap])
    hit = n = 0
    for t, acc in enumerate(seq):
        if t < warm:
            continue
        for key in acc:
            hit += key in hot
            n += 1
    return hit / n


def freq_of(seq):
    f = defaultdict(int)
    for acc in seq:
        for key in acc:
            f[key] += 1
    return f


def belady(seq, cap, warm):
    flat = [(t, key) for t, acc in enumerate(seq) for key in acc]
    nxt, last = [0] * len(flat), {}
    for i in range(len(flat) - 1, -1, -1):
        key = flat[i][1]
        nxt[i] = last.get(key, 1 << 60)
        last[key] = i
    cached, heap, hit, n = {}, [], 0, 0
    for i, (t, key) in enumerate(flat):
        if key in cached:
            hit += t >= warm
        elif len(cached) >= cap:
            while True:
                neg_next, k = heapq.heappop(heap)
                if cached.get(k) == -neg_next:
                    del cached[k]
                    break
        cached[key] = nxt[i]
        heapq.heappush(heap, (-nxt[i], key))
        n += t >= warm
    return hit / n


def main():
    seq = load(sys.argv[1])
    prof = freq_of(load(sys.argv[2]))
    warm = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    # 每个 (层,专家) 被使用的 token 比例：衡量分布是否集中
    use = defaultdict(int)
    for acc in seq:
        for key in set(acc):
            use[key] += 1
    counts = sorted(use.values(), reverse=True)
    tot = sum(counts)
    for frac in (0.1, 0.25, 0.5):
        k = int(TOTAL * frac)
        print(f"  最常用的 {frac:.0%} 专家承担了 {sum(counts[:k]) / tot:.1%} 的访问；"
              f"轨迹中出现过的专家 {len(use) / TOTAL:.1%}")
        break
    print(f"  tokens={len(seq)} warmup={warm}")
    print(f"  {'容量':>6} {'LRU':>7} {'静态':>7} {'Belady':>7}")
    for frac in (0.2, 0.3, 0.37, 0.5, 0.65, 0.8):
        cap = int(TOTAL * frac)
        print(f"  {frac:6.0%} {lru(seq, cap, warm):7.1%} {static(seq, cap, warm, prof):7.1%} {belady(seq, cap, warm):7.1%}")


if __name__ == "__main__":
    main()
