#!/usr/bin/env python3
"""估算 Flint 每个 decode token 需要读取的权重字节数（按张量种类分组）。"""
import re, subprocess, sys
from collections import defaultdict
out = subprocess.run([sys.executable, "-I", "tools/gguf_inspect.py", sys.argv[1], "--tensors"], capture_output=True, text=True).stdout
TOP_K, N_EXP = 6, 256
groups = defaultdict(float)
for line in out.splitlines():
    m = re.match(r"\s+(\S+)\s+\[.*\]\s+(\S+)\s+([\d.]+) MiB", line)
    if not m: continue
    name, _, mib = m.group(1), m.group(2), float(m.group(3))
    if name == "token_embd.weight": continue  # 只取一行
    kind = re.sub(r"^blk\.\d+\.", "", name)
    if "_exps" in kind: mib *= TOP_K / N_EXP
    kind = ("experts(top6)" if "_exps" in kind else "attn" if kind.startswith("attn_") else
            "shared_expert" if "shexp" in kind else "dense_ffn" if kind.startswith("ffn_") and "inp" not in kind and "norm" not in kind else
            "lm_head" if kind == "output.weight" else "router/norm/other")
    groups[kind] += mib
total = sum(groups.values())
for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
    print(f"  {k:20s} {v:8.1f} MiB  {v/total*100:5.1f}%")
print(f"  {'total':20s} {total:8.1f} MiB  -> 153 GB/s 带宽上限 {153e9/(total*2**20):.0f} tok/s")
