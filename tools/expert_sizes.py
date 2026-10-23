#!/usr/bin/env python3
"""从 GGUF（可只是文件头）统计：专家权重总量、单个专家大小、非专家部分大小。"""
import re, subprocess, sys
from collections import defaultdict
for path in sys.argv[1:]:
    out = subprocess.run([sys.executable, "-I", "tools/gguf_inspect.py", path, "--tensors"], capture_output=True, text=True).stdout
    exp, other, per_layer = 0.0, 0.0, defaultdict(float)
    for line in out.splitlines():
        m = re.match(r"\s+(\S+)\s+\[.*\]\s+(\S+)\s+([\d.]+) MiB", line)
        if not m: continue
        name, mib = m.group(1), float(m.group(3))
        if "_exps" in name:
            exp += mib; per_layer[int(name.split(".")[1])] += mib / 256
        else:
            other += mib
    sizes = sorted(per_layer.values())
    print(f"{path}: experts {exp/1024:.2f} GiB, non-expert {other/1024:.2f} GiB, "
          f"per-expert (gate+up+down) {sizes[0]:.2f}–{sizes[-1]:.2f} MiB, mean {sum(sizes)/len(sizes):.2f} MiB, "
          f"per-token expert read {sum(sizes)/len(sizes)*6*47:.0f} MiB")
