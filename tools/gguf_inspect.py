#!/usr/bin/env python3
"""列出 GGUF 文件的 metadata 和张量（名字、形状、量化类型、大小），按类型汇总。

用法: python3 -I tools/gguf_inspect.py <file.gguf> [--tensors]
"""
import struct
import sys
from collections import defaultdict

# ggml_type -> (名字, 每块元素数, 每块字节数)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1),
    25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17),
}


class Reader:
    def __init__(self, f):
        self.f = f

    def unpack(self, fmt):
        size = struct.calcsize(fmt)
        return struct.unpack("<" + fmt, self.f.read(size))

    def u32(self):
        return self.unpack("I")[0]

    def u64(self):
        return self.unpack("Q")[0]

    def string(self):
        return self.f.read(self.u64()).decode("utf-8", errors="replace")

    def value(self, vtype):
        scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if vtype in scalar:
            return self.unpack(scalar[vtype])[0]
        if vtype == 8:
            return self.string()
        if vtype == 9:
            etype, n = self.u32(), self.u64()
            return [self.value(etype) for _ in range(n)]
        raise ValueError(f"unknown gguf value type {vtype}")


def main():
    path = sys.argv[1]
    show_tensors = "--tensors" in sys.argv
    with open(path, "rb") as f:
        r = Reader(f)
        assert f.read(4) == b"GGUF", "not a GGUF file"
        version, n_tensors, n_kv = r.u32(), r.u64(), r.u64()
        print(f"GGUF v{version}: {n_tensors} tensors, {n_kv} metadata keys")
        for _ in range(n_kv):
            key = r.string()
            val = r.value(r.u32())
            if isinstance(val, list) and len(val) > 16:
                val = f"[{len(val)} items] {val[:4]}..."
            elif isinstance(val, str) and len(val) > 200:
                val = val[:200] + "..."
            print(f"  {key} = {val}")

        by_type = defaultdict(lambda: [0, 0])
        by_group = defaultdict(lambda: defaultdict(int))
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.u64() for _ in range(r.u32())]
            ttype = r.u32()
            r.u64()  # offset
            tname, block, bsize = GGML_TYPES.get(ttype, (f"type{ttype}", 1, 0))
            n = 1
            for d in dims:
                n *= d
            nbytes = n // block * bsize
            by_type[tname][0] += 1
            by_type[tname][1] += nbytes
            group = name.split(".", 2)[-1] if name.startswith("blk.") else name
            by_group[group][tname] += 1
            if show_tensors:
                print(f"  {name:48s} {str(dims):28s} {tname:8s} {nbytes / 2**20:9.2f} MiB")

    print("\n按量化类型汇总:")
    for tname, (count, nbytes) in sorted(by_type.items(), key=lambda kv: -kv[1][1]):
        print(f"  {tname:8s} {count:5d} tensors {nbytes / 2**30:8.2f} GiB")
    print("\n按张量种类汇总（各层合并）:")
    for group, types in sorted(by_group.items()):
        print(f"  {group:40s} " + ", ".join(f"{t}x{c}" for t, c in sorted(types.items())))


if __name__ == "__main__":
    main()
