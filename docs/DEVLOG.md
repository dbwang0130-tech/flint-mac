# flint-mac 技术说明

本文按模块说明 flint-mac 做了什么、为什么这样设计、实测效果如何。使用方法和总体对比见 [README](../README.md)。

测试平台：MacBook Pro M5，24 GB 统一内存，内存带宽标称 153 GB/s、实测纯读约 123 GB/s，SSD 冷读约 5.7 GB/s。

---

## 1. 只为 Flint 服务

Flint 的结构：

| 项目 | 值 |
|---|---|
| 隐层 / 层数 / 词表 | 2048 / 49 / 150144，lm_head 与 embedding 不共享 |
| 注意力 | GQA 32Q/4KV×128，QK RMSNorm，learned sink（每个头一个加进 softmax 分母的 logit） |
| 层类型 | 14 层全注意力（层 0、1、5、9…45、48，不用 RoPE），35 层滑动窗口 513 + NEOX RoPE |
| FFN | 层 0、1 为 dense SwiGLU 6144；其余 47 层 MoE |
| MoE | 256 个专家 × 512，sigmoid 打分 → top-6 → 选中权重归一化；另有 1024 维共享专家（无门控） |

这些都写成编译期常量（`src/metal/flint_config.h`），kernel 按这些形状特化。加载时校验 GGUF 的每一项，不一致就直接报错，不做任何通用兼容。

**每个 token 的读取量决定了 decode 的上限**（UD-Q3_K_XL，`tools/decode_bytes.py`）：

| 部分 | 每 token 读取量 |
|---|---|
| 注意力（大多为 Q8_0） | 849 MiB |
| 路由专家（top-6） | 370 MiB |
| lm_head | 312 MiB |
| 共享专家 | 266 MiB |
| 其他 | 124 MiB |
| **合计** | **1.92 GiB → 理论上限约 61 tok/s** |

---

## 2. Decode

**执行结构**：每个 token 只提交一个 command buffer，路由在 GPU 上完成，不回 CPU。使用并发 encoder，只在有数据依赖的地方插 barrier，互不依赖的调度同时执行（q/k/v 一起；路由和共享专家的 gate/up 一起）。CPU 编码一个 token 只需 0.27 ms。

**kernel**：
- 矩阵乘：每个 simdgroup 算 2 行，各 lane 按 16 元素块交错读取。中等大小的 Q8_0 矩阵能跑到 115 GB/s，约为纯读带宽的 93%。
- 融合注意力（`k_attn_dec`）：QK RMSNorm、NEOX RoPE、learned sink、写 KV cache 全部在一个 kernel 里完成；当前 token 的 k/v 直接用寄存器里的值。上下文长时自动把 key 切成多段并行（split-K），最后合并。
- KV cache：全注意力层按上下文长度分配；SWA 层只用 1024 个槽的环形缓冲（窗口 513 加一个 prefill 块，取 2 的幂，取模变成位运算）。

**结果**（Q3_K_XL 常驻，交替测中位数）：flint-mac **50.6 tok/s**，llama.cpp 48.0 tok/s，已达到实测可达带宽的约 83%。

---

## 3. 量化解码

- 16 元素块的解码函数复用 ggml 的 `dequantize.h`（MIT）。
- **IQ3_XXS、IQ3_S、IQ2_S、IQ4_XS 各写了专用的 32 元素点积**：码本先装进 threadgroup memory，每 32 个元素只乘一次 scale。原因是 constant 内存在下标分散时会被串行化，直接调用通用解码函数时，专家 gate/up 只能跑到 53–79 GB/s。
- **修正 Q4_K 的精度问题**：ggml 的 `dequantize_q4_K` 用 half 计算 `d/16`，当 d 很小时会落入次正规数而丢失精度，导致后续层误差放大（KL 从 2e-8 涨到 1e-3）。我们改成了 float 版本。

---

## 4. 专家流式加载

目标是让 24 GB 的机器跑 25–42 GB 的模型。

### 设计依据：真实路由轨迹 + 缓存模拟

我们先用 `flint trace` 记录了约 3600 个 token 的路由，再用 `tools/cache_sim.py` 模拟不同的缓存策略：

| 缓存容量 | LRU | 常驻热门专家（静态） | 理论最优（Belady） |
|---|---|---|---|
| 37% | 72–73% | 37% | 88–89% |
| 50% | 82–84% | 50–51% | 93–94% |
| 65% | 90–92% | 65–67% | 96–97% |

结论：专家的使用在整体上不集中，常驻热门专家等于随机挑选；但时间局部性很强，适合用 LRU。

### 实现

- **非专家权重**（约 2 GB）拷贝进一块常驻 buffer。不把整个 GGUF mmap 成 Metal buffer，因为 Metal 使用 buffer 时会要求整块常驻。
- **槽池按专家尺寸分级**：不同层的专家尺寸不同，同尺寸的层共用一级，每级用独立的 buffer 和 LRU，太大时再按单个 MTLBuffer 的上限拆分。原来按最大尺寸统一分槽会浪费 25%，分级后同样 14 GB 能缓存的专家从 46% 提高到 61%。
- **`mlock` 锁住槽池**：residency set 并不锁页，内存紧张时系统会把槽池换出，GPU 访问时缺页，速度会降到 3 tok/s。
- **GPU kernel 不用改**：专家矩阵乘本来就按 `W + id × stride` 寻址，流式模式下把 W 换成槽池、stride 换成槽大小、id 换成槽号即可。
- **每层握手一次**：topk 完成后 GPU 用 `MTLSharedEvent` 发信号 → CPU 查 LRU，并发 `pread`（`F_NOCACHE`）读入未命中的专家，写好槽号 → 再发信号让 GPU 继续。每次握手约 88 µs。
- **预测并预取下一层的专家**：用下一层的 router 给当前层的 FFN 输入打分。6 个专家能猜中 72–76%，取 top-8/10 时召回率 80–88%，默认预取 8 个。命中率因此从 91% 提高到 97%。
- **自动槽池大小**：取"GPU 推荐工作集 − 常驻权重 − KV − 余量"和"系统可用内存 − 1.5 GB"两者中的较小值。

### 关键发现：GPU 一空等就会降频

我们在每层人为插入等待来测量：等待 200 µs，每个 token 多花 8 ms；等待 400 µs，多花 17 ms。也就是说，每空等 1 ms 要额外付出约 0.9 ms。所以优化方向是**少让 GPU 停下来**，而不只是减少 SSD 的读取量。

### 结果（真实生成，300 个 token）

| | llama.cpp 最好成绩 | flint-mac |
|---|---|---|
| Q4_K_XL（25 GB） | 4.85 tok/s（全 GPU 加载 OOM） | **17.9 tok/s** |
| Q8_0（42 GB） | 1.07 tok/s | **9.4 tok/s** |

---

## 5. 批量 prefill

### 按层优先（layer-major）

整个 prompt 的所有 token 先全部算完第 l 层，再进入第 l+1 层：
- **注意力按 ≤512 个 token 一块**依次计算，因果依赖只存在于层内；
- **FFN 整批计算**（每批 ≤1024 个 token），每个专家一次能分到更多 token；
- **流式模式下每个专家在整个 prompt 中只读一次**。

每层分两个 command buffer：第一个做注意力和路由；中间 CPU 按专家做计数排序、生成分块表；第二个做专家和共享专家的计算并合并结果。

### kernel（`src/metal/prefill.metal`）

- **批量矩阵乘**：按 ggml `mul_mm` 的 tensor-op 路径改写。权重按块解码成 half 放进 threadgroup memory，激活直接作为 device tensor 交给 `matmul2d`，实测 10–12 TFLOPS（M5 Neural Accelerator）。分块大小为 64 行 × 128 个 token；专家版用 64 × 32，否则 prompt 较短时（每个专家平均只分到约 12 个 token）大部分都在空算。改成 32 后 MoE 耗时 686 → 350 ms（512 个 token）。
- **flash attention**：每个 threadgroup 处理 1 个头 × 32 个 query，用 simdgroup 8×8 矩阵计算 S = QKᵀ 和 O += PV，因果遮罩和 SWA 窗口按元素判断；在线 softmax 的初值设为 m = sink、l = 1，sink 就自然加进了分母。
- **按行并行的小 kernel**：embedding、RMSNorm、残差、SwiGLU、逐 token 的 topk、按专家重排、MoE 合并。

### 流式模式的三处关键优化

| 优化 | 原因 | Q4 pp512 / pp2048 / pp4000（tok/s） |
|---|---|---|
| 同步装载（起点） | 每层排序后才开始读，GPU 一直空等 | 108 / 306 / 403 |
| 整层预取下一层 | prompt 一长几乎每个专家都会用到，不需要预测 | 119 / 366 / 531 |
| prefill 期间改用 MRU 淘汰 | 按层顺序扫描是循环访问，LRU 会全部重读（12032 个里重读 11077 个） | 208 / 505 / 602 |
| 小尺寸级给满容量 + 第一批 FFN 后就开始预取 | 单层的尺寸级装不下一整层，每批都要分波重读；预取越早，重叠越充分 | **233 / 668 / 834** |

prompt 少于 64 个 token 时不做整层预取，也不切换到 MRU：增量对话的短 prefill 只装载真正用到的专家，避免把 decode 的热专家挤出缓存。7 个 token 的 prefill 因此从 3.1 秒降到 0.48 秒。

### 内存管理

- 中间缓冲区按槽位复用，并且只挂在 prefill 自己的 command buffer 上（`useResidencySet:`）。如果每次都新分配，Metal 在每个 command buffer 之前都要重新换入（2048 个 token 时多花约 1 秒）；如果挂在整个 command queue 上，decode 也会带着它，导致 OOM。
- 常驻 Q3 时 GPU 内存余量很小（模型 18.5 GB，上限 19.07 GB），所以 FFN 每批最多 1024 个 token。

### 结果

| | pp512 | pp2048 | pp4000 |
|---|---|---|---|
| llama.cpp Q3（常驻） | 516 | 590 | OOM |
| **flint-mac Q3（常驻）** | **891** | **989** | **906** |
| llama.cpp Q4（`-ncmoe 18`） | 28 | 26 | — |
| **flint-mac Q4（流式）** | **233** | **668** | **834** |
| flint-mac Q8（流式） | 90 | 337 | 638 |

---

## 6. 服务（`server/`，Rust）

- **引擎 C API**（`src/capi.h`）：open / prefill / decode / logits / truncate，编成静态库 `build/libflint.a`，Rust 通过 `build.rs` 链接。
- **HTTP 直接用标准库实现**：只需要 `/v1/chat/completions` 和 `/v1/models` 两个接口；每个连接一个线程，引擎由一个 Mutex 串行化；SSE 以关闭连接结束。
- **chat template**：用 minijinja 渲染，按 transformers 的设置开启 trim_blocks 和 lstrip_blocks。自己注册了与 Python `json.dumps(ensure_ascii=False)` 行为相同的 `tojson`（分隔符 `", "` 和 `": "`、保持键顺序、不做 HTML 转义）；加载模板时把 `dict.items()` 替换成 minijinja 的 `|items`。渲染并分词后，与 llama.cpp 的结果逐 token 一致。
- **输出解析**：`<think>` 内容 → `reasoning_content`；`<tool_call>名<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>` → OpenAI 格式的 `tool_calls`，参数类型按工具 schema 判断（声明为 string 的保留原文，其余按 JSON 解析）。和官方解析器一样，不预设本轮一开始就在推理。
- **前缀 KV 复用**：新请求与上一次的 token 序列求公共前缀，只 prefill 后面的部分。模型停止时生成的 EOG token（`<|observation|>`、`<|user|>`）也写进 KV，正好和下一轮 prompt 对上。SWA 层的环形缓冲只保留最近 1024 个位置，回退超过 511 个 token 时，从回退点往前 512 个 token 开始重算。在 pi 的实际任务中，每轮复用 95% 以上的前缀。
- **采样**：temperature 1.0、top-k 20、top-p 0.95（官方推荐值）。

---

## 7. 正确性验证

| 层次 | 方法 | 结果 |
|---|---|---|
| 模型理解 | CPU fp32 逐 token 参考实现（`src/ref/`）vs llama.cpp | 30/30 top-1，KL 4e-5 |
| 量化解码 | `flint check-deq`：GPU vs CPU，每种类型抽样 | 逐位一致 |
| kernel | `flint test-mm`（9 种类型）、`flint test-attn`（1300 个 token，跨窗口和环形缓冲回绕） | 相对误差 1–3e-3（half 精度） |
| decode | `flint check` vs llama.cpp 对照数据 | 30/30，KL 2e-8 |
| prefill | `flint check-prefill`：批量 vs 逐 token；prefill 一半 + decode 一半 | KL 2e-5 / 2e-6 |
| 流式 | 流式 vs 常驻 | 完全一致 |

在一段重复度很高的 C++ 代码上，我们的困惑度比 llama.cpp 高 1.2%（3.12 对 3.08）。全 fp32 的参考实现与 GPU 结果一致；而 llama.cpp 自己换个设置，NLL 就会波动 11%（Metal + flash attention 1.696、关闭 flash attention 1.882、CPU 后端 1.787）。结论是：3-bit 模型在这类文本上对数值误差极其敏感，这不是 bug。

---

## 8. 性能测试方法

- 两个引擎**交替**各跑多轮取中位数（`tools/ab.sh`）。后台有高负载程序时噪声可达 ±10%，必须先清理干净再测。
- 只采信**并发模式**下的数据：逐阶段消融（`FLINT_SKIP` 加上 `tools/stage_cost.sh`）。按 dispatch 单独计时的串行 profile 会误导：曾经有一个版本在串行 profile 里更快，在实际并发执行中却慢了 8%。
- 微基准测试要避开 SLC 缓存：几 MB 的矩阵反复读会被缓存命中，测出超过内存带宽的数字；要轮流读多份互不相同的数据（`src/mbench_q8.mm`）。

---

## 9. 试过但没有采用的方案

| 方案 | 结果 |
|---|---|
| Q8_0 权重重排成 SoA 布局 | 无收益，原布局已达纯读带宽的 93% |
| 每个 simdgroup 内各自融合 RMSNorm | 更慢：每个 simdgroup 都要读一遍 x，而 IQ3 权重行很紧凑，读激活的开销比读权重还大 |
| threadgroup 级先算 norm 再计算 | 更慢：等 norm 的 barrier 让权重读取无法尽早开始 |
| 把 topk 融进专家 gate/up | 无收益：阶段之间的 barrier 不是主要开销 |
| GPU 内用共享内存标志自旋握手 | 不可行：GPU 执行中途看不到 CPU 新写的数据，结果错误且极慢 |
| 常驻热门专家 | 无效：命中率约等于缓存容量占比 |
| 提前两层预取 | 和提前一层 top-8 相比，差别在噪声范围内 |

---

## 10. 已知问题和下一步

- **pi 实际使用时生成约 7–11 tok/s**，低于单独测试时的 18：每轮 prefill 扫进来的专家会把 decode 的热专家挤出缓存。下一步是保护热专家，并让短 prefill 也使用预测预取。
- **注意力 kernel**：4000 个 token 时注意力占 prefill 一半的时间。全注意力层只跑到约 0.86 TFLOPS，可以让同一 GQA 组的 8 个头共享 K/V 读取。
- **流式 prefill 有固定开销**：每次都要把缓存外的专家扫一遍（Q4 约 1.7 秒）。
- **官方 NVFP4 GGUF 还不支持**（量化感知训练过的 4-bit，质量可能更好）。
