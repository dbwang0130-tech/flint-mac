# flint-mac

**专为 [Flint](https://huggingface.co/RecursiveSI/flint) 写的 Apple Silicon 推理引擎**：Metal kernel 加一个 OpenAI 兼容服务，可以直接接 [pi](https://pi.dev) 这类编程 agent。

> A Flint-only inference engine for Apple Silicon. On a 24 GB MacBook it runs the 25 GB Q4_K_XL at **~18 tok/s** and the 42 GB Q8_0 at **~9 tok/s** by streaming MoE experts from SSD; llama.cpp either runs out of GPU memory or manages 1–5 tok/s on the same machine. Batched prefill (layer-major, M5 tensor ops) beats llama.cpp at every prompt length we tested, and the OpenAI-compatible Rust server drives pi end to end. All numbers below were measured on one MacBook Pro (M5, 24 GB).

Flint 是 Recursive 发布的开源编程模型：MoE 结构，总参数 40B，每个 token 只激活 2.5B（256 个专家里选 6 个）。本项目**只支持 Flint**：网络形状都是编译期常量，加载时校验 GGUF，形状对不上就直接报错。我们的原则是：**和通用引擎（llama.cpp）相比，时间或空间至少要明显赢一头**。

---

## 优势

### 1. 空间：24 GB 的 Mac 能跑 25 GB / 42 GB 的模型（主要卖点）

Flint 每个 token 只用到 2.3% 的专家权重。我们只把非专家权重（约 2 GB）常驻内存，专家权重放进一个按尺寸分级的槽池，按需从 SSD 读取，同时预测下一层会用到哪些专家并提前读取。

真实生成场景：Flint 先推理、再写代码，共 300 个 token：

| 模型（文件大小） | llama.cpp 最好成绩 | **flint-mac** | 倍数 |
|---|---|---|---|
| Q4_K_XL（25 GB） | 全 GPU 加载：**OOM**<br>`-ncmoe 18`：4.85 tok/s<br>`-ot exps=CPU`：3.0 tok/s | **17.9 tok/s**<br>（槽池 15 GB，命中率 97%） | **3.7×** |
| Q8_0（42 GB） | `-ot exps=CPU`：1.07 tok/s<br>`-ncmoe 30`：0.48 tok/s | **9.4 tok/s**<br>（槽池 15.5 GB，命中率 94%） | **8.8×** |

同一段 1619 个 token 的代码上，模型质量随量化精度提升：

| | Q3_K_XL（常驻） | Q4_K_XL（流式） | Q8_0（流式） |
|---|---|---|---|
| 困惑度 | 3.12 | 2.875 | **2.828** |

也就是说，同一台机器可以跑更好的模型，而且速度能用。

### 2. 时间：prefill 更快，处理长 prompt 不会 OOM

批量 prefill 采用"按层优先"的顺序，矩阵乘用 Metal tensor op（M5 的 Neural Accelerator），实测 10–12 TFLOPS。

**常驻模式（Q3_K_XL）**：

| prompt 长度 | llama.cpp | **flint-mac** |
|---|---|---|
| 512 | 516 tok/s（另一次测得 818 ± 186） | **891 tok/s** |
| 2048 | 590 tok/s（另一次测得 670 ± 109） | **989 tok/s** |
| 4000 | **OOM** | **906 tok/s** |

**流式模式**：

| prompt 长度 | llama.cpp Q4（`-ncmoe 18`） | **flint-mac Q4** | **flint-mac Q8** |
|---|---|---|---|
| 512 | 28 tok/s | **233 tok/s** | 90 tok/s |
| 2048 | 26 tok/s | **668 tok/s** | 337 tok/s |
| 4000 | — | **834 tok/s** | 638 tok/s |

流式模式下，Q4 处理 4000 个 token 时，从 SSD 读专家的时间已经完全藏在 GPU 计算后面（只等了 53 ms），速度接近常驻模式。

### 3. 常驻 decode：和 llama.cpp 持平略快

Q3_K_XL 全部常驻，两个引擎交替各跑多轮，取中位数：**flint-mac 50.6 tok/s，llama.cpp 48.0 tok/s（+5%）**。这个文件每个 token 要从内存读 2.0 GB 权重，我们已经用到约 83% 的实际可达带宽，剩下的提升空间不大，所以这里不算优势，只说明没有落后。

### 4. 能直接接编程 agent

- OpenAI 兼容的 `/v1/chat/completions`，支持 SSE 流式输出；
- 推理内容（`<think>`）放在 `reasoning_content`，工具调用转成 OpenAI 格式的 `tool_calls`，参数类型按工具的 JSON schema 判断；
- **跨轮复用 KV**：每轮只处理新增的 token。在 pi 的实际任务里，每轮都能复用 95% 以上的前缀（比如 1798/1805）；
- 用 pi + Q4 流式完成了一个真实任务：读代码、加函数、补测试，测试通过。

---

## 测试环境和方法

- **硬件**：MacBook Pro，Apple M5，24 GB 统一内存，内置 SSD（冷读实测约 5.7 GB/s）。
- **对照**：llama.cpp master（2026-10-23），Metal 后端，`-fa 1`，和我们使用同一个 GGUF 文件。
- **测速**：两个引擎**交替**各跑多轮，取中位数。这台笔记本的测量噪声约为 ±2%（后台有高负载程序时可达 ±10%）。
- **权重来源**：[unsloth/flint-GGUF](https://huggingface.co/unsloth/flint-GGUF)，包括 UD-Q3_K_XL、UD-Q4_K_XL 和 Q8_0。
- 所有数据**只在这一台机器上测过**。换成其他 Mac（内存、带宽、SSD 不同），结果会不一样。

---

## 我们做了什么

### 正确性优先

我们为每一层都建立了可对照的参照：

1. **CPU 参考实现**（`src/ref/`）：全程 fp32、逐 token 计算，和 llama.cpp 对比 30/30 个位置 top-1 一致，平均 KL 4e-5。
2. **llama.cpp 对照数据**（`tools/oracle.cpp`）：记录每个位置的 logits 和真实下一个 token 的 NLL。
3. **单元测试**：
   - `check-deq`：9 种量化格式的 GPU 解码和 CPU 逐位比较；
   - `test-mm`：批量矩阵乘；
   - `test-attn`：注意力，1300 个 token，跨越 SWA 窗口和环形缓冲回绕。
4. **整个模型**：
   - decode 和 llama.cpp 对比：30/30 一致，KL 2e-8；
   - 批量 prefill 和逐 token decode 对比：KL 2e-5；
   - 流式模式和常驻模式对比：结果完全一致。
5. **长文本困惑度**：在 1619 个 token 的重复代码上，我们和 llama.cpp 有 1.2% 的差异。逐项排查后确认这不是 bug：全 fp32 的参考实现和 GPU 结果一致，而 llama.cpp 自己换个设置（关闭 flash attention 或用 CPU 后端），NLL 就会波动 11%。3-bit 模型在这类文本上对数值误差极其敏感。

排查中修掉的一个真实 bug：ggml 的 `dequantize_q4_K` 有一步用 half 精度计算 `d/16`，当 d 很小时会落入次正规数而丢失精度。我们改成了 float 版本。

### Decode（`src/metal/flint.metal`、`src/metal/fused.metal`、`src/engine.mm`）

- **每个 token 只提交一个 command buffer**：路由在 GPU 上完成，不需要回到 CPU；用并发 encoder 加 barrier，互不依赖的调度同时执行（q/k/v；路由和共享专家）。
- **融合注意力**：QK RMSNorm、NEOX RoPE、写 KV cache、learned sink 都在一个 kernel 里完成；长上下文时自动把 KV 切成多段并行计算（split-K）。
- **SWA 层的 KV 只用 1024 个槽的环形缓冲**（窗口 513 加上一个 prefill 块），全注意力层才按上下文长度分配。
- **IQ3_XXS、IQ3_S、IQ2_S、IQ4_XS 各有专用的 32 元素点积函数**，码本放在 threadgroup memory，避免 constant 内存下标分散时被串行化。

### 专家流式加载（"空间"那一头）

- **数据驱动的设计**：我们先记录了真实的路由轨迹，再模拟不同的缓存策略。结论是：专家的使用在整体上并不集中（最常用的 10% 专家只承担 30–38% 的访问），所以常驻热门专家没有用；但时间局部性很强，LRU 在 65% 容量时命中率达到 90–92%。
- **槽池按专家尺寸分级**：不同层的专家尺寸不同，按最大尺寸统一分槽会浪费 25%。每级用独立的 buffer 和 LRU，太大时再按单个 MTLBuffer 的上限拆分。
- **`mlock` 锁住槽池**：residency set 只是请求常驻，并不锁页。内存一紧张，系统就会把槽池换出去，GPU 访问时缺页，速度会崩到 3 tok/s。
- **预测下一层的专家**：用下一层的 router 给当前层的 FFN 输入打分，6 个专家能猜中 72–76%（取 top-10 时召回率 85–88%）。命中率因此从 91% 提高到 97%。
- **每层通过 `MTLSharedEvent` 和 CPU 握手一次**（约 88 µs）。我们实测发现 GPU 一空等就会降频，每空等 1 ms 实际要多付出约 0.9 ms，所以优化的核心是尽量不让 GPU 停下来。

### 批量 prefill（`src/metal/prefill.metal`、`Engine::prefill`）

- **按层优先**：整个 prompt 先全部算完第 l 层，再进入第 l+1 层。流式模式下，每个专家在整个 prompt 里只需要读一次；算第 l 层的同时，后台整层预取第 l+1 层。
- **prefill 期间改用 MRU 淘汰**：按层顺序扫过所有专家是一种循环访问，LRU 在这种模式下会全部重读。改成淘汰"最近用过的"，缓存可以稳定保留一部分层。
- **专家矩阵乘用 32 个 token 一块**（普通矩阵乘是 128）：prompt 较短时每个专家分到的 token 很少，大分块大部分都是空算。
- **flash attention**：用 simdgroup 8×8 矩阵实现，因果遮罩、SWA 窗口和 sink 都在 kernel 里处理。
- **中间缓冲区复用**，并且只在 prefill 自己的 command buffer 上挂常驻集合，避免 decode 也被带着一起占用内存。

### 服务（`server/`，Rust）

- 引擎的 C API（`src/capi.h`）编成静态库，Rust 链接后得到单个可执行文件。
- HTTP 直接用标准库实现：每个连接一个线程，SSE 以关闭连接结束。依赖只有 `tokenizers`、`minijinja` 和 `serde_json`。
- **chat template 和官方逐 token 一致**：按 transformers 的设置开启 trim_blocks 和 lstrip_blocks；自己实现了与 Python `json.dumps(ensure_ascii=False)` 行为相同的 `tojson`（minijinja 自带的会转义 `<>&'`）。
- **前缀 KV 复用**：
  - 模型停止时生成的 EOG token（如 `<|observation|>`）也写进 KV，下一轮渲染出的 prompt 正好包含它，前缀能接得更完整；
  - SWA 层的环形缓冲只保留最近 1024 个位置，所以回退超过 511 个 token 时，会从回退点往前 512 个 token 开始重算。

---

## 经验教训

| 现象 | 结论 |
|---|---|
| 某个版本在串行 profile 里更快，实际并发执行时却慢 8% | 只有并发模式下的 A/B 测试和逐阶段消融结果可信（`tools/stage_cost.sh`） |
| 把 RMSNorm 融进每个矩阵乘反而变慢 | 每个 simdgroup 都要读一遍 x，而 IQ3 的权重行很紧凑，读激活的开销比读权重还大 |
| GPU 内用共享内存标志自旋握手，结果错误而且极慢 | GPU 执行中途，看不到 CPU 新写的数据，只能用 `MTLSharedEvent` |
| 流式 prefill 每次都要重读几乎所有专家 | 按层顺序扫描是循环访问，LRU 全部失效，改用 MRU |
| 槽池越大越快，直到系统开始换页 | 要锁页（`mlock`），自动大小也要参考系统当前的可用内存 |
| 常驻集合挂在 command queue 上，decode 时 OOM | 只挂在需要它的 command buffer 上（`useResidencySet:`） |

各模块的设计细节、数据和试过但没采用的方案见 [docs/DEVLOG.md](docs/DEVLOG.md)。

---

## 快速开始

**环境要求**：Apple Silicon Mac（只在 M5 24 GB 上测试过）、带 Metal Toolchain 的 Xcode（`xcodebuild -downloadComponent MetalToolchain`）、Rust。prefill 用到了 Metal 4 的 tensor op。

```sh
# 1. 下载权重、tokenizer 和 chat template（国内可设 HF_ENDPOINT=https://hf-mirror.com）
tools/fetch_model.sh Q4_K_XL            # 也可以下 Q3_K_XL / Q8_0

# 2. 编译引擎、Metal kernel 和服务
make
(cd server && cargo build --release)

# 3. 启动服务（默认 Q4_K_XL、32K 上下文、自动槽池大小）
tools/serve.sh                          # 监听 http://127.0.0.1:8080/v1
```

**接入 pi**：在 `~/.pi/agent/models.json` 里添加：

```json
{
  "providers": {
    "flint-local": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "models": [{
        "id": "flint", "name": "Flint (flint-mac)", "reasoning": true, "input": ["text"],
        "contextWindow": 32768, "maxTokens": 16384,
        "compat": { "supportsStore": false, "supportsReasoningEffort": false, "maxTokensField": "max_tokens" }
      }]
    }
  }
}
```

然后运行 `pi --provider flint-local --model flint`，或者在 pi 里用 `/model` 选择它。

**引擎自带的命令行工具**（`build/flint <命令> <model.gguf> …`）：

| 命令 | 用途 |
|---|---|
| `bench` | decode 测速（`FLINT_POOL_GB=auto` 时使用流式模式） |
| `bench-prefill` | prefill 测速 |
| `check` | 和 llama.cpp 的对照数据比较 logits 与 NLL |
| `check-prefill` | 批量 prefill 和逐 token decode 对比 |
| `check-deq` / `test-mm` / `test-attn` | kernel 单元测试 |
| `profile` | 按 kernel 统计 GPU 耗时 |
| `trace` | 记录路由轨迹（配合 `tools/cache_sim.py` 模拟缓存） |

生成对照数据的 `tools/oracle.cpp` 需要链接完整的 llama.cpp，把它放到 `third_party/llama.cpp/` 下编译即可（可选）。

**常用环境变量**：

| 变量 | 说明 |
|---|---|
| `FLINT_POOL_GB` | 槽池大小：`auto` 或 GB 数；不设置就是全部常驻 |
| `FLINT_PRED_K` | 预取个数，默认 8 |
| `FLINT_PF_STATS=1` | 打印 prefill 各阶段的耗时 |

---

## 限制

- **只支持 Flint**，测过的格式有 unsloth 的 UD-Q3_K_XL、UD-Q4_K_XL 和 Q8_0。官方的 NVFP4 GGUF 还不支持。
- **只在一台 M5 24 GB 上测过**。在内存更大的 Mac 上，常驻模式能放下更高精度的量化，流式模式的优势会小一些。
- **单会话、batch=1**：服务同一时间只处理一个请求。
- **流式 prefill 有固定开销**：prompt 只要超过约 100 个 token，每层就会用到绝大多数专家，所以每次都要把缓存放不下的专家扫一遍（Q4 约 1.7 秒）。
- **pi 实际使用时生成约 7–11 tok/s**，低于单独测试时的 18：每轮 prefill 扫进来的专家会把 decode 正在用的热专家挤出缓存，这还有改进空间。
- **长上下文的注意力 kernel 还没优化**：4000 个 token 时注意力占了 prefill 一半的时间。
- **常驻 Q3 + 32K 上下文在 24 GB 机器上会 GPU OOM**，常驻模式请用较小的 `--ctx`。

---

## 致谢与许可

- 本项目采用 **MIT** 许可证（见 [LICENSE](LICENSE)）。
- `third_party/ggml/` 下的文件，以及从 ggml 抽取、改写的代码，来自 [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)（MIT，The ggml authors）。详见 [third_party/ggml/README.md](third_party/ggml/README.md)。
- [Flint](https://huggingface.co/RecursiveSI/flint) 模型由 Recursive 发布（Apache-2.0）。本仓库**不包含**任何模型权重。
- GGUF 量化文件来自 [Unsloth](https://huggingface.co/unsloth/flint-GGUF)。
