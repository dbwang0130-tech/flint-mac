#!/bin/sh
# 启动 flint-server（OpenAI 兼容，http://127.0.0.1:8080/v1）。
# 用法: tools/serve.sh [model.gguf] [ctx]   默认 Q4_K_XL 流式加载、32K 上下文、自动槽池
# pi 接入：PI_CODING_AGENT_DIR="$PWD/.pi-flint" pi --provider flint-local --model flint
set -e
cd "$(dirname "$0")/.."
MODEL=${1:-models/flint-UD-Q4_K_XL.gguf}
CTX=${2:-32768}
make -s
(cd server && cargo build --release -q)
exec server/target/release/flint-server --model "$MODEL" --ctx "$CTX" --pool auto
