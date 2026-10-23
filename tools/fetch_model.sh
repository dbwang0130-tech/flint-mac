#!/bin/sh
# 下载 GGUF 权重与运行服务所需的 tokenizer / chat template。
# 用法: tools/fetch_model.sh [Q3_K_XL|Q4_K_XL|Q8_0 ...]   默认 Q4_K_XL
# 国内可设 HF_ENDPOINT=https://hf-mirror.com；装了 aria2c 会自动多线程下载
set -e
cd "$(dirname "$0")/.."
HF=${HF_ENDPOINT:-https://huggingface.co}
get() {  # get <repo> <file> <out>
    mkdir -p "$(dirname "$3")"
    if command -v aria2c >/dev/null; then
        aria2c -x 16 -s 16 -k 8M -c --file-allocation=none --console-log-level=warn \
            -d "$(dirname "$3")" -o "$(basename "$3")" "$HF/$1/resolve/main/$2"
    else
        curl -fL -C - -o "$3" "$HF/$1/resolve/main/$2"
    fi
}
get RecursiveSI/flint tokenizer.json ref/flint-hf/tokenizer.json
get RecursiveSI/flint chat_template.jinja ref/flint-hf/chat_template.jinja
for q in "${@:-Q4_K_XL}"; do
    case "$q" in
        Q8_0) f=flint-Q8_0.gguf ;;
        *) f=flint-UD-$q.gguf ;;
    esac
    get unsloth/flint-GGUF "$f" "models/$f"
done
