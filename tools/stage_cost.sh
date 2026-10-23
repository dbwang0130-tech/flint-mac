#!/bin/sh
# 并发模式下逐阶段消融：每轮依次跑 base 与去掉各阶段的版本，输出各阶段的 GPU ms/token 贡献（中位数）。
# 用法: tools/stage_cost.sh <flint 可执行文件> <model> <rounds> stage1 stage2 ...
bin=$1; model=$2; rounds=$3; shift 3
for r in $(seq 1 "$rounds"); do
  for st in base "$@"; do
    if [ "$st" = base ]; then sk=""; else sk="$st"; fi
    v=$(FLINT_SKIP="$sk" "$bin" bench "$model" 48 0 | grep "avg GPU" | awk '{print $3}')
    echo "$st $v"
  done
done | awk '{v[$1]=v[$1]" "$2} END {
  for (k in v) { n=split(v[k],a," "); for(i=1;i<=n;i++)for(j=i+1;j<=n;j++)if(a[j]<a[i]){t=a[i];a[i]=a[j];a[j]=t}; med[k]=(n%2?a[(n+1)/2]:(a[n/2]+a[n/2+1])/2) }
  printf "base GPU %.2f ms/token\n", med["base"];
  for (k in med) if (k!="base") printf "  %-14s %6.2f ms\n", k, med["base"]-med[k] }'
