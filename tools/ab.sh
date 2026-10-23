#!/bin/sh
# 交替跑多组配置测 tg，减少发热/降频带来的偏差。
# 用法: tools/ab.sh <rounds> <label=command> ...   每条命令需输出含 "tok/s" 的一行
rounds=$1; shift
for r in $(seq 1 "$rounds"); do
  for spec in "$@"; do
    label=${spec%%=*}; cmd=${spec#*=}
    v=$(sh -c "$cmd" 2>/dev/null | grep -Eo '[0-9]+\.[0-9]+ (± [0-9.]+ )?\|?$|[0-9]+\.[0-9]+ tok/s' | tail -1 | grep -Eo '^[0-9]+\.[0-9]+')
    echo "$label $v"
    sleep 5
  done
done | awk '{v[$1]=v[$1]" "$2} END {for (k in v) {n=split(v[k],a," "); asort_n=n; for(i=1;i<=n;i++)for(j=i+1;j<=n;j++)if(a[j]<a[i]){t=a[i];a[i]=a[j];a[j]=t}; printf "%-10s median %.2f  [%s ]\n", k, (n%2?a[(n+1)/2]:(a[n/2]+a[n/2+1])/2), v[k]}}'
