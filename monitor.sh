#!/bin/bash

LOG=monitor.log
INTERVAL=2

echo "time,\
mem_used_MB,mem_total_MB,mem_used_pct,\
swap_used_MB,swap_total_MB,swap_used_pct,\
cpu_load_pct,\
gpu_mem_used_MB,gpu_mem_total_MB,gpu_mem_used_pct,gpu_util_pct" > $LOG

while true; do
  TS=$(date "+%F %T")

  # ===== CPU / MEM =====
  read MEM_USED MEM_TOTAL <<< $(free -m | awk '/Mem:/ {print $3, $2}')
  read SWAP_USED SWAP_TOTAL <<< $(free -m | awk '/Swap:/ {print $3, $2}')

  MEM_PCT=$(awk "BEGIN {printf \"%.1f\", $MEM_USED/$MEM_TOTAL*100}")
  SWAP_PCT=$(awk "BEGIN {printf \"%.1f\", ($SWAP_TOTAL==0?0:$SWAP_USED/$SWAP_TOTAL*100)}")

  # CPU 使用率（1 - idle）
  CPU_PCT=$(top -bn1 | awk '/Cpu\(s\)/ {printf "%.1f", 100-$8}')

  # ===== GPU =====
  GPU_INFO=$(nvidia-smi \
    --query-gpu=memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits | tr '\n' ';')

  # 支持多卡，逐卡展开
  echo "$GPU_INFO" | tr ';' '\n' | while read LINE; do
    [ -z "$LINE" ] && continue
    read GPU_MEM_USED GPU_MEM_TOTAL GPU_UTIL <<< $(echo $LINE | tr ',' ' ')
    GPU_MEM_PCT=$(awk "BEGIN {printf \"%.1f\", $GPU_MEM_USED/$GPU_MEM_TOTAL*100}")

    echo "$TS,\
$MEM_USED,$MEM_TOTAL,$MEM_PCT,\
$SWAP_USED,$SWAP_TOTAL,$SWAP_PCT,\
$CPU_PCT,\
$GPU_MEM_USED,$GPU_MEM_TOTAL,$GPU_MEM_PCT,$GPU_UTIL" >> $LOG
  done

  sleep $INTERVAL
done
