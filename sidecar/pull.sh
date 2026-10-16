#!/usr/bin/env bash
# usage: pull.sh v2base v2err
cd "$(dirname "$0")" || exit 1

status() { kaggle kernels status "${KAGGLE_OWNER:-admincoas}/lfm-sidecar-$1" 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*' | head -1; }

terminal() {
  case "$1" in *COMPLETE*|*ERROR*|*CANCEL*) return 0 ;; esac
  return 1
}

declare -A seen
for k in "$@"; do seen[$k]=""; done

for i in $(seq 1 400); do
  all_done=1
  for k in "$@"; do
    s=$(status "$k")
    [ -z "$s" ] && s="UNKNOWN"
    if [ "$s" != "${seen[$k]}" ]; then echo "$(date -u +%H:%M:%SZ) $k $s"; seen[$k]="$s"; fi
    terminal "$s" || all_done=0
  done
  [ "$all_done" -eq 1 ] && break
  sleep 300
done

for k in "$@"; do
  mkdir -p "kaggle_out/$k"
  if [ -f "kaggle_out/$k/train.log" ]; then
    n=$(ls "kaggle_out/$k" | grep -c '^train-v[0-9]*\.log$')
    mv "kaggle_out/$k/train.log" "kaggle_out/$k/train-v$n.log"
    echo "archived $k/train.log -> train-v$n.log"
  fi
  if kaggle kernels output "${KAGGLE_OWNER:-admincoas}/lfm-sidecar-$k" -p "kaggle_out/$k" --force >/dev/null 2>&1; then
    echo "$(date -u +%H:%M:%SZ) downloaded $k"
  else
    echo "$(date -u +%H:%M:%SZ) download failed $k"
  fi
done

python3 report.py "$@"
