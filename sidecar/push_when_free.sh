#!/usr/bin/env bash
# usage: push_when_free.sh v2sweep,v25sweep v3asst v3long
cd "$(dirname "$0")" || exit 1

IFS=',' read -r -a WAIT <<< "$1"
shift

status() { kaggle kernels status "${KAGGLE_OWNER:-admincoas}/lfm-sidecar-$1" 2>&1 | tr -d '\r'; }

live() {
  case "$(status "$1")" in
    *RUNNING*|*Queue*|*QUEUE*|*PENDING*|*Pending*) return 0 ;;
  esac
  return 1
}

push() {
  local k=$1 i out
  for i in $(seq 1 40); do
    out=$( cd "kaggle_pkg/kernel_$k" && kaggle kernels push 2>&1 | tr -d '\r' )
    case "$out" in
      *"push error"*|*"Maximum batch GPU session"*|*"409"*|*"403"*)
        echo "$(date -u +%H:%M:%SZ) $k rejected, retrying in 120s: $out"
        sleep 120
        continue
        ;;
    esac
    echo "$(date -u +%H:%M:%SZ) $k pushed: $out"
    return 0
  done
  echo "$(date -u +%H:%M:%SZ) $k NEVER PUSHED"
  return 1
}

for i in $(seq 1 720); do
  any=0
  for k in "${WAIT[@]}"; do live "$k" && any=1; done
  if [ "$any" -eq 1 ]; then
    if [ $((i % 10)) -eq 1 ]; then
      echo "$(date -u +%H:%M:%SZ) waiting on ${WAIT[*]}: $(status "${WAIT[0]}" | sed 's/.*has status //')"
    fi
    sleep 60
    continue
  fi
  echo "$(date -u +%H:%M:%SZ) slots free, pushing $*"
  break
done

for k in "$@"; do
  push "$k"
  sleep 45
done

for k in "$@"; do
  echo "$(date -u +%H:%M:%SZ) $k $(status "$k")"
done
