# Shared helpers for the experiment drivers. Source it, do not execute it.
# Every driver runs from the repository root.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
JULIA="${JULIA:-julia}"
JL=("$JULIA" --project="$ROOT_DIR" --threads auto)

_bg_pids=()

start_bg() {
  # start_bg <logfile> <cmd...>
  local log="$1"; shift
  "$@" >"$log" 2>&1 &
  _bg_pids+=("$!")
  echo "  started pid $! ($*) -> $log"
}

stop_pid() {
  local pid="$1"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

cleanup_bg() {
  for pid in "${_bg_pids[@]:-}"; do
    [[ -n "$pid" ]] && stop_pid "$pid"
  done
}
trap cleanup_bg EXIT INT TERM

wait_http() {
  # wait_http <url> <timeout_s>
  local url="$1" timeout="${2:-60}" t=0
  until curl -sf "$url" >/dev/null 2>&1; do
    sleep 1; t=$((t + 1))
    if (( t >= timeout )); then echo "timeout waiting for $url" >&2; return 1; fi
  done
}

api() {
  # api <METHOD> <url> [json-body]
  if [[ $# -ge 3 ]]; then
    curl -sf -X "$1" -H 'Content-Type: application/json' -d "$3" "$2"
  else
    curl -sf -X "$1" "$2"
  fi
  echo
}

wait_cost_source() {
  # wait_cost_source <base_url> <timeout_s> <since_epoch_s> <source...>
  # Set WAIT_PID to abort as soon as that process (the controller) exits.
  # Blocks until a controller has posted a /cost newer than since_epoch_s (i.e. it is compiled and running).
  local base="$1" timeout="$2" since="$3"; shift 3
  local t=0
  until curl -sf "$base/cost" | "$PY" -c '
import json, sys
c = json.load(sys.stdin)
ok = str(c.get("source", "")).lower() in sys.argv[2:] and float(c.get("timestamp") or 0) > float(sys.argv[1])
sys.exit(0 if ok else 1)' "$since" "$@" 2>/dev/null; do
    if [[ -n "${WAIT_PID:-}" ]] && ! kill -0 "$WAIT_PID" 2>/dev/null; then
      echo "controller process $WAIT_PID exited before posting /cost" >&2; return 1
    fi
    sleep 2; t=$((t + 2))
    if (( t >= timeout )); then echo "timeout waiting for controller ($*) on $base" >&2; return 1; fi
  done
}
