#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

PY="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
SERVICE="$ROOT_DIR/src/services/realtime_powerflow_service_routed_apicontrolled.py"
PISR="$ROOT_DIR/src/controllers/rt_controller_showerror_routed.jl"
SENS="$ROOT_DIR/src/controllers/rt_sensitivity_traindata_routed.jl"

PORT="${PORT:-8012}"
HOST="${HOST:-127.0.0.1}"
PF_RATE="${PF_RATE:-20}"
SCENARIO_RATE="${SCENARIO_RATE:-1}"
LOOP_PERIOD_S="${LOOP_PERIOD_S:-0.10}"   # 10 Hz controllers (Sec. IV-B)
PARAMS_FILE="$SCRIPT_DIR/controller_params.env"
WARMUP_CONTROLLERS="${WARMUP_CONTROLLERS:-true}"
WARMUP_SETTLE_S="${WARMUP_SETTLE_S:-1.5}"
WARMUP_TIMEOUT_S="${WARMUP_TIMEOUT_S:-0}"

if [[ ! -x "$PY" ]]; then
  echo "Missing Python venv interpreter: $PY" >&2
  exit 1
fi
if ! command -v julia >/dev/null 2>&1; then
  echo "Julia not found in PATH" >&2
  exit 1
fi

if [[ -f "$PARAMS_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$PARAMS_FILE"
fi

export PF_BASE_URL="http://${HOST}:${PORT}"
export LOOP_PERIOD_S
export VERBOSE="${VERBOSE:-false}"

pids=()

cleanup() {
  for pid in "${pids[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait || true
}
trap cleanup EXIT INT TERM

assert_no_stale_routed_processes() {
  local matches
  matches="$("$PY" - "$$" "$SERVICE" "$PISR" "$SENS" <<'PY'
import os
import subprocess
import sys

self_pid = int(sys.argv[1])
targets = [os.path.realpath(p) for p in sys.argv[2:]]

try:
    out = subprocess.check_output(["ps", "-ef"], text=True)
except Exception:
    sys.exit(0)

hits = []
for line in out.splitlines():
    if not line.strip():
        continue
    parts = line.split(None, 7)
    if len(parts) < 8:
        continue
    try:
        pid = int(parts[1])
    except Exception:
        continue
    cmd = parts[7]
    if pid in (self_pid, os.getpid()):
        continue
    if any(target in cmd for target in targets):
        hits.append(f"{pid}\t{cmd}")

if hits:
    print("\n".join(hits))
PY
)"
  if [[ -n "$matches" ]]; then
    echo "[run_routed_stack] Refusing to start because routed processes are already running:" >&2
    echo "$matches" >&2
    echo "[run_routed_stack] Stop the old stack first, then start a fresh one." >&2
    exit 1
  fi
}

start_bg() {
  "$@" &
  pids+=("$!")
}

wait_http_ready() {
  local url="$1"
  local timeout_s="$2"
  "$PY" - "$url" "$timeout_s" <<'PY'
import json
import sys
import time
import urllib.request

url = sys.argv[1]
timeout_s = float(sys.argv[2])
deadline = time.time() + timeout_s
last_err = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=1.0) as resp:
            if 200 <= resp.status < 300:
                sys.exit(0)
    except Exception as exc:
        last_err = exc
    time.sleep(0.2)
print(f"timeout waiting for {url}: {last_err}", file=sys.stderr)
sys.exit(1)
PY
}

post_mode() {
  local mode="$1"
  "$PY" - "$PF_BASE_URL/control_router" "$mode" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1]
mode = sys.argv[2]
req = urllib.request.Request(
    url=url,
    method="POST",
    data=json.dumps({"mode": mode}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=2.0) as resp:
    if not (200 <= resp.status < 300):
        raise RuntimeError(f"POST {url} failed with status {resp.status}")
PY
}

warmup_controller() {
  local source="$1"
  local mode="$2"
  local timeout_s="${3:-0}"
  "$PY" - "$PF_BASE_URL" "$source" "$mode" "$timeout_s" <<'PY'
import json
import sys
import time
import urllib.request
import urllib.error

base = sys.argv[1].rstrip("/")
source = sys.argv[2]
mode = sys.argv[3]
timeout_s = float(sys.argv[4])

def jget(path, timeout=30.0):
  with urllib.request.urlopen(base + path, timeout=timeout) as resp:
    return json.loads(resp.read().decode("utf-8"))

def jpost(path, body, timeout=30.0):
  req = urllib.request.Request(
    url=base + path,
    method="POST",
    data=json.dumps(body).encode("utf-8"),
    headers={"Content-Type": "application/json"},
  )
  with urllib.request.urlopen(req, timeout=timeout) as resp:
    return json.loads(resp.read().decode("utf-8"))

def jdelete(path, timeout=30.0):
  req = urllib.request.Request(url=base + path, method="DELETE")
  with urllib.request.urlopen(req, timeout=timeout) as resp:
    if resp.status < 200 or resp.status >= 300:
      raise RuntimeError(f"DELETE {path} failed ({resp.status})")

def retry_call(fn, *args, **kwargs):
  while True:
    try:
      return fn(*args, **kwargs)
    except Exception:
      time.sleep(0.2)

status = retry_call(jget, "/status")
ctrl_buses = [int(b) for b in status.get("opf_controlled_buses", [])]
if not ctrl_buses:
  ctrl_buses = [int(b) for b in status.get("controlled_buses", [])]
if not ctrl_buses:
  raise RuntimeError("no controllable buses available for warmup")

# Activate the route first (the service rejects mods from an inactive source with 409), then
# seed tiny non-zero controller mods so PISR doesn't short-circuit on the first enable.
retry_call(jpost, "/control_router", {"mode": mode})
items = [{"bus_ext": b, "dP_mw": 0.0, "dQ_mvar": 0.001} for b in ctrl_buses]
retry_call(jpost, f"/controller/{source}/mods", {"items": items})

deadline = None if timeout_s <= 0 else (time.time() + timeout_s)
last_cost = None
while True:
  if deadline is not None and time.time() >= deadline:
    raise RuntimeError(f"warmup timeout for {source}; last_cost={last_cost}")
  cost = retry_call(jget, "/cost")
  last_cost = cost
  src = str(cost.get("source", "")).strip().lower()
  val = cost.get("value")
  accepted_sources = {source}
  if source == "pisr":
    accepted_sources.add("julia")
  if src in accepted_sources and val is not None:
    break
  time.sleep(0.2)

retry_call(jdelete, f"/controller/{source}/mods")
PY
}

assert_alive() {
  local pid="$1"
  local name="$2"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    echo "[run_routed_stack] ${name} process is not running (pid=${pid})" >&2
    exit 1
  fi
}

assert_no_stale_routed_processes

echo "[run_routed_stack] Starting service on ${HOST}:${PORT}"
start_bg "$PY" "$SERVICE" --host "$HOST" --port "$PORT" --pf-rate "$PF_RATE" --scenario-rate "$SCENARIO_RATE"
service_pid="${pids[$((${#pids[@]} - 1))]}"

echo "[run_routed_stack] Waiting for service readiness"
wait_http_ready "$PF_BASE_URL/status" 25
assert_alive "$service_pid" "service"

echo "[run_routed_stack] Starting routed PISR controller"
start_bg julia --project=. "$PISR"
pisr_pid="${pids[$((${#pids[@]} - 1))]}"

echo "[run_routed_stack] Starting routed sensitivity controller"
start_bg julia --project=. "$SENS"
sens_pid="${pids[$((${#pids[@]} - 1))]}"

sleep 1
assert_alive "$pisr_pid" "PISR"
assert_alive "$sens_pid" "sensitivity"

if [[ "$WARMUP_CONTROLLERS" == "1" || "$WARMUP_CONTROLLERS" == "true" || "$WARMUP_CONTROLLERS" == "yes" ]]; then
  echo "[run_routed_stack] Warming controllers (pisr -> sensitivity -> none)"
  if ! warmup_controller "pisr" "pisr" "$WARMUP_TIMEOUT_S"; then
    echo "[run_routed_stack] WARN: PISR warmup timed out, continuing startup." >&2
  fi
  sleep "$WARMUP_SETTLE_S"
  if ! warmup_controller "sensitivity" "sensitivity" "$WARMUP_TIMEOUT_S"; then
    echo "[run_routed_stack] WARN: Sensitivity warmup timed out, continuing startup." >&2
  fi
  sleep "$WARMUP_SETTLE_S"
  post_mode "none"
fi

echo "[run_routed_stack] Stack running. UI: http://${HOST}:${PORT}/ui"
echo "[run_routed_stack] Press Ctrl+C to stop all processes."

wait
