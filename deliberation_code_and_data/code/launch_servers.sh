#!/bin/bash
# ==============================================================================
# Starts one vLLM server per agent (11 agents, ports 8001-8011).
#
#   ./launch_servers.sh start                  # start all servers in the background
#   ./launch_servers.sh stop                   # stop all servers
#   ./launch_servers.sh status                 # show which ports respond
#   ./launch_servers.sh restart <Agent_NN>     # restart one agent's server
#   ./launch_servers.sh cleanup-orphans        # kill leftover vLLM engine processes
#
# Settings below (GPU assignment and GPU memory fraction per model) were tuned for
# two 80+ GB GPUs; adjust GPUS and GMU for other hardware. Every server uses a
# 32,768-token context window, enough for the full debate history.
#
# Precision: eight models use 4-bit AWQ checkpoints; Gemma 3 4B is quantised
# to FP8 at load time; Gemma 3 1B and Ministral 3 3B run in 16-bit precision.
# ==============================================================================
set -uo pipefail

LOG_DIR="$(dirname "$0")/logs"
PID_DIR="$(dirname "$0")/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

AGENTS=(  "Agent_01" "Agent_02" "Agent_03" "Agent_04" "Agent_05" "Agent_06" "Agent_07" "Agent_08" "Agent_09" "Agent_10" "Agent_11")
MODELS=(  "Qwen/Qwen2.5-3B-Instruct-AWQ" \
          "Qwen/Qwen2.5-7B-Instruct-AWQ" \
          "Qwen/Qwen2.5-32B-Instruct-AWQ" \
          "google/gemma-3-1b-it" \
          "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4" \
          "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" \
          "gaunernst/gemma-3-12b-it-int4-awq" \
          "google/gemma-3-4b-it" \
          "mistralai/Ministral-3-3B-Instruct-2512" \
          "solidrust/Mistral-7B-Instruct-v0.3-AWQ" \
          "casperhansen/llama-3.2-3b-instruct-awq")
PORTS=(   8001 8002 8003 8004 8005 8006 8007 8008 8009 8010 8011)
GPUS=(    1    0    1    0    0    1    0    0    0    0    0)
GMU=(     0.065 0.09 0.30 0.07 0.12 0.61 0.24 0.14 0.10 0.099 0.075)
QUANTIZE=("no" "no" "no" "no" "no" "no" "no" "fp8" "no" "no" "no")

MAX_MODEL_LEN=32768
MAX_NUM_SEQS=16

launch_one() {
  local i="$1"
  local agent="${AGENTS[$i]}"
  local model="${MODELS[$i]}"
  local port="${PORTS[$i]}"
  local gpu="${GPUS[$i]}"
  local gmu="${GMU[$i]}"
  local quant="${QUANTIZE[$i]}"

  local quant_flags=()
  if [ "$quant" = "yes" ]; then
    quant_flags=(--quantization bitsandbytes --load-format bitsandbytes)
  elif [ "$quant" = "fp8" ]; then
    quant_flags=(--quantization fp8)
  fi

  local eager_flags=(--enforce-eager)

  echo "Starting $agent ($model) on port $port, GPU $gpu, gpu-memory-utilization $gmu, quantized=$quant"
  CUDA_VISIBLE_DEVICES="$gpu" setsid nohup vllm serve "$model" \
    --port "$port" \
    --gpu-memory-utilization "$gmu" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --dtype bfloat16 \
    "${quant_flags[@]+"${quant_flags[@]}"}" \
    "${eager_flags[@]+"${eager_flags[@]}"}" \
    > "$LOG_DIR/${agent}.log" 2>&1 &
  echo $! > "$PID_DIR/${agent}.pid"
}

wait_for_ready() {
  local port="$1"
  local agent="$2"
  local max_wait="${3:-600}"
  local waited=0
  while [ "$waited" -lt "$max_wait" ]; do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/v1/models" --max-time 3 2>/dev/null || echo "000")
    if [ "$code" = "200" ]; then
      echo "  -> $agent is UP (took ${waited}s)"
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
    if [ $((waited % 30)) -eq 0 ]; then
      echo "  ... still waiting for $agent (${waited}s elapsed)"
    fi
  done
  echo "  -> WARNING: $agent did not become ready within ${max_wait}s - continuing to the"
  echo "     next agent anyway. Check logs/${agent}.log for what's actually happening."
  return 1
}

start_all() {
  echo "Launching GPU 0 and GPU 1's agents, one at a time per GPU"
  echo "(simultaneous start-up on the same GPU can cause"
  echo "memory-profiling contention). This takes a while."
  echo ""

  (
    for i in "${!AGENTS[@]}"; do
      if [ "${GPUS[$i]}" = "0" ]; then
        launch_one "$i"
        wait_for_ready "${PORTS[$i]}" "${AGENTS[$i]}"
      fi
    done
  ) &
  local gpu0_pid=$!

  (
    for i in "${!AGENTS[@]}"; do
      if [ "${GPUS[$i]}" = "1" ]; then
        launch_one "$i"
        wait_for_ready "${PORTS[$i]}" "${AGENTS[$i]}"
      fi
    done
  ) &
  local gpu1_pid=$!

  wait "$gpu0_pid" "$gpu1_pid"

  echo ""
  echo "Launch sequence complete. Verifying all 12 actually came up (wait_for_ready"
  echo "only warns and moves on for one that times out - it does not stop the launch"
  echo "or fail loudly, so check explicitly here instead of assuming success):"
  echo ""
  local any_down=0
  for i in "${!AGENTS[@]}"; do
    agent="${AGENTS[$i]}"
    port="${PORTS[$i]}"
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/v1/models" --max-time 3 2>/dev/null || echo "000")
    if [ "$code" = "200" ]; then
      echo "  $agent (port $port): UP"
    else
      echo "  $agent (port $port): NOT UP (http $code) - check logs/${agent}.log"
      any_down=1
    fi
  done
  echo ""
  if [ "$any_down" = "1" ]; then
    echo "WARNING: at least one agent is not responding. Do NOT start the sweep yet -"
    echo "fix it first (try: ./launch_servers.sh restart <Agent_NN>)."
    return 1
  fi
  echo "All 11 servers are up."
}

stop_one() {
  local agent="$1"
  local pid_file="$PID_DIR/${agent}.pid"
  if [ ! -f "$pid_file" ]; then
    echo "$agent: no pid file, nothing to stop"
    return 0
  fi
  local pid
  pid=$(cat "$pid_file")
  rm -f "$pid_file"
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  echo "Stopping $agent (process group $pid)"
  kill -TERM -- "-$pid" 2>/dev/null
  local waited=0
  while [ "$waited" -lt 15 ] && kill -0 "$pid" 2>/dev/null; do
    sleep 1
    waited=$((waited + 1))
  done
  kill -KILL -- "-$pid" 2>/dev/null
}

cleanup_orphans() {
  local found=0
  while read -r opid; do
    [ -z "$opid" ] && continue
    echo "Killing orphaned EngineCore process (pid $opid, ppid=1)"
    kill -9 "$opid" 2>/dev/null
    found=1
  done < <(ps -eo pid,ppid,cmd | awk '$1 != "PID" && $2==1 && $0 ~ /EngineCore/ {print $1}')
  if [ "$found" = "0" ]; then
    echo "No orphaned EngineCore processes found."
  fi
}

stop_all() {
  for i in "${!AGENTS[@]}"; do
    stop_one "${AGENTS[$i]}"
  done
}

restart_one() {
  local agent_name="$1"
  if [ -z "$agent_name" ]; then
    echo "Usage: $0 restart <Agent_NN>"
    exit 1
  fi
  local idx=-1
  for i in "${!AGENTS[@]}"; do
    if [ "${AGENTS[$i]}" = "$agent_name" ]; then idx="$i"; fi
  done
  if [ "$idx" = "-1" ]; then
    echo "Unknown agent: $agent_name (expected one of: ${AGENTS[*]})"
    exit 1
  fi

  stop_one "$agent_name"
  sleep 3

  local gpu="${GPUS[$idx]}" gmu="${GMU[$idx]}"
  local free_mib total_mib needed_mib
  free_mib=$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
  total_mib=$(nvidia-smi --id="$gpu" --query-gpu=memory.total --format=csv,noheader,nounits)
  needed_mib=$(awk -v t="$total_mib" -v g="$gmu" 'BEGIN{printf "%d", (t*g)+717+0.999}')
  echo "GPU $gpu: ${free_mib} MiB free, need ~${needed_mib} MiB for $agent_name at gpu-memory-utilization=$gmu"
  if [ "$free_mib" -lt "$needed_mib" ]; then
    echo "REFUSING to start $agent_name: not enough real free memory."
    echo "Check for orphaned processes first: ps -eo pid,ppid,cmd | grep EngineCore"
    echo "(look for any EngineCore entry with ppid=1 - kill -9 it, then retry this command)"
    return 1
  fi

  launch_one "$idx"
  wait_for_ready "${PORTS[$idx]}" "$agent_name"
}

status_all() {
  for i in "${!AGENTS[@]}"; do
    agent="${AGENTS[$i]}"
    port="${PORTS[$i]}"
    if curl -s -o /dev/null -w "" "http://localhost:${port}/v1/models" --max-time 3; then
      code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/v1/models" --max-time 3)
      if [ "$code" = "200" ]; then
        local q
        q=$(grep "Running:" "$LOG_DIR/${agent}.log" 2>/dev/null | tail -1 | grep -oE "Running: [0-9]+ reqs, Waiting: [0-9]+ reqs, GPU KV cache usage: [0-9.]+%")
        echo "$agent (port $port): UP${q:+ | $q}"
      else
        echo "$agent (port $port): NOT READY (http $code - probably still loading)"
      fi
    else
      echo "$agent (port $port): DOWN"
    fi
  done
}

case "${1:-}" in
  start) start_all ;;
  stop) stop_all ;;
  status) status_all ;;
  restart) restart_one "${2:-}" ;;
  cleanup-orphans) cleanup_orphans ;;
  *) echo "Usage: $0 {start|stop|status|restart <Agent_NN>|cleanup-orphans}"; exit 1 ;;
esac
