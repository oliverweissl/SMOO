#!/usr/bin/env bash
# Shared vLLM lifecycle helpers; source this file from experiment scripts.
vllm_model_id() {
  case "$1" in
    qwen) echo "Qwen/Qwen3-VL-4B-Instruct" ;; kimi) echo "moonshotai/Kimi-VL-A3B-Instruct" ;;
    intern) echo "OpenGVLab/InternVL3_5-8B" ;; gemma) echo "google/gemma-3-4b-it" ;;
    deepseek) echo "deepseek-ai/deepseek-vl2-tiny" ;; nemotron) echo "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8" ;;
    *) echo "Unknown model: $1" >&2; return 1 ;;
  esac
}
start_vllm_server() {
  local model="$1" gpu="$2" port="$3" repo_root="$4" model_id
  model_id="$(vllm_model_id "$model")"
  mkdir -p "${repo_root}/logs"
  fuser -k "${port}/tcp" 2>/dev/null || true
  local -a args=(--port "$port" --enforce-eager --gpu-memory-utilization 0.8 --trust-remote-code --max-model-len 4096)
  [[ "${model_id,,}" == *nemotron* ]] && args+=(--default-chat-template-kwargs '{"enable_thinking": false}')
  sleep 1
  CUDA_VISIBLE_DEVICES="$gpu" vllm serve "$model_id" "${args[@]}" > "${repo_root}/logs/${model}_server.log" 2>&1 &
  SERVER_PID=$!
  until curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; do
    sleep 3
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: $model server died" >&2; return 1; }
  done
}
stop_vllm_server() {
  if [ -n "${SERVER_PID:-}" ]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; unset SERVER_PID; fi
}
