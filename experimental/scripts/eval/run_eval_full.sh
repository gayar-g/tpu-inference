#!/bin/bash
# run_eval_full.sh
# Full evaluation of Gemma 4 on the complete MMLU-Pro benchmark using TPU

set -eo pipefail

ENV_DIR="/mnt/pd/gayar/src/vllm_env"
OUTPUT_DIR="/mnt/pd/gayar/src/eval_logs"

mkdir -p "${OUTPUT_DIR}"

export VLLM_TARGET_DEVICE=tpu
export VLLM_PLATFORM=tpu
export USE_BATCHED_RPA_KERNEL=1
export MODEL_IMPL_TYPE=vllm

# Gemma 4 models available:
# - "google/gemma-4-31B-it" (Dense 31B - best direct replacement for Qwen3-32B)
# - "google/gemma-4-26B-A4B-it" (Mixture of Experts 26B)
# - "google/gemma-4-12B-it" (Medium-sized dense)
# - "google/gemma-4-E4B-it" (Effective 4B)
# - "google/gemma-4-E2B-it" (Effective 2B)
MODEL_ID="google/gemma-4-31B-it"

MODEL_ARGS_COMPACT='{"pretrained":"'"${MODEL_ID}"'","tensor_parallel_size":4,"kv_cache_dtype":"fp8","max_model_len":8192,"max_num_batched_tokens":4096,"max_num_seqs":256,"gpu_memory_utilization":0.85,"enable_prefix_caching":false,"compilation_config":{"cudagraph_capture_sizes":[]},"additional_config":{"quantization":{"qwix":{"rules":[{"module_path":".*","weight_qtype":"float8_e4m3fn","act_qtype":"float8_e4m3fn"}]}}},"enable_thinking":false}'

echo "Running full MMLU-Pro evaluation with Gemma 4 on TPU..."
"${ENV_DIR}/bin/python3" -m lm_eval \
  --model vllm \
  --model_args "${MODEL_ARGS_COMPACT}" \
  --tasks mmlu_pro \
  --apply_chat_template \
  --verbosity DEBUG \
  --log_samples \
  --output_path "${OUTPUT_DIR}" \
  --seed 42

