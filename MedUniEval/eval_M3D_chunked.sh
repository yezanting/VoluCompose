#!/usr/bin/env bash
# Run from any directory; relative dataset/video paths resolve from the repo root.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

DATASETS_PATH="${DATASETS_PATH:-$REPO_DIR/data/Eval}"
EVAL_DATASETS="${EVAL_DATASETS:-M3D}"
MODEL_NAME="${MODEL_NAME:-Hulumed_qwen2}"
MODEL_PATH="${MODEL_PATH:-$REPO_DIR/workspace/model/hulu_7B}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_DIR/workspace/eval/hulu}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/config/config.json}"
COMPRESSION="${COMPRESSION:-medpruner++}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${PYTHON:-python}"
USE_VLLM="${USE_VLLM:-False}"
SEED="${SEED:-42}"
REASONING="${REASONING:-False}"
TEST_TIMES="${TEST_TIMES:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_IMAGE_NUM="${MAX_IMAGE_NUM:-600}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-0.95}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
USE_LLM_JUDGE="${USE_LLM_JUDGE:-False}"
GPT_MODEL="${GPT_MODEL:-gpt-4.1-2025-04-14}"
OPENAI_API_KEY="${OPENAI_API_KEY:-None}"

# The inherited aggregator lets chunk 0 clear old files and the last chunk
# compute metrics. Keep shards sequential to avoid cleanup/merge races.
CHUNKS="${CHUNKS:-8}"
if ! [[ "$CHUNKS" =~ ^[1-9][0-9]*$ ]]; then
    echo "CHUNKS must be a positive integer" >&2
    exit 2
fi
for ((IDX=0; IDX<CHUNKS; IDX++)); do
    echo "Starting chunk $IDX of $CHUNKS"
"$PYTHON" "$SCRIPT_DIR/eval.py" \
    --eval_datasets "$EVAL_DATASETS" \
    --datasets_path "$DATASETS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --seed "$SEED" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_image_num "$MAX_IMAGE_NUM" \
    --use_vllm "$USE_VLLM" \
    --compression "$COMPRESSION" \
    --config "$CONFIG_PATH" \
    --num_chunks "$CHUNKS" \
    --chunk_idx "$IDX" \
    --reasoning "$REASONING" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --use_llm_judge "$USE_LLM_JUDGE" \
    --judge_gpt_model "$GPT_MODEL" \
    --openai_api_key "$OPENAI_API_KEY" \
    --test_times "$TEST_TIMES"
done
