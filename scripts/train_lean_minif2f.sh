#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Multi-turn RL training on miniF2F-lean4 with the Lean REPL agent.
#
# Prerequisites:
#   1. LeanDojo cache imported:
#        bash scripts/cache_leandojo_minif2f.sh import s3://rl-guidance/lean-dojo-cache-miniF2F
#   2. Setup script completed:
#        python scripts/setup_minif2f_leandojo.py
#   3. Ray cluster running:
#        ray start --head
#
# Usage:
#   bash scripts/train_lean_minif2f.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Model ----
#MODEL_PATH="${MODEL_PATH:-/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775}"
#MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Math-7B-Instruct}"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-Prover-V1.5-SFT}"
#MODEL_TAG="${MODEL_TAG:-Q2.5-M-7b}"
MODEL_TAG="${MODEL_TAG:-D1.5-P-7b-SFT}"

# ---- Data (local JSONL) ----
TRAIN_DATA="${WORK_DIR}/data/minif2f_valid.jsonl"
EVAL_DATA="${WORK_DIR}/data/minif2f_test.jsonl"
VAL_DATA="${WORK_DIR}/data/minif2f_few_shot.jsonl"
# ---- Agent ----
AGENT_FUNC_PATH="${WORK_DIR}/examples/python/agent_func_lean_minif2f.py"

# ---- Output ----
SAVE_PATH="${WORK_DIR}/exp/lean_minif2f_$(date +%Y%m%d)"


# ---- Lean REPL tuning (exported so the agent process picks them up) ----
export LEAN_REPL_TIMEOUT="${LEAN_REPL_TIMEOUT:-60}"    # seconds per REPL response
export LEAN_THREADS="${LEAN_THREADS:-1}"                # --threads for lake env lean
export LEAN_MEMORY="${LEAN_MEMORY:-8192}"              # --memory (MiB) for lake env lean
export LEAN_MAX_STEPS="${LEAN_MAX_STEPS:-10}"          # max tactic steps per proof
export HF_HUB_OFFLINE=0                                 # download model from hub 

# ============================================================================
# Arguments — grouped for readability
# ============================================================================

CKPT_ARGS=(
    --pretrain "${MODEL_PATH}"
    --hf_token "${HF_TOKEN}"
    --hf_hub_offline 0
    --load_checkpoint

    --save_path "${SAVE_PATH}"
    --ckpt_path "${SAVE_PATH}/ckpt"
    --save_hf_ckpt
    --max_ckpt_num 3
    --save_steps 20
)

ROLLOUT_ARGS=(
    --agent_func_path "${AGENT_FUNC_PATH}"

    --prompt_data "${TRAIN_DATA}"
    --input_key prompt
    --label_key label
    --apply_chat_template

    # --- Evaluation dataset (miniF2F test, 244 problems) ---
    --eval_dataset "${EVAL_DATA}"
    --eval_split train
    --eval_temperature 0.7
    --eval_n_samples_per_prompt 1

    # --- Sequence lengths ---
    # miniF2F prompts are ~200-800 tokens; tactic responses are short.
    --prompt_max_len 2048
    --generate_max_len 2048
    # stop_sequences default ("```\n") is set automatically in train_ppo_ray.py for agent mode

    # --- Batch sizes ---
    # Each rollout spawns a Lean REPL (~5s per step), so keep batch modest.
    --rollout_batch_size 16
    --n_samples_per_prompt 8
    --micro_rollout_batch_size 2

    --train_batch_size 64
    --micro_train_batch_size 2

    # --- Dataset / epoch control ---
    --max_samples 1000
    --max_epochs 1
    --num_episodes 25

    # --- Dynamic filtering: drop rollouts with reward outside [0, 1] ---
    # --dynamic_filtering
    # --dynamic_filtering_reward_range 0.0 1.0

    # --- Token-level dynamic batching ---
    --use_dynamic_batch
    --packing_samples
    --train_max_tokens_per_gpu 16384
    --rollout_max_tokens_per_gpu 32768
)

ENGINE_ARGS=(
    # NOTE: --async_train disabled for 0.5B sanity check (avoids NCCL weight-sync).
    # Re-enable for larger models: --async_train

    # --- GPU allocation ---
    # For 0.5B: 1 GPU actor + ref colocated, 1 vLLM engine TP=1
    # For 7B+:  bump actor_num_gpus_per_node=4, vllm_num_engines=2, vllm_tensor_parallel_size=2
    --actor_num_nodes 1
    --actor_num_gpus_per_node 8
    --ref_num_nodes 1
    --ref_num_gpus_per_node 8
    --colocate_all_models
    --deepspeed_enable_sleep
    --vllm_enable_sleep

    --vllm_num_engines 8
    --vllm_tensor_parallel_size 1
    --vllm_gpu_memory_utilization 0.6
    --vllm_sync_backend nccl
    --enforce_eager

    # --- DeepSpeed ---
    --zero_stage 2
    --gradient_checkpointing
    --param_dtype bf16
)

OPTIMIZER_ARGS=(
    --advantage_estimator group_norm #reinforce_baseline
    --actor_learning_rate 5e-7
    --entropy_loss_coef 0.0
    --init_kl_coef 1e-3
    --use_kl_loss
    --kl_estimator k3 #k2
)

LOG_ARGS=(
    --use_tensorboard "${SAVE_PATH}/runs"
    --logging_steps 1
    --eval_steps 30

    --use_wandb enabled
    --wandb_org "${WANDB_ENTITY:-}"
    --wandb_project guidance-lean
    --wandb_run_name "${MODEL_TAG}-lean_minif2f_$(date +%m%dT%H%M)"
)

# ============================================================================
# Launch
# ============================================================================

echo "================================================================"
echo "  Lean miniF2F RL Training"
echo "  Model:     ${MODEL_PATH}"
echo "  Model Tag: ${MODEL_TAG}"
echo "  Data:      ${TRAIN_DATA}"
echo "  Save:      ${SAVE_PATH}"
echo "  GPUs:      8× B200"
echo "  Agent:     ${AGENT_FUNC_PATH}"
echo "================================================================"

# Combine all arguments and submit the Ray job
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{
        "working_dir": "/root/Guidance",
        "excludes": [
            ".git",
            "third_party",
            "__pycache__",
            "*.pyc",
            ".venv"
        ]
    }' \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${ENGINE_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${LOG_ARGS[@]}"
