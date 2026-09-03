#!/bin/bash
# Driver: for each (ckpt, benchmark) combo, merge LoRA → run eval_vllm.py.
#
# Expected env from sbatch:
#   CKPT_STEPS         space-separated list, e.g. "30 50 60"
#   BENCHMARKS         space-separated, e.g. "amc23 math500 aime24 gpqa"
#   GPU_ID             which GPU (0-7) this driver shard uses
#   MAX_TOKENS         default 16384
#   TEMPERATURE        default 0.0
#   PROJECT_DIR        repo root (default: the parent of this script's directory)
#   CKPT_BASE          path to the directory holding the global_step_* checkpoints
#   EVAL_OUTPUT_DIR    where to write per-ckpt eval results
#   EVAL_LIMIT         optional; cap problems per benchmark (smoke tests)
#   DELETE_MERGED_AFTER=1  drop merged_hf once a step's benchmarks are done
#   PRUNE_FSDP_AFTER=1     drop FSDP shards / optimizer state (blocks resuming)

set -uo pipefail   # NOTE: no -e — we want a single benchmark failure (e.g. dataset 401) NOT to kill the rest
unset ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CKPT_BASE=${CKPT_BASE:?}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:?}
MAX_TOKENS=${MAX_TOKENS:-16384}
TEMPERATURE=${TEMPERATURE:-0.0}

mkdir -p "${EVAL_OUTPUT_DIR}"

for STEP in ${CKPT_STEPS}; do
    CKPT_DIR="${CKPT_BASE}/global_step_${STEP}/actor"
    MERGED="${CKPT_DIR}/merged_hf"

    if [ ! -d "${CKPT_DIR}/lora_adapter" ]; then
        echo "[run_eval] step ${STEP}: lora_adapter not found, skip"
        continue
    fi

    # 1. Merge LoRA if not already done
    if [ ! -f "${MERGED}/config.json" ]; then
        # Read base model from adapter_config.json so we don't hardcode the wrong one
        BASE=$(python3 -c "import json; print(json.load(open('${CKPT_DIR}/lora_adapter/adapter_config.json'))['base_model_name_or_path'])")
        echo "[run_eval] step ${STEP}: merging LoRA (base=${BASE}) -> ${MERGED}"
        python3 ${PROJECT_DIR}/eval/merge_lora.py \
            --base_model "${BASE}" \
            --lora_path "${CKPT_DIR}/lora_adapter" \
            --tokenizer_dir "${CKPT_DIR}/huggingface" \
            --output_dir "${MERGED}"
    else
        echo "[run_eval] step ${STEP}: merged_hf exists, reuse"
    fi

    # 2. For each benchmark
    for BENCH in ${BENCHMARKS}; do
        OUT="${EVAL_OUTPUT_DIR}/step_${STEP}"
        SUMMARY="${OUT}/$(basename ${MERGED})__${BENCH}__t${TEMPERATURE}.summary.json"
        if [ -f "${SUMMARY}" ]; then
            echo "[run_eval] step ${STEP} ${BENCH}: summary exists, skip"
            continue
        fi
        mkdir -p "${OUT}"
        echo "[run_eval] step ${STEP} ${BENCH}: starting at $(date)"
        if python3 ${PROJECT_DIR}/eval/eval_vllm.py \
                --model_path "${MERGED}" \
                --benchmark "${BENCH}" \
                --max_tokens ${MAX_TOKENS} \
                --temperature ${TEMPERATURE} \
                --tp 1 \
                --gpu_memory_utilization 0.85 \
                ${EVAL_LIMIT:+--limit ${EVAL_LIMIT}} \
                --output_dir "${OUT}" 2>&1 | tee "${OUT}/${BENCH}.log"; then
            echo "[run_eval] step ${STEP} ${BENCH}: done at $(date)"
        else
            echo "[run_eval] step ${STEP} ${BENCH}: FAILED, continuing"
        fi
    done

    # 3. Optionally delete merged_hf after all benchmarks are done (saves disk)
    if [ "${DELETE_MERGED_AFTER:-0}" = "1" ]; then
        echo "[run_eval] step ${STEP}: deleting merged_hf to free disk"
        rm -rf "${MERGED}"
    fi

    # 4. Optionally prune FSDP shards / optimizer state, keep only lora_adapter + huggingface
    # (~18G → ~280MB per ckpt). Use only if we don't need to resume training from this ckpt.
    if [ "${PRUNE_FSDP_AFTER:-0}" = "1" ]; then
        echo "[run_eval] step ${STEP}: pruning FSDP shards / optimizer state"
        rm -f "${CKPT_DIR}"/model_world_size_*_rank_*.pt
        rm -f "${CKPT_DIR}"/optim_world_size_*_rank_*.pt
        rm -f "${CKPT_DIR}"/extra_state_world_size_*_rank_*.pt
        # data.pt at the parent (per-step) level is small (~MB) but optional
        rm -f "${CKPT_BASE}/global_step_${STEP}/data.pt" 2>/dev/null || true
    fi
done

echo "[run_eval] all done at $(date)"
