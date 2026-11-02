set -euo pipefail
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=6000
export HCCL_EXEC_TIMEOUT=6000

# log
export ASCEND_SLOG_PRINT_TO_STDOUT=0   # 日志打屏, 可选
export ASCEND_GLOBAL_LOG_LEVEL=3       # 日志级别常用 1 INFO级别; 3 ERROR级别
export ASCEND_GLOBAL_EVENT_ENABLE=0    # 默认不使能event日志信息
export ASCEND_LAUNCH_BLOCKING=0        # 默认不开启算子下发同步，影响训练性能; 开启后每执行完一个算子会做一次流同步

NNODES="1"
NODE_RANK="0"
NGPUS_PER_NODE="1"
NUM_PROCESSES=$(($NGPUS_PER_NODE * $NNODES))

MASTER_PORT="1235"
MASTER_ADDR="127.0.0.1"

PIPELINE_FAILED=0

run_step () {
  local step_name="$1"
  shift

  # After first failure, skip all remaining steps.
  [[ "${PIPELINE_FAILED}" -ne 0 ]] && return 0

  echo -e "\n=== ${step_name} ==="

  if "$@"; then
    echo "✅  Step OK: ${step_name}"
  else
    local status=$?
    echo "❌  Step FAILED (exit code ${status}): ${step_name}"
    PIPELINE_FAILED=1
    exit 1
  fi
}

METHOD_NAME="GAE_qwen"
ALIAS="dummy_run"
SEED=42

RQVAE_DIR="./RQ-VAE"
CKPT_BASE="./quantizer_checkpoints/${ALIAS}/${ALIAS}"
DATASET="/home/ma-user/work/dataset"

INTERACTIONS_PATH="${DATASET}/ag_gr_data_sync_wulan/20260309/date_level_train_eval_out_sharded_subsampled_pr_v2/train_3"
EMB_PATH="${DATASET}/semantic_data/20260301/embeddings/20260301_qwen3_emb_06B_name_categories_desc_ag-rank_20260309_subset_normalized_id.parquet"
SAVE_MODEL_DIR="sampled_run_model"
# CONFIG="${RQVAE_DIR}/configs/local_run.yaml"
CONFIG="${RQVAE_DIR}/configs/mtp_rqvae.yaml"


run_step "Train RQ-VAE" \
# torchrun --nproc_per_node=${NGPUS_PER_NODE} \
#          --nnodes=${NNODES} \
#          --node_rank=${NODE_RANK} \
#          --master_addr=${MASTER_ADDR} \
#          --master_port=${MASTER_PORT} \
                      python -m pdb "${RQVAE_DIR}/main.py" \
                        --config="${CONFIG}" \
                        --ckpt_dir="${CKPT_BASE}" \
                        --data_path="${EMB_PATH}" \
                        --interactions_path="${INTERACTIONS_PATH}" \
                        --saved_model_dir="${SAVE_MODEL_DIR}"


MODEL_ROOT="${CKPT_BASE}/${SAVE_MODEL_DIR}"

run_step "Generate indices" \
  python "${RQVAE_DIR}/generate_indices.py" \
    --save_parquet \
    --dataset "${ALIAS}" \
    --root_path "${MODEL_ROOT}/" \
    --interactions_path "${INTERACTIONS_PATH}" \
    --seed "${SEED}"
