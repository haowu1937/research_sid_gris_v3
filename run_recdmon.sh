#!/bin/bash

set -euo pipefail

# =========================
# CONFIG
# =========================
DATASET="/home/ma-user/work/dataset"

# INTERACTIONS_PATH="${DATASET}/service_dis_GR_recall_ulan_preprocess/20260728/tran_orc_to_parquet_irc_preprocessed_2_only_train"
# EMB_PATH="${DATASET}/semantic_data/20260301/embeddings/20260301_qwen3_emb_06B_name_categories_desc_sd_20260728_subset.parquet"
INTERACTIONS_PATH="${DATASET}/servicedis_GR/20260728/test_scene/test_model/data/train"
EMB_PATH="${DATASET}/semantic_data/20260301/embeddings/20260301_qwen3_emb_06B_name_categories_desc_sd-rank_20260728.parquet"

ALIAS="dmon_$(date +%F_%H-%M-%S)"
OUT_DIR="./mincut_runs/${ALIAS}"
GRAPH_CACHE="./graph_cache/${ALIAS}/pyg_data.pt"

DEVICE="npu:0"

pause_on_exit() {
  status=$?

  echo
  if [ "$status" -eq 0 ]; then
    echo "Done. Results in ${OUT_DIR}"
  else
    echo "FAILED with exit code ${status}"
  fi

  echo
  read -r -p "Press Enter to close terminal..."

  exit "$status"
}

trap pause_on_exit EXIT

# =========================
# RUN
# =========================
python ./RQ-VAE/hierarchical_mincut_clustering.py \
  --interactions_path "${INTERACTIONS_PATH}" \
  --embeddings_path "${EMB_PATH}" \
  --method recursive_dmon \
  --graph_type adjacent_cooc_prod \
  --recursive_factor 8 \
  --recursive_levels 4 \
  --recursive_epochs 100 \
  --device "${DEVICE}" \
  --output_dir "${OUT_DIR}" \
  --graph_builder_dump_path "${GRAPH_CACHE}" \
  --max_dense_nodes 100000