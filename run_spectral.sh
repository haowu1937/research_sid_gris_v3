#!/bin/bash

set -euo pipefail

# =========================
# CONFIG
# =========================
DATASET="/home/ma-user/work/dataset"

INTERACTIONS_PATH="${DATASET}/ag_gr_data_sync_wulan/20260309/date_level_train_eval_out_sharded_subsampled_pr_v2/train/"
EMB_PATH="${DATASET}/semantic_data/20260301/embeddings/20260301_qwen3_emb_06B_name_categories_desc_ag-rank_20260309_subset_normalized_id.parquet"

OUT_DIR="./mincut_runs/${DATASET}_$(date +%F_%H-%M-%S)"
GRAPH_CACHE="./graph_cache/${DATASET}_adjacent"

DEVICE="cuda:0"

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
  --method recursive_spectral \
  --graph_type adjacent_cooc_prod \
  --recursive_factor 8 \
  --recursive_levels 3 \
  --recursive_epochs 100 \
  --device "${DEVICE}" \
  --output_dir "${OUT_DIR}" \
  --graph_builder_dump_path "${GRAPH_CACHE}" \
  --max_dense_nodes 10000