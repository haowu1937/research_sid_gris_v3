#!/bin/bash

set -euo pipefail


DATASET="/home/ma-user/work/dataset"
# labels_path="mincut_runs/dataset_2026-08-25_19-27-45/graph_hierarchy_labels.json"
# EMB_PATH="${DATASET}/semantic_data/20260301/embeddings/20260301_qwen3_emb_06B_name_categories_desc_sd_20260728_2.parquet"
labels_path="mincut_runs/dmon_2026-09-01_22-18-06/graph_hierarchy_labels.json"
EMB_PATH="${DATASET}/semantic_data/20260301/embeddings/20260301_qwen3_emb_06B_name_categories_desc_sd-rank_20260728.parquet"
all_items_index="trash/sd_rank_all_items.parquet"

python RQ-VAE/generate_recdmon_indices.py \
  --labels_path $labels_path \
  --embedding_table $EMB_PATH \
  --all_items_index ${all_items_index} \
  --output_file trash/dmon_semantic_ids.parquet # has to be parquet file