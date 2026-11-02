#!/bin/bash

set -euo pipefail

# =========================
# CONFIG
# =========================
DATASET="Instruments"

INTERACTIONS_PATH="data/${DATASET}/${DATASET}.inter.json"
EMB_PATH="data/${DATASET}/${DATASET}.emb-qwen3-td.npy"

OUT_DIR="./mincut_runs/${DATASET}_$(date +%F_%H-%M-%S)"
GRAPH_CACHE="./graph_cache/${DATASET}_adjacent"

DEVICE="cuda:0"

# =========================
# RUN
# =========================
python ./RQ-VAE/hierarchical_mincut_clustering.py \
  --interactions_path ${INTERACTIONS_PATH} \
  --embeddings_path ${EMB_PATH} \
  --graph_type adjacent_cooc \
  --recursive_factor 8 \
  --recursive_levels 3 \
  --recursive_epochs 100 \
  --device ${DEVICE} \
  --output_dir ${OUT_DIR} \
  --graph_builder_dump_path ${GRAPH_CACHE}\
  --max_dense_nodes 10000

echo "Done. Results in ${OUT_DIR}"
