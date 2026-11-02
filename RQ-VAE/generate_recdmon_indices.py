#!/usr/bin/env python3
"""Build RecDMoN semantic IDs for training and new OOV items."""

import argparse
import sys
import yaml
import json
from collections import defaultdict
from pathlib import Path
from utils import yaml_to_argv

import polars as pl


def generate_semantic_ids(
    labels_path: str,
    embedding_table: str,
    all_items_index: str,
) -> dict[str, list[int]]:
    with open(labels_path) as file:
        levels = list(json.load(file)["coarse_to_fine_item_labels"].values())

    items = pl.read_parquet(
        embedding_table,
        columns=["normalized_id", "item_id"],
    )
    all_items_index = pl.read_parquet(
        all_items_index,
        columns=["item_id", "raw_id"],
    )
    oov_items = all_items_index.join(items, how="anti", on="item_id")

    # RecDMoN labels are zero-based. The next code at each level is for OOVs.
    oov_prefix = tuple(max(level) + 2 for level in levels)

    item_prefixes = [
        (item_id, tuple(level[normalized_id] + 1 for level in levels))
        for normalized_id, item_id in items.iter_rows()
    ]
    item_prefixes.extend(
        (item_id, oov_prefix)
        for (item_id, raw_id) in oov_items.iter_rows()
    )

    leaf_counts = defaultdict(int)
    semantic_ids = {}
    for item_id, prefix in sorted(item_prefixes):
        leaf_counts[prefix] += 1
        semantic_ids[item_id] = [*prefix, leaf_counts[prefix]]

    semantic_ids = pl.DataFrame({
        "item_id": list(semantic_ids.keys()),
        "semantic_id": list(semantic_ids.values()),
    })

    semantic_ids = semantic_ids.join(all_items_index, on="item_id").select(
        pl.col("item_id"),
        pl.col("semantic_id"),
        pl.col("raw_id"),
    ).sort("item_id")

    return semantic_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument("--labels_path", required=True)
    parser.add_argument("--embedding_table", required=True)
    parser.add_argument("--all_items_index", required=True)
    parser.add_argument("--output_file", required=True)

    config_args, _ = parser.parse_known_args()
    yaml_argv = []
    if config_args.config:
        with open(config_args.config, encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

        config = config.pop("sid_gen_conf", {})
        yaml_argv = yaml_to_argv(config)

    # Later arguments win, so real CLI arguments override YAML.
    args = parser.parse_args(yaml_argv + sys.argv[1:])
    return args

def main() -> None:
    args = parse_args()
    semantic_ids = generate_semantic_ids(
        args.labels_path,
        args.embedding_table,
        args.all_items_index,
    )

    # output_path = Path(args.output_file)
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    semantic_ids.write_parquet(args.output_file)

    print(f"Saved {len(semantic_ids)} semantic IDs to {args.output_file}")


if __name__ == "__main__":
    main()
