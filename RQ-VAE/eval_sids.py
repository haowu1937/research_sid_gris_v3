import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sklearn.metrics import adjusted_rand_score


# --------------------------------------------------
# Original metrics helpers from eval_sids
# --------------------------------------------------

def _utilization(codes: np.ndarray, codebook_size: int) -> float:
    used = np.unique(codes)
    return len(used) / codebook_size


def _codebook_entropy_and_perplexity(codes: np.ndarray, codebook_size: int) -> Tuple[float, float]:
    counts = np.bincount(codes, minlength=codebook_size).astype(np.float64)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -(probs * np.log2(probs)).sum()
    perplexity = 2 ** entropy
    return entropy, perplexity


def _gini_coefficient(codes: np.ndarray, codebook_size: int) -> float:
    counts = np.bincount(codes, minlength=codebook_size).astype(np.float64)
    if counts.sum() == 0:
        return 0.0
    counts = np.sort(counts)
    n = len(counts)
    cumx = np.cumsum(counts)
    gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return gini


def _collision_rate(sids: np.ndarray) -> float:
    tuples = [tuple(row) for row in sids]
    from collections import Counter
    freq = Counter(tuples)
    colliding = sum(cnt for cnt in freq.values() if cnt > 1)
    return colliding / len(tuples)


def _prefix_collision_rate(sids: np.ndarray, depth: int) -> float:
    return _collision_rate(sids[:, :depth])


def evaluate_sids(sids: np.ndarray, codebook_sizes: List[int]) -> None:
    n_items, n_codebooks = sids.shape
    assert len(codebook_sizes) == n_codebooks, (
        f"Mismatch between sids depth ({n_codebooks}) and codebook_sizes ({len(codebook_sizes)})."
    )

    sep = "─" * 70
    print(f"\n{sep}")
    print(f"Semantic ID Evaluation  |  {n_items} items  |  {n_codebooks} codebooks")
    print(sep)

    # 1. Per-codebook metrics
    print("\n[ Per-Codebook Metrics ]")
    print(f"{'CB':>4}  {'Size':>6}  {'Util%':>7}  {'Entropy':>9}  {'Perplx':>9}  {'PPL/Size':>9}  {'Gini':>7}")
    for cb in range(n_codebooks):
        codes = sids[:, cb]
        cb_size = codebook_sizes[cb]
        util = _utilization(codes, cb_size)
        h, ppl = _codebook_entropy_and_perplexity(codes, cb_size)
        gini = _gini_coefficient(codes, cb_size)
        print(f"{cb:>4}  {cb_size:>6}  {util*100:>6.1f}%  {h:>9.4f}  {ppl:>9.2f}  {ppl/cb_size:>9.4f}  {gini:>7.4f}")

    # 2. Prefix-level collision rates
    print("\n[ Prefix-Level Collision Rates ]")
    print(f"{'Prefix depth':>14}  {'Collision%':>12}  {'Unique IDs':>12}  {'Max collisions':>16}")

    for depth in range(1, n_codebooks + 1):
        prefix_sids = sids[:, :depth]
        tuples = [tuple(row) for row in prefix_sids]
        from collections import Counter
        freq = Counter(tuples)
        n_unique = len(freq)
        max_coll = max(freq.values())
        coll_rate = _prefix_collision_rate(sids, depth)
        print(f"{depth:>14}  {coll_rate*100:>11.2f}%  {n_unique:>12}  {max_coll:>16}")

    full_unique = len(set(map(tuple, sids)))
    print(f"\nFull SID unique ratio : {full_unique}/{n_items} ({100 * full_unique / n_items:.2f}%)")
    print(f"Full SID collision rate: {_collision_rate(sids) * 100:.2f}%")

    # 3. Pairwise mutual information
    print("\n[ Pairwise Mutual Information Between Codebooks ]")
    print(f"{'CB i':>6}  {'CB j':>6}  {'MI (bits)':>12}  {'Normalized MI':>14}")

    def mutual_information(x, y, sx, sy):
        joint = np.zeros((sx, sy), dtype=np.float64)
        for xi, yi in zip(x, y):
            joint[xi, yi] += 1
        joint /= joint.sum()
        px = joint.sum(axis=1, keepdims=True) + 1e-12
        py = joint.sum(axis=0, keepdims=True) + 1e-12
        pxy = joint + 1e-12
        mi = (pxy * np.log2(pxy / (px * py))).sum()
        hx = -(px * np.log2(px)).sum()
        hy = -(py * np.log2(py)).sum()
        nmi = mi / (0.5 * (hx + hy) + 1e-12)
        return mi, nmi

    for i in range(n_codebooks):
        for j in range(i + 1, n_codebooks):
            mi, nmi = mutual_information(
                sids[:, i], sids[:, j], codebook_sizes[i], codebook_sizes[j]
            )
            print(f"{i:>6}  {j:>6}  {mi:>12.4f}  {nmi:>14.4f}")

    # 4. Prefix-tree balance
    print("\n[ Prefix Tree Balance ]")
    print(f"{'Depth':>6}  {'Nodes':>8}  {'Mean children':>15}  {'Min':>6}  {'Max':>6}  {'Std':>8}")

    from collections import defaultdict
    for depth in range(1, n_codebooks):
        parent_to_children = defaultdict(set)
        for row in sids:
            parent = tuple(row[:depth])
            child = tuple(row[:depth + 1])
            parent_to_children[parent].add(child)
        branching = np.array([len(v) for v in parent_to_children.values()], dtype=np.float64)
        print(f"{depth:>6}  {len(branching):>8}  {branching.mean():>15.2f}  {branching.min():>6.0f}  {branching.max():>6.0f}  {branching.std():>8.2f}")

    # 5. Adjacent-codebook ARI
    print("\n[ Clustering Consistency (ARI between adjacent codebooks) ]")
    print(f"{'CB i → CB j':>14}  {'ARI':>8}")
    for i in range(n_codebooks - 1):
        ari = adjusted_rand_score(sids[:, i], sids[:, i + 1])
        print(f"{f'{i} -> {i+1}':>14}  {ari:>8.4f}")

    # 6. Health summary
    avg_util = np.mean([_utilization(sids[:, cb], codebook_sizes[cb]) for cb in range(n_codebooks)])
    avg_ppl_ratio = np.mean([
        _codebook_entropy_and_perplexity(sids[:, cb], codebook_sizes[cb])[1] / codebook_sizes[cb]
        for cb in range(n_codebooks)
    ])
    full_coll = _collision_rate(sids)

    def _flag(val, good_thresh, bad_thresh, higher_is_better=True):
        if higher_is_better:
            return "✅" if val >= good_thresh else ("⚠️" if val >= bad_thresh else "❌")
        return "✅" if val <= good_thresh else ("⚠️" if val <= bad_thresh else "❌")

    print("\n[ Health Summary ]")
    print(f"Avg codebook utilization : {avg_util * 100:.1f}%  {_flag(avg_util, 0.7, 0.4)}")
    print(f"Avg perplexity / CB size : {avg_ppl_ratio:.4f}  {_flag(avg_ppl_ratio, 0.7, 0.4)}")
    print(f"Full SID collision rate  : {full_coll * 100:.2f}%  {_flag(full_coll, 0.05, 0.20, higher_is_better=False)}")
    print(sep)


# --------------------------------------------------
# Adaptation layer for OUR index JSON format
# --------------------------------------------------

TOKEN_RE = re.compile(r"<([a-zA-Z]+)_(\d+)>")


def parse_sid_token(token: str) -> int:
    """
    "<a_12>" -> 12
    "<b_0>"  -> 0
    """
    m = TOKEN_RE.fullmatch(token.strip())
    if not m:
        raise ValueError(f"Unrecognized SID token format: {token}")
    return int(m.group(2))


def load_our_index_json(index_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Loads JSON of form:
      {
        "0": ["<a_12>", "<b_44>", "<c_3>", "<d_91>"],
        "1": ...
      }

    Returns
    -------
    sids : np.ndarray [n_items, n_codebooks]
    item_ids : list[str]
        Original item ids in sorted order.
    """
    with open(index_path, "r") as f:
        raw = json.load(f)

    # keys are item ids serialized as strings; sort numerically when possible
    def sort_key(x: str):
        try:
            return (0, int(x))
        except ValueError:
            return (1, x)

    item_ids = sorted(raw.keys(), key=sort_key)

    rows = []
    expected_depth = None

    for item_id in item_ids:
        tokens = raw[item_id]

        if not isinstance(tokens, list) or len(tokens) == 0:
            raise ValueError(f"Item {item_id} has invalid SID entry: {tokens}")

        if expected_depth is None:
            expected_depth = len(tokens)
        elif len(tokens) != expected_depth:
            raise ValueError(
                f"Inconsistent SID depth for item {item_id}: "
                f"got {len(tokens)}, expected {expected_depth}"
            )

        rows.append([parse_sid_token(tok) for tok in tokens])

    sids = np.asarray(rows, dtype=np.int64)
    return sids, item_ids


def infer_codebook_sizes(sids: np.ndarray) -> List[int]:
    """
    Conservative inference: size = max code observed + 1 for each codebook.
    Good enough when explicit sizes are not available.
    """
    return [(int(sids[:, i].max()) + 1) for i in range(sids.shape[1])]


def parse_codebook_sizes(arg: str | None, n_codebooks: int) -> List[int] | None:
    if arg is None:
        return None
    vals = [int(x.strip()) for x in arg.split(",") if x.strip()]
    if len(vals) != n_codebooks:
        raise ValueError(
            f"--codebook-sizes has {len(vals)} values, but SIDs have {n_codebooks} codebooks."
        )
    return vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-file", type=str, required=True, help="Path to our generated index JSON")
    parser.add_argument(
        "--codebook-sizes",
        type=str,
        default=None,
        help="Comma-separated sizes, e.g. 256,256,256,256. "
             "If omitted, inferred as max_token+1 per codebook.",
    )
    args = parser.parse_args()

    sids, item_ids = load_our_index_json(args.index_file)

    codebook_sizes = parse_codebook_sizes(args.codebook_sizes, sids.shape[1])
    if codebook_sizes is None:
        codebook_sizes = infer_codebook_sizes(sids)
        print(f"[info] Inferred codebook sizes: {codebook_sizes}")

    print(f"[info] Loaded {len(item_ids)} items from {args.index_file}")
    print(f"[info] SID shape: {sids.shape}")

    evaluate_sids(sids, codebook_sizes)


if __name__ == "__main__":
    main()