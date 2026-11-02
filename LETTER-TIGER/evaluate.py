import math
import random


def get_topk_results(
    predictions,
    scores,
    targets,
    k,
    all_items=None,
    collision_counts=None,
    collision_metric_mode="none",
    collision_sample_threshold=1,
    rng=None,
):
    results = []
    B = len(targets)
    # predictions = [_.split("Response:")[-1] for _ in predictions]
    predictions = [_.strip().replace(" ","") for _ in predictions]
    # print(predictions)##################
    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

    # print(scores)
    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]

        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        # print(pairs)
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        target_item = targets[b]
        if collision_metric_mode == "sampled_unrolled":
            one_results = _sampled_unrolled_relevance(
                sorted_pairs=sorted_pairs,
                target_item=target_item,
                collision_counts=collision_counts,
                rng=rng,
            )
        else:
            one_results = []
            for sorted_pred in sorted_pairs:
                pred_item = sorted_pred[0]
                if pred_item == target_item:
                    one_results.append(
                        _collision_aware_relevance(
                            pred_item=pred_item,
                            collision_counts=collision_counts,
                            collision_metric_mode=collision_metric_mode,
                            collision_sample_threshold=collision_sample_threshold,
                            rng=rng,
                        )
                    )
                else:
                    one_results.append(0.0)

        results.append(one_results)

    return results


def _sampled_unrolled_relevance(
    sorted_pairs,
    target_item,
    collision_counts=None,
    rng=None,
):
    if rng is None:
        rng = random.Random()

    one_results = []
    for sorted_pred in sorted_pairs:
        pred_item = sorted_pred[0]
        bucket_size = 1 if collision_counts is None else collision_counts.get(pred_item, 1)
        bucket = [0.0] * bucket_size

        if pred_item == target_item:
            pos = rng.randrange(bucket_size)
            bucket[pos] = 1.0

        rng.shuffle(bucket)
        one_results.extend(bucket)

    return one_results


def _collision_aware_relevance(
    pred_item,
    collision_counts=None,
    collision_metric_mode="none",
    collision_sample_threshold=1,
    rng=None,
):
    if collision_counts is None or collision_metric_mode == "none":
        return 1.0

    collision_count = collision_counts.get(pred_item, 1)
    if collision_count <= 1:
        return 1.0

    if collision_metric_mode == "sampled":
        if rng is None:
            rng = random.Random()
        return 1.0 if rng.random() < (1.0 / collision_count) else 0.0

    raise ValueError(f"Unknown collision metric mode: {collision_metric_mode}")


def get_topk_ranking_results(predictions, targets, k, all_items=None):
    results = []
    B = len(targets)

    for b in range(B):
        batch_seqs = predictions[b]
        target_item = targets[b]
        one_results = []
        for sorted_pred in predictions:
            if sorted_pred == target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)

    return results


def get_metrics_results(topk_results, metrics):
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError

    return res


def ndcg_k(topk_results, k):
    """
    Since we apply leave-one-out, each user only have one ground truth item, so the idcg would be 1.0
    """
    ndcg = 0.0
    for row in topk_results:
        res = row[:k]
        one_ndcg = 0.0
        for i in range(len(res)):
            one_ndcg += res[i] / math.log(i + 2, 2)
        ndcg += one_ndcg
    return ndcg


def hit_k(topk_results, k):
    hit = 0.0
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:
            hit += 1
    return hit
