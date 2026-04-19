"""Accuracy / F1 style metrics."""

from __future__ import annotations


def exact_match(prediction: str, gold: str) -> bool:
    return prediction.strip() == gold.strip()


def f1_token_overlap(prediction: str, gold: str) -> float:
    pt = prediction.lower().split()
    gt = gold.lower().split()
    if not pt or not gt:
        return 0.0
    common = set(pt) & set(gt)
    if not common:
        return 0.0
    precision = len(common) / len(set(pt))
    recall = len(common) / len(set(gt))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
