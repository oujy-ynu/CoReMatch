"""Retrieval metrics used by the CoReMatch pipeline."""

import numpy as np


def ranking_func(data):
   
    data = np.asarray(data)
    sort_ids = np.argsort(-data)
    rank = np.zeros(len(data), dtype=np.float64)
    for position, item_id in enumerate(sort_ids):
        first_tied_position = position
        while first_tied_position > 0:
            previous_id = sort_ids[first_tied_position - 1]
            if data[previous_id] != data[item_id]:
                break
            first_tied_position -= 1
        rank[item_id] = first_tied_position + 1
    return rank


def calculate_ranking_correlation(rank_corr_function, prediction, target):
    
    prediction = np.asarray(prediction)
    target = np.asarray(target)

    if len(prediction) < 2:
        return np.nan

    r_prediction = ranking_func(prediction)
    r_target = ranking_func(target)
    correlation = rank_corr_function(r_prediction, r_target).correlation
    if correlation is None:
        return np.nan
    return correlation


def top_k_ids(data, k):

    data = np.asarray(data)
    sort_ids = np.argsort(-data)
    if k <= 0 or k >= len(sort_ids):
        raise ValueError(f"k must satisfy 0 < k < {len(sort_ids)}, got {k}.")
    expanded_k = k
    while expanded_k < len(sort_ids):
        current_id = sort_ids[expanded_k - 1]
        next_id = sort_ids[expanded_k]
        if data[current_id] != data[next_id]:
            break
        expanded_k += 1
    return sort_ids[:expanded_k]


def calculate_prec_at_k(k, prediction, target):

    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if len(prediction) <= k:
        return np.nan

    best_k_target = top_k_ids(target, k)
    best_k_prediction = top_k_ids(prediction, k)
    overlap = len(set(best_k_target).intersection(set(best_k_prediction)))
    return min(overlap, k) / k
