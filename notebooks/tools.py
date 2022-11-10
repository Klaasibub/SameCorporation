import time
import pandas as pd
import matplotlib.pyplot as plt
from spp import tokenize
from utils import get_unique_values
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support
)
from methods import Method
from typing import List


def compute_v2(dfct: pd.DataFrame, method: Method):
    res = dfct.copy(deep=True)
    start = time.perf_counter()
    method.cache(get_unique_values(res))
    res["names_sim"] = res.apply(method, axis=1)
    end = time.perf_counter()
    res = res.dropna()
    dt = end - start
    return res, dt


def validate_v2(res: pd.DataFrame, threshold: int):
    y = lambda val: val > threshold
    y_true = res["is_duplicate"].to_numpy()
    y_pred = res["names_sim"].apply(y).to_numpy()
    a = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred)
    return a, p[1], r[1], f[1]


def benchmark_v2_core(dfc: pd.DataFrame, method: Method, top_n: int, thresholds: List[int], ax=plt, title="Metrics", **kwargs):
    dfct = tokenize(dfc, top_n, **kwargs)
    res, dt = compute_v2(dfct, method)
    a_scores, p_scores, r_scores, f_scores = [], [], [], []
    for threshold in thresholds:
        a, p, r, f = validate_v2(res, threshold)
        a_scores.append(a)
        p_scores.append(p)
        r_scores.append(r)
        f_scores.append(f)
    ax.plot(thresholds, a_scores, label="Accuracy")
    ax.plot(thresholds, p_scores, label="Precision")
    ax.plot(thresholds, r_scores, label="Recall")
    ax.plot(thresholds, f_scores, label="F-score")
    if ax != plt:
        ax.set_title(title)
    else:
        ax.title(title)
    ax.legend()
    max_f_score = max(f_scores)
    max_f_score_index = f_scores.index(max_f_score)
    print(f"Max f-score: {max(f_scores)}")
    print(f"Accuracy   : {a_scores[max_f_score_index]}")
    print(f"Precision  : {p_scores[max_f_score_index]}")
    print(f"Recall     : {r_scores[max_f_score_index]}")
    print(f"Performance: {dt}")


def benchmark_v2(dfc: pd.DataFrame, method: Method, top_n: int, suptitle: str):
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
    kwargs_list = ({}, {"stemming": True, "case": False})
    title_list =  (
        "Case-sensitive without stemming", 
        "Case-insensitive with stemming",
    )
    fig.suptitle(suptitle, fontsize=16)
    for i, (kwargs, title) in enumerate(zip(kwargs_list, title_list)):
        benchmark_v2_core(dfc, method, top_n, thresholds, ax=ax[i], title=title, **kwargs)
