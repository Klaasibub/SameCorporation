import numpy as np
import pandas as pd
from utils import get_unique_values


def get_sims(dfn: pd.DataFrame, s2v, name):
    vector = s2v[name]
    sims = dfn["vector"].apply(lambda x: np.dot(x, vector))
    return sims


def get_relatives(dfc: pd.DataFrame, name):
    dfcd = dfc[dfc["is_duplicate"] == 1]
    left = dfcd[dfcd["name_1"] == name]
    right = dfcd[dfcd["name_2"] == name]
    right = right.rename(columns={"name_1": "name_2", "name_2": "name_1"})
    relatives = pd.merge(left, right, on=["name_1", "name_2"], how="outer")
    relatives = set(relatives["name_2"].to_list())
    return relatives


def precision_at_k(sims, relatives, k=5):
    top_k_sims = sims.nlargest(n=k)
    top_k_names = top_k_sims.index.to_list()
    return sum([int(name in relatives) for name in top_k_names]) / k


def average_precision_at_k(sims, rels, k=5):
    vs, precisions = [], []
    top_k_names = sims.nlargest(n=k).index.to_list()
    for k_i in range(1, k + 1):
        pr = precision_at_k(sims, rels, k=k_i)
        v = int(top_k_names[k_i - 1] in rels)
        precisions.append(pr * v)
        vs.append(v)
    if sum(vs) == 0: return 0
    return sum(precisions) / sum(vs)


def mean_average_precision_at_k(dfc, dfn, k=5):
    aprs = []
    names_unique = get_unique_values(dfc, names=["name_1", "name_2"])
    for i, name in enumerate(names_unique):
        print(f"\r{i+1}/{len(names_unique)}", end="")
        sims = get_sims(dfn, name)
        relatives = get_relatives(dfc, name)
        apr = average_precision_at_k(sims, relatives, k=5)
        aprs.append(apr)
    print()
    return sum(aprs) / len(names_unique)
