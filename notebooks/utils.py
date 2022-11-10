import pandas as pd


def word_frequency(df: pd.DataFrame, tokenizer=str.split, names=["name_1", "name_2"]):
    freq = pd.concat(
        [df[n].apply(lambda x: tokenizer(x)).explode() for n in names]
    ).value_counts().to_frame()
    freq.columns = ["Count"]
    return freq


def freq_to_set(freq: pd.DataFrame, top_n = 5):
    return set(freq["Count"].head(top_n).index.to_list())


def clear(df: pd.DataFrame):
    non_ascii = "[^\x00-\x7F]"
    df = df[df["name_1"] != df["name_2"]]
    df = df[df["name_1"].str.lower() != df["name_2"].str.lower()]
    df = df[~(df["name_1"].str.contains(non_ascii) | 
              df["name_2"].str.contains(non_ascii))]
    return df


def get_unique_values(dfct: pd.DataFrame, names=["name_1_tokens", "name_2_tokens"]):
    uniques = set()
    for col_name in names:
        unique = set(dfct[col_name].unique())
        uniques = uniques.union(unique)
    return list(uniques)
