import nltk
import string
import pandas as pd
from functools import lru_cache
from nltk.stem import PorterStemmer
from utils import word_frequency, freq_to_set


class SentenceProcessingPipeline:
    def __init__(self, tokenizer=str.split, stopwords=set(), stemming=False,
                 remove_punkt=True, case=True, default_stopwords=True):
        self.stopwords = set()
        if default_stopwords:
            self.stopwords = set(nltk.corpus.stopwords.words())
        self.stopwords = self.stopwords.union(stopwords)
        self.tokenizer = tokenizer
        self.punkt = lambda s: s
        if remove_punkt:
            trans = str.maketrans("", "", string.punctuation)
            self.punkt = lambda s: s.translate(trans)
            self.stopwords = set([w.translate(trans) for w in self.stopwords])
        self.stemming = lambda s: s
        if stemming:
            ps = PorterStemmer()
            self.stemming = lambda w: ps.stem(w)
            self.stopwords = set([ps.stem(w) for w in self.stopwords])
        self.case = lambda s: s
        if not case:
            self.case = lambda s: s.lower()
            self.stopwords = set([w.lower() for w in self.stopwords])
    @lru_cache(None)
    def __call__(self, sentence):
        filtered_tokens = []
        sentence = self.punkt(sentence)
        sentence = self.case(sentence)
        tokenized_sent = self.tokenizer(sentence)
        for token in tokenized_sent:
            token = self.stemming(token)
            if token in self.stopwords:
                continue
            filtered_tokens.append(token)
        return tuple(filtered_tokens)
    @classmethod
    def with_df_freq_words(cls, df: pd.DataFrame, top_n: int, names=["name_1", "name_2"], **kwargs):
        tokenizer = str.split
        if "tokenizer" in kwargs:
            tokenizer = kwargs["tokenizer"]
        freq = word_frequency(df, tokenizer=tokenizer, names=names)
        freq = freq_to_set(freq, top_n)
        return cls(stopwords=freq, **kwargs)


def tokenize(dfc: pd.DataFrame, top_n: int, names=["name_1", "name_2"], **kwargs):
    spp = SentenceProcessingPipeline.with_df_freq_words(
        dfc, top_n, names=names, **kwargs
    )
    res = dfc.copy(deep=True)
    for name in names:
        column_name = f"{name}_tokens"
        res[column_name] = res[name].apply(spp)
        res = res[res[column_name].astype(bool) != False]
    return res
