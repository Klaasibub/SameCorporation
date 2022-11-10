import pylev
from numpy import dot
from numpy.linalg import norm
from functools import lru_cache
from typing import List


class Base:
    def __init__(self, names=["name_1_tokens", "name_2_tokens"]):
        self.names = names
    def calc(self, x_words: List[str], y_words: List[str]):
        raise NotImplementedError()
    def cache(self, tokens: List[str]):
        pass
    def __call__(self, row):
        name_1, name_2 = [row[n] for n in self.names]
        return self.calc(name_1, name_2)


class Rule(Base):
    def __init__(self):
        super().__init__()
    def calc(self, x_words: List[str], y_words: List[str]):
        counter = 0
        for word in x_words:
            if word in y_words:
                counter += 1
        return counter / max(len(x_words), len(y_words))


class Levenshtein(Base):
    def __init__(self):
        super().__init__()
        self.distance = lru_cache()(pylev.levenshtein)
    def calc(self, x_words: List[str], y_words: List[str]):
        distance = self.distance(x_words, y_words)
        return 1 - distance/max(len(x_words), len(y_words))


class Fasttext(Base):
    def __init__(self, model):
        super().__init__()
        self.model = model
    @lru_cache(None)
    def s2v(self, tokens: List[str]):
        return self.model.get_sentence_vector(tokens)
    def calc(self, x_words: List[str], y_words: List[str]):
        v1, v2 = self.s2v(x_words), self.s2v(y_words)
        return dot(v1, v2) / max((norm(v1) * norm(v2)), 1e-8)


class Transformer(Base):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def cache(self, tokens: List[str]):
        names = [" ".join(token) for token in tokens]
        encoded = self.model.encode(names)
        self.t2v = {token:vec for token, vec in zip(tokens, encoded)}
    def calc(self, x_tokens: List[str], y_tokens: List[str]):
        emb1 = self.t2v[x_tokens]
        emb2 = self.t2v[y_tokens]
        return dot(emb1, emb2)
    def encode(self, tokens: List[str]):
        sent = " ".join(tokens)
        return self.model.encode(sent)
