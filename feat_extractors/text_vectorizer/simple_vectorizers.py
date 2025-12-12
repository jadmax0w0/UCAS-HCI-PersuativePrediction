import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame
from typing import List, Optional, Union, Literal, Callable

import sys; sys.path.append(".")
from feat_extractors.base_extractor import BaseTextFeatureExtractor


EXTRA_STOPWORDS = ['ain', 'daren', 'hadn', 'herse', 'himse', 'itse', 'mayn', 'mightn', 'mon', 'mustn', 'myse', 'needn', 'oughtn', 'shan']


def load_stopwords(
        paths: Union[str, List[str]] = ["./feat_extractors/text_vectorizer/en_stopwords.txt"],
        sep: str = '\n',
        append_extra: bool = True,
):
    if isinstance(paths, str):
        paths = [paths]
    
    words = []
    for p in paths:
        with open(p, mode='r', encoding='utf-8') as f:
            cont = f.read()
        words.extend(cont.split(sep))
    
    if append_extra:
        words.extend(EXTRA_STOPWORDS)
    
    words = list(set(words))
    return words


class CountVectorizer(BaseTextFeatureExtractor):
    def __init__(
            self,
            max_feat_count: int = 2000,
            ngram_range: tuple[int] = (1,2),
            stopwords: Optional[list[str]] = None,
            tokenzier: Optional[Callable] = None
    ):
        import sklearn.feature_extraction.text as FE
        super().__init__()

        self.max_feat_count = max_feat_count
        self.stopwords = stopwords
        self.tokenizer = tokenzier

        self.vectorizer = FE.CountVectorizer(
            stop_words=stopwords if stopwords is not None else load_stopwords(),
            ngram_range=ngram_range,
            max_features=max_feat_count,
            tokenizer=tokenzier,
            min_df=10,
        )
    
    def train(self, training_texts: list[str]):
        self.vectorizer.fit(training_texts)
    
    def extract(self, text: Union[str, list[str]]) -> NDArray:
        """
        Returns:
            vectorized texts shaped `[samples_count, feat_count]`
        """
        if isinstance(text, str):
            text = [text]
        return self.vectorizer.transform(text).toarray()


class TfidfVectorizer(BaseTextFeatureExtractor):
    def __init__(
            self,
            max_feat_count: int = 2000,
            ngram_range: tuple[int] = (1,2),
            stopwords: Optional[list[str]] = None,
            tokenzier: Optional[Callable] = None
    ):
        import sklearn.feature_extraction.text as FE
        super().__init__()

        self.max_feat_count = max_feat_count
        self.stopwords = stopwords
        self.tokenizer = tokenzier

        self.vectorizer = FE.TfidfVectorizer(
            tokenizer=tokenzier,
            stop_words=stopwords if stopwords is not None else load_stopwords(),
            ngram_range=ngram_range,
            max_features=max_feat_count,
        )
    
    def train(self, training_texts: list[str]):
        self.vectorizer.fit(training_texts)
    
    def extract(self, text: Union[str, list[str]]) -> NDArray:
        """
        Returns:
            vectorized texts shaped `[samples_count, feat_count]`
        """
        if isinstance(text, str):
            text = [text]
        return self.vectorizer.transform(text).toarray()