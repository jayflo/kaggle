
from sklearn.base import BaseEstimator, TransformerMixin


class IntTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.__map = {}

    def fit(self, X, y=None):
        self.__map = dict(list(map(
            lambda tup: tup[::-1],
            enumerate(X.unique().tolist())
        )))

        return self

    def transform(self, X, y=None, col=None):
        tmp = X

        if col is not None:
            tmp = X[col]

        for k, v in self.__map.items():
            tmp[X == k] = v

        return tmp

    def inverse_transform(self, X, y=None):
        for k, v in self.__map.items():
            X[X == v] = k
