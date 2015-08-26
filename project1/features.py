
import numpy as np
from sklearn.base import BaseEstimator

__all__ = ['CommonNeighbors']


class CommonNeighbors(BaseEstimator):
    def __init__(self, g):
        self.g = g

    def fit(self, edges, y=None):
        return self

    def transform(self, edges):
        res = []
        for u, v in edges:
            res.append(len(set(self.g.vertex(u).out_neighbours()) & set(self.g.vertex(v).out_neighbours())))
        return np.vstack(res)
