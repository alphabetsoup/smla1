
import numpy as np
from sklearn.base import BaseEstimator

__all__ = ['CommonNeighbors', 'Degrees']


class BaseGraphEstimator(BaseEstimator):
    def __init__(self, g):
        self.g = g

    def fit(self, edges, y=None):
        return self


class CommonNeighbors(BaseGraphEstimator):
    def transform(self, edges):
        res = []
        for u, v in edges:
            res.append(len(set(self.g.vertex(u).out_neighbours()) & set(self.g.vertex(v).out_neighbours())))
        return np.vstack(res)


class Degrees(BaseGraphEstimator):
    def transform(self, edges):
        res = []
        for u, v in edges:
            res.append(np.array([
                self.g.vertex(u).in_degree(), self.g.vertex(u).out_degree(),
                self.g.vertex(v).in_degree(), self.g.vertex(v).out_degree()
            ]))
        return np.log(np.vstack(res) + 1)
