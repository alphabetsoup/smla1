
import numpy as np
from sklearn.base import BaseEstimator

__all__ = ['CommonNeighbors', 'AdamicAdar', 'Degrees']


class BaseGraphEstimator(BaseEstimator):
    def __init__(self, g):
        self.g = g

    def fit(self, edges, y=None):
        return self


class CommonNeighbors(BaseGraphEstimator):
    def transform(self, edges):
        res = []
        for u, v in edges:
            u_in = set(self.g.vertex(u).in_neighbours())
            u_out = set(self.g.vertex(u).out_neighbours())
            v_in = set(self.g.vertex(v).in_neighbours())
            v_out = set(self.g.vertex(v).out_neighbours())
            res.append(np.array([len(u_in & v_in), len(u_in & v_out), len(u_out & v_in), len(u_out & v_out)]))
        return np.vstack(res)


class AdamicAdar(BaseGraphEstimator):
    def transform(self, edges):
        res = []
        for u, v in edges:
            u_in = set(self.g.vertex(u).in_neighbours())
            u_out = set(self.g.vertex(u).out_neighbours())
            v_in = set(self.g.vertex(v).in_neighbours())
            v_out = set(self.g.vertex(v).out_neighbours())
            res.append(np.array([
                sum(1/np.log(z.out_degree()) for z in u_in & v_in),
                sum(1/np.log(z.in_degree() + z.out_degree()) for z in u_in & v_out),
                sum(1/np.log(z.in_degree() + z.out_degree()) for z in u_out & v_in),
                sum(1/np.log(z.in_degree()) for z in u_out & v_out),
            ]))
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
