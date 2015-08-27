
from queue import deque

import numpy as np
from sklearn.base import BaseEstimator

__all__ = ['Degrees', 'CommonNeighbors', 'AdamicAdar', 'Katz']


class BaseGraphEstimator(BaseEstimator):
    def __init__(self, g):
        self.g = g

    def fit(self, edges, y=None):
        return self


class Degrees(BaseGraphEstimator):
    def transform(self, edges):
        res = []
        for u, v in edges:
            res.append(np.array([
                self.g.vertex(u).in_degree(), self.g.vertex(u).out_degree(),
                self.g.vertex(v).in_degree(), self.g.vertex(v).out_degree()
            ]))
        return np.log(np.vstack(res) + 1)


class CommonNeighbors(BaseGraphEstimator):
    def transform(self, edges):
        res = []
        for u, v in edges:
            u_in = set(self.g.in_dict[u])
            u_out = set(self.g.out_dict[u])
            v_in = set(self.g.in_dict[v])
            v_out = set(self.g.out_dict[v])
            res.append(np.array([len(u_in & v_in), len(u_in & v_out), len(u_out & v_in), len(u_out & v_out)]))
        return np.vstack(res)


class AdamicAdar(BaseGraphEstimator):
    def transform(self, edges):
        res = []
        for u, v in edges:
            u_in = set(self.g.in_dict[u])
            u_out = set(self.g.out_dict[u])
            v_in = set(self.g.in_dict[v])
            v_out = set(self.g.out_dict[v])
            res.append(np.array([
                sum(1/np.log(len(self.g.out_dict[z])) for z in u_in & v_in),
                sum(1/np.log(len(self.g.in_dict[z]) + len(self.g.out_dict[z])) for z in u_in & v_out),
                sum(1/np.log(len(self.g.in_dict[z]) + len(self.g.out_dict[z])) for z in u_out & v_in),
                sum(1/np.log(len(self.g.in_dict[z])) for z in u_out & v_out),
            ]))
        return np.vstack(res)


class Katz(BaseGraphEstimator):
    '''
    Does not search the entire graph due to the high computational cost of even partial matrix inversion.
    '''

    def __init__(self, g, depth, beta):
        super().__init__(g)
        self.depth = depth
        self.beta = beta

    def transform(self, edges):
        res = []
        for (u, v) in edges:
            # bfs search
            score = 0
            q = deque([u])
            cur_depth = 0
            while q and cur_depth < self.depth:
                cur_depth += 1
                node = q.popleft()
                for neighbor in self.g.out_dict[node]:
                    if neighbor == v:
                        score += self.beta**cur_depth
                    q.append(neighbor)
            res.append(score)
        return np.vstack(res)
