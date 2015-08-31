
import numpy as np
from sklearn.base import BaseEstimator
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

__all__ = ['Degrees', 'CommonNeighbors', 'AdamicAdar', 'Katz']


class BaseGraphEstimator(BaseEstimator):
    def fit(self, X, y=None):
        return self


class Degrees(BaseGraphEstimator):
    @staticmethod
    def transform(X):
        g, g_matrix, edges = X
        res = []
        for u, v in edges:
            res.append(np.array([
                len(g.in_dict[u]), len(g.out_dict[u]),
                len(g.in_dict[v]), len(g.out_dict[v]),
            ]))
        return np.log(np.vstack(res) + 1)


class CommonNeighbors(BaseGraphEstimator):
    @staticmethod
    def transform(X):
        g, g_matrix, edges = X
        res = []
        for u, v in edges:
            u_in = set(g.in_dict[u])
            u_out = set(g.out_dict[u])
            v_in = set(g.in_dict[v])
            v_out = set(g.out_dict[v])
            res.append(np.array([len(u_in & v_in), len(u_in & v_out), len(u_out & v_in), len(u_out & v_out)]))
        return np.vstack(res)


class AdamicAdar(BaseGraphEstimator):
    @staticmethod
    def transform(X):
        g, g_matrix, edges = X
        res = []
        for u, v in edges:
            u_in = set(g.in_dict[u])
            u_out = set(g.out_dict[u])
            v_in = set(g.in_dict[v])
            v_out = set(g.out_dict[v])
            res.append(np.array([
                sum(1/np.log(len(g.out_dict[z])) for z in u_in & v_in),
                sum(1/np.log(len(g.in_dict[z]) + len(g.out_dict[z])) for z in u_in & v_out),
                sum(1/np.log(len(g.in_dict[z]) + len(g.out_dict[z])) for z in u_out & v_in),
                sum(1/np.log(len(g.in_dict[z])) for z in u_out & v_out),
            ]))
        return np.vstack(res)


class Katz(BaseGraphEstimator):
    '''
    Does not search the entire graph due to the high computational cost of even partial matrix inversion.
    '''

    def __init__(self, beta):
        self.beta = beta

    def transform(self, X):
        g, g_matrix, edges = X
        n = g.num_vertices
        res = []
        for (u, v) in edges:
            b = np.zeros(n)
            b[v] = 1
            scores = spsolve(csr_matrix((np.ones(n), (np.arange(0, n), np.arange(0, n)))) - self.beta*g_matrix, b)
            res.append(scores[u])
        return np.vstack(res)
