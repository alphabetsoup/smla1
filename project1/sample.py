
import random
import numpy as np

__all__ = ['SampleClassifData']


class SampleClassifData:
    def __init__(self, g):
        self.g = g
        # store as a list as python has no structure to randomly sample from sets efficiently
        self.all_edges = []
        for u, vs in g.out_dict.items():
            self.all_edges.extend((u, v) for v in vs)

    def sample(self, n):
        # unique u, this guarantees that edges and non-edges do not overlap
        us = set()
        # sample edges
        edges = set()
        random.shuffle(self.all_edges)
        all_edges_iter = iter(self.all_edges)
        while len(edges) < n:
            u, v = next(all_edges_iter)
            if u in us:
                continue
            if not (len(self.g.out_dict[u]) >= 2 and len(self.g.in_dict[u]) >= 1 and len(self.g.in_dict[v]) >= 2):
                continue
            # remove edges for subsequent sampling, early so degrees are correct
            self.g.remove_edge(u, v)
            edges.add((u, v))
            us.add(u)
        # remove edges for subsequent sampling
        all_edges = []
        for edge in self.all_edges:
            if edge not in edges:
                all_edges.append(edge)
        self.all_edges = all_edges
        # sample non-edges
        non_edges = set()
        while len(non_edges) < n:
            u = random.randrange(self.g.num_vertices)
            if u in us:
                continue
            v = random.randrange(self.g.num_vertices)
            if not (u != v and v not in self.g.out_dict[u]):
                continue
            if not (len(self.g.out_dict[u]) >= 1 and len(self.g.in_dict[u]) >= 1 and len(self.g.in_dict[v]) >= 1):
                continue
            non_edges.add((u, v))
            us.add(u)
        return np.array(list(edges) + list(non_edges)), np.hstack([np.ones(n), np.zeros(n)])
