
import pickle

import numpy as np
from scipy import sparse


def iter_adj_list(name):
    with open(name) as sr:
        for line in sr:
            node_ids = list(map(int, line.split()))
            yield node_ids[0], node_ids[1:]


def add_edge(row, col, i, j):
    row.extend([i, j])
    col.extend([j, i])


def load_train(name):
    '''
    Training file is numbered from 0 to n. Not all nodes in the training file have their own row.

    This converts the directed graph to an undirected bipartite one, by replace each vertex with an inbound (odd) and
    outbound (even) vertex.

    Returns a csc matrix, note that indexing rows is the same as indexing columns due to symmetry.
    '''
    row = []
    col = []
    for node, neighbors in iter_adj_list(name):
        for neighbor in neighbors:
            add_edge(row, col, node*2, neighbor*2 + 1)
    for i in range(len(row)):
        add_edge(row, col, i*2, i*2 + 1)
    return sparse.csc_matrix((np.ones(len(row)), (row, col)))


def main():
    with open('data/train_sparse_matrix.pickle', 'wb') as sr:
        pickle.dump(load_train('data/train.txt'), sr)


if __name__ == '__main__':
    main()
