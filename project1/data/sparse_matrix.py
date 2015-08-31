
import pickle

import numpy as np
from project1.data.utils import iter_adj_list
from scipy import sparse


def add_edge(row, col, i, j):
    row.append(i)
    col.append(j)


def load_train(name):
    '''
    Training file is numbered from 0 to n. Not all nodes in the training file have their own row.

    Returns a csr matrix.
    '''
    row = []
    col = []
    for node, neighbors in iter_adj_list(name):
        for neighbor in neighbors:
            add_edge(row, col, node, neighbor)
    n = max(max(row), max(col)) + 1
    return sparse.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))


def main():
    with open('data/train_sparse_matrix.pickle', 'wb') as sr:
        pickle.dump(load_train('data/train.txt'), sr)

if __name__ == '__main__':
    main()
