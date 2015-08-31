
import pickle
import csv

from project1.data.utils import Graph, iter_adj_list


def load_train(name,testname):
    '''
    Training file is numbered from 0 to n. Not all nodes in the training file have their own row.
    '''
    g = Graph()
    for node, neighbors in iter_adj_list(name):
        for neighbor in neighbors:
            g.add_edge(node, neighbor)
    with open(testname, 'r') as sr:
        for row in csv.DictReader(sr, delimiter='\t'):
            g.exclude_edge(int(row['from']), int(row['to']))
    g.compute_in_deg_dict()
    return g


def main():
    with open('data/train_python_dict.pickle', 'wb') as sr:
        pickle.dump(load_train('data/train.txt','data/test-public.txt'), sr)

if __name__ == '__main__':
    main()
