
import pickle

from project1.data.utils import Graph, iter_adj_list
from project1.sample import SampleClassifData


def load_train(name):
    '''
    Training file is numbered from 0 to n. Not all nodes in the training file have their own row.
    '''
    g = Graph()
    for node, neighbors in iter_adj_list(name):
        for neighbor in neighbors:
            g.add_edge(node, neighbor)
    return g


def main():
    g = load_train('data/train.txt')
    with open('data/train_python_dict.pickle', 'wb') as sr:
        pickle.dump(g, sr)
    sampler = SampleClassifData(g)
    with open('data/dev_1000.pickle', 'wb') as sr:
        pickle.dump(sampler.sample(1000), sr)
    with open('data/train_1000.pickle', 'wb') as sr:
        pickle.dump(sampler.sample(1000), sr)

if __name__ == '__main__':
    main()
