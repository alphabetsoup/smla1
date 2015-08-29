
import csv
import pickle
import os

import matplotlib.pyplot as plt


def main():
    os.chdir(r'/home/linux/unimelb/masters/year3sem2/statistical_machine_learning/project1/project1')
    with open('data/train_python_dict.pickle', 'rb') as sr:
        g = pickle.load(sr)
    with open('data/test-public.txt', 'r') as sr:
        edges = []
        for row in csv.DictReader(sr, delimiter='\t'):
            edges.append((int(row['from']), int(row['to'])))
    plt.hist(list(len(g.in_dict[u]) for u, v in edges), 10, normed=1, histtype='step')

if __name__ == '__main__':
    main()
