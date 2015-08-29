
import argparse
import csv
import pickle
import random
from contextlib import contextmanager

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC

from project1.classifiers import *
from project1.features import *


@contextmanager
def gen_classif_data(g, n):
    # sample some edges and non-edges
    us = set()
    edges = []
    while len(edges) < n:
        u = random.randrange(g.num_vertices)
        if u in us:
            continue
        if not (len(g.out_dict[u]) >= 2 and len(g.in_dict[u]) >= 1):
            continue
        v = random.choice(g.out_dict[u])
        if not len(g.in_dict[v]) >= 2:
            continue
        if len(g.out_dict[v]) == 0:
            continue
        # remove edges since the edges to predict are not supposed to be in the training graph
        g.remove_edge(u, v)
        edges.append((u, v))
        us.add(u)
    non_edges = []
    while len(non_edges) < n:
        u = random.randrange(g.num_vertices)
        if u in us:
            continue
        v = random.randrange(g.num_vertices)
        if not (u != v and v not in g.out_dict[u]):
            continue
        if not (len(g.out_dict[u]) >= 1 and len(g.in_dict[u]) >= 1 and len(g.in_dict[v]) >= 1):
            continue
        non_edges.append((u, v))
        us.add(u)
    yield np.array(edges + non_edges), np.hstack([np.ones(n), np.zeros(n)])
    for u, v in edges:
        g.add_edge(u, v)


def score(name, y, probs, classes_):
    print('{} auc:\t\t{:.4f}'.format(name, roc_auc_score(y, probs[:, list(classes_).index(1)])))
    print('{} accuracy:\t{:.4f}'.format(name, accuracy_score(y, classes_[np.argmax(probs, axis=1)])))


def dev(g, estimator):
    # generate results for the dev set
    with gen_classif_data(g, 1000) as (dev_edges, dev_y):
        with gen_classif_data(g, 1000) as (train_edges, train_y):
            estimator.fit((g, train_edges), train_y)
            score('train', train_y, estimator.predict_proba((g, train_edges)), estimator.classes_)
        score('dev', dev_y, estimator.predict_proba((g, dev_edges)), estimator.classes_)


def test(g, estimator):
    # generate results for the test set
    with gen_classif_data(g, 1000) as (train_edges, train_y):
        estimator.fit((g, train_edges), train_y)
    with open('data/test-public.txt', 'r') as sr:
        edges = []
        for row in csv.DictReader(sr, delimiter='\t'):
            edges.append((int(row['from']), int(row['to'])))
        probs = estimator.predict_proba((g, edges))
    with open('data/test-public-predict.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Id', 'Prediction'])
        writer.writeheader()
        for i, prob in enumerate(probs):
            writer.writerow({'Id': i + 1, 'Prediction': prob[list(estimator.classes_).index(1)]})


def main():
    arg_parser = argparse.ArgumentParser()
    sub_parsers = arg_parser.add_subparsers(dest='mode')
    sub_parsers.required = True
    sub_parsers.add_parser('dev')
    sub_parsers.add_parser('test')
    args = arg_parser.parse_args()
    with open('data/train_python_dict.pickle', 'rb') as sr:
        g = pickle.load(sr)
        pipeline = Pipeline([
            ('features', FeatureUnion([
                ('degrees', Degrees()),
                ('common_neighbors', CommonNeighbors()),
                ('adamic_adar', AdamicAdar()),
                ('katz', Katz(5, 0.5)),
            ])),
            ('logreg', LogisticRegression()),
            # ('svm', SVC(kernel='rbf', probability=True))
        ])
        # pipeline = GraphBaggingClassifier(pipeline, 10)
        if args.mode == 'dev':
            dev(g, pipeline)
        elif args.mode == 'test':
            test(g, pipeline)

if __name__ == '__main__':
    main()
