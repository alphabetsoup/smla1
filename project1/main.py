
import csv
import pickle
import random
from contextlib import contextmanager, suppress

import numpy as np
from project1.features import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline


@contextmanager
def gen_classif_data(g, n):
    # sample some edges and non-edges
    edges = []
    while len(edges) < n:
        with suppress(IndexError):
            u = random.randrange(g.num_vertices)
            v = random.choice(g.out_dict[u])
            # remove edges since the edges to predict are not supposed to be in the training graph
            g.remove_edge(u, v)
            edges.append((u, v))
    non_edges = []
    while len(non_edges) < n:
        u = random.randrange(g.num_vertices)
        v = random.randrange(g.num_vertices)
        if v not in g.out_dict[u]:
            non_edges.append((u, v))
    yield edges + non_edges, np.hstack([np.ones(n), np.zeros(n)])
    for u, v in edges:
        g.add_edge(u, v)


def dev(g, pipeline):
    with gen_classif_data(g, 1000) as (dev_edges, dev_y):
        with gen_classif_data(g, 1000) as (train_edges, train_y):
            pipeline.fit(train_edges, train_y)
        print('training auc: {}'.format(roc_auc_score(
            train_y, pipeline.predict_proba(train_edges)[:, list(pipeline.classes_).index(1)]
        )))
        print('dev auc: {}'.format(roc_auc_score(
            dev_y, pipeline.predict_proba(dev_edges)[:, list(pipeline.classes_).index(1)]
        )))


def test(g, pipeline):
    with gen_classif_data(g, 1000) as (train_edges, train_y):
        pipeline.fit(train_edges, train_y)
    with open('data/test-public.txt', 'r') as sr:
        edges = []
        for row in csv.DictReader(sr, delimiter='\t'):
            edges.append((int(row['from']), int(row['to'])))
        probs = pipeline.predict_proba(edges)
    with open('data/test-public-predict.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Id', 'Prediction'])
        writer.writeheader()
        for i, prob in enumerate(probs):
            writer.writerow({'Id': i + 1, 'Prediction': prob[list(pipeline.classes_).index(1)]})


def main():
    with open('data/train_python_dict.pickle', 'rb') as sr:
        g = pickle.load(sr)
        pipeline = Pipeline([
            ('features', FeatureUnion([
                # ('degrees', Degrees(g)),
                ('common_neighbors', CommonNeighbors(g)),
                ('adamic_adar', AdamicAdar(g)),
                ('katz', Katz(g, 5, 0.5)),
            ])),
            ('logreg', LogisticRegression()),
        ])
        dev(g, pipeline)
        # test(g, pipeline)

if __name__ == '__main__':
    main()
