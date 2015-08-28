
import csv
import pickle
import random
from contextlib import contextmanager, suppress
import argparse

import numpy as np
from features import *
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
        if g.in_dict[u] and g.out_dict[u] and v not in g.out_dict[u]:
            non_edges.append((u, v))
    yield edges + non_edges, np.hstack([np.ones(n), np.zeros(n)])
    for u, v in edges:
        g.add_edge(u, v)

# What does this method do?? Please use more self-explanatory method names to help us, Steve.
def dev(g, pipeline):
    with gen_classif_data(g, 1000) as (dev_edges, dev_y):
        dev_probs = [0.0] * len(dev_y)
        for p in range(10):
            print("Training bootstrap "+str(p))
            with gen_classif_data(g, 1000) as (train_edges, train_y):
                pipeline.fit(train_edges, train_y)
                temp_probs = pipeline.predict_proba(dev_edges)[:, list(pipeline.classes_).index(1)]
                for i in range(len(temp_probs)):
                    dev_probs[i] += temp_probs[i]
        for i in range(len(dev_probs)):
            dev_probs[i] *= 0.1
        #print('training auc: {}'.format(roc_auc_score(
        #    train_y, pipeline.predict_proba(train_edges)[:, list(pipeline.classes_).index(1)]
        #)))
        #print('dev auc: {}'.format(roc_auc_score(
        #    dev_y, pipeline.predict_proba(dev_edges)[:, list(pipeline.classes_).index(1)]
        #)))
        print('dev auc after 10 bootstraps: {}'.format(roc_auc_score(
            dev_y, dev_probs
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
                ('degrees', Degrees(g)),
                ('common_neighbors', CommonNeighbors(g)),
                ('adamic_adar', AdamicAdar(g)),
                ('katz', Katz(g, 5, 0.5)),
            ])),
            ('logreg', LogisticRegression()),
        ])
        if args.mode == 'dev':
            dev(g, pipeline)
        elif args.mode == 'test':
            test(g, pipeline)

if __name__ == '__main__':
    main()
