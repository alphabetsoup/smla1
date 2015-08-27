
import csv
import pickle
import random
from contextlib import suppress

import numpy as np
from project1.features import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline


def main():
    with open('data/train_python_dict.pickle', 'rb') as sr:
        g = pickle.load(sr)
        pipeline = Pipeline([
            ('features', FeatureUnion([
                # ('degrees', Degrees(g)),
                ('common_neighbors', CommonNeighbors(g)),
                ('adamic_adar', AdamicAdar(g)),
                ('katz', Katz(g, 5, 0.25)),
            ])),
            ('logreg', LogisticRegression()),
        ])
        # sample some edges and non-edges
        n = 1000
        edges = []
        while len(edges) < n:
            with suppress(IndexError):
                u = random.randrange(g.num_vertices)
                v = random.choice(g.out_dict[u])
                edges.append((u, v))
        non_edges = []
        while len(non_edges) < n:
            u = random.randrange(g.num_vertices)
            v = random.randrange(g.num_vertices)
            if v not in g.out_dict[u]:
                non_edges.append((u, v))
        pipeline.fit(edges + non_edges, np.hstack([np.ones(n), np.zeros(n)]))

    # training scores
    # print('training auc: {}'.format(roc_auc_score(
    #     np.hstack([np.ones(n), np.zeros(n)]),
    #     pipeline.predict_proba(edges + non_edges)[:, list(pipeline.classes_).index(1)]
    # )))

    # predict
    with open('data/test-public.txt', 'r') as sr:
        edges = []
        for row in csv.DictReader(sr, delimiter='\t'):
            edges.append((int(row['from']), int(row['to'])))
        probs = pipeline.predict_proba(edges)

    # write results
    with open('data/test-public-predict.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Id', 'Prediction'])
        writer.writeheader()
        for i, prob in enumerate(probs):
            writer.writerow({'Id': i + 1, 'Prediction': prob[list(pipeline.classes_).index(1)]})

if __name__ == '__main__':
    main()
