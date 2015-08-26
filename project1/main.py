
import csv
import itertools
import pickle
import random

import numpy as np
from project1.features import *
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def main():
    with open('data/train.pickle', 'rb') as sr:
        g = pickle.load(sr)
        pipeline = Pipeline([
            ('features', FeatureUnion([
                ('common_neighbors', CommonNeighbors(g)),
            ])),
            ('logreg', LogisticRegression()),
        ])
        # sample some edges and non-edges
        n = 1000
        edges = list(itertools.islice(g.edges(), n))
        non_edges = []
        while len(non_edges) < n:
            u = random.randrange(g.num_vertices())
            v = random.randrange(g.num_vertices())
            if g.edge(u, v) is None:
                non_edges.append((u, v))
        pipeline.fit(edges + non_edges, np.hstack([np.ones(n), np.zeros(n)]))

    # predict
    with open('data/test-public.txt', 'r') as sr:
        edges = []
        for row in csv.DictReader(sr, delimiter='\t'):
            edges.append((row['from'], row['to']))
        probs = pipeline.predict_proba(edges)

    # write results
    with open('data/test-public-predict.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Id', 'Prediction'])
        writer.writeheader()
        for i, prob in enumerate(probs):
            writer.writerow({'Id': i + 1, 'Prediction': prob[list(pipeline.classes_).index(1)]})

if __name__ == '__main__':
    main()