
import argparse
import csv
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC

from project1.classifiers import *
from project1.features import *


def score(name, y, probs, classes_):
    print('{} auc:\t\t{:.4f}'.format(name, roc_auc_score(y, probs[:, list(classes_).index(1)])))
    print('{} accuracy:\t{:.4f}'.format(name, accuracy_score(y, classes_[np.argmax(probs, axis=1)])))


def dev(g, estimator):
    # generate results for the dev set
    with open('data/dev_1000.pickle', 'rb') as dev_sr, open('data/train_1000.pickle', 'rb') as train_sr:
        dev_edges, dev_y = pickle.load(dev_sr)
        train_edges, train_y = pickle.load(train_sr)
    with g.temp_remove_edges(dev_edges[dev_y == 1]):
        with g.temp_remove_edges(train_edges[train_y == 1]):
            estimator.fit((g, train_edges), train_y)
            score('train', train_y, estimator.predict_proba((g, train_edges)), estimator.classes_)
        score('dev', dev_y, estimator.predict_proba((g, dev_edges)), estimator.classes_)


def test(g, estimator):
    # generate results for the test set
    with open('data/dev_1000.pickle', 'rb') as train_sr:
        train_edges, train_y = pickle.load(train_sr)
    with g.temp_remove_edges(train_edges[train_y == 1]):
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
                # ('degrees', Degrees()),
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
