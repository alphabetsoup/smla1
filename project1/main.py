
import argparse
import csv
import pickle
import random
from contextlib import contextmanager
import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC

from scipy import stats

from project1.classifiers import *
from project1.features import *


@contextmanager
def gen_classif_data(g, n):
    # sample some edges and non-edges
    us = set()
    edges = []
    g.compute_in_deg_dict()
    in_deg_dict_keys = sorted(list(g.in_deg_dict.keys()))
    #with open('data/in_deg_dict_keys.txt','w') as degkeyfile:
    #    print(in_deg_dict_keys,sep="\n",file=degkeyfile)
    #in_deg_dict_keys.remove(1) # do not allow nodes of degree zero

    sampdist_x = []
    sampdist_p = []
    with open('data/sample-dist-no-test.txt','r') as sdf:
        for row in csv.reader(sdf, delimiter=','):
            sampdist_x.append(int(row[0]))
            sampdist_p.append(float(row[1]))
    
    sd = stats.rv_discrete(name="test_in_deg",values=(sampdist_x,sampdist_p))

    # Force positive set (TODO put in graph class)
    positive_list = [
#        (2747594, 3555429),
#        (980324, 3363547),
#        (1153747, 2749491),
#        (81127, 2846365),
#        (2534470, 1240409),
#        (1615949, 138889),
#        (3886839, 3651482),
#        (2139873, 4263684),
#        (3089721, 1740255),
#        (1190100, 2440343),
#        (2424413, 2420807),
#        (1305126, 2108404),
#        (4521234, 4307990),
#        (314895, 4556298),
#        (2213160, 49007),
#        (3238736, 988752),
#        (1140489, 2001517),
#        (713720, 1525392),
#        (3288815, 1234319),
#        (1112232, 3286322),
#        (2863272, 855019)
    ]
    for e in positive_list:
        #edges.append((e[0],e[1]))
        us.add(e[0])

    loglen = math.log(len(in_deg_dict_keys))
    while len(edges) < (n - len(positive_list)):
        #v = random.choice(g.in_deg_dict[in_deg_dict_keys[max(0,min(len(in_deg_dict_keys)-1,int(math.floor(math.exp(random.uniform(0,loglen)))))-2)]])
        #deg_dict_index = max(0,min(len(in_deg_dict_keys)-1,int(math.floor(math.exp(random.uniform(0,loglen))))-3))
        #deg_dict_index = max(0,min(len(in_deg_dict_keys)-1,int(math.floor(random.expovariate(1.0/143.2)))))
        #v = random.choice(g.in_deg_dict[in_deg_dict_keys[deg_dict_index]])
        v = random.choice(g.in_deg_dict[sd.rvs()])


        if not len(g.in_dict[v]) >= 1:
            continue
        u = random.choice(g.in_dict[v])
        if u in us:
            continue

        if g.edge_excluded(u,v):
            continue
        # remove edges since the edges to predict are not supposed to be in the training graph
        g.remove_edge(u, v)
        edges.append((u, v))
        us.add(u)
    non_edges = []
    while len(non_edges) < n:
        #v = random.choice(g.in_deg_dict[in_deg_dict_keys[min(len(in_deg_dict_keys)-1,int(math.exp(random.uniform(0,loglen))))]])
        #deg_dict_index = max(0,min(len(in_deg_dict_keys)-1,int(math.floor(math.exp(random.uniform(0,loglen))))-3))
        #deg_dict_index = max(0,min(len(in_deg_dict_keys)-1,int(math.floor(random.expovariate(1.0/143.2)))))
        #v = random.choice(g.in_deg_dict[in_deg_dict_keys[deg_dict_index]])
        v = random.choice(g.in_deg_dict[sd.rvs()])

        u = random.randrange(g.num_vertices)
        #while u < g.num_vertices and (u in us or u in g.in_dict[v] or u==v):
        #    u += 1
        #if u >= g.num_vertices:
        #    continue
        if u in us:
            continue
        if not (u != v and u not in g.in_dict[v]):
            continue
        if g.edge_excluded(u,v):
            continue
        if not len(g.in_dict[v]) >= 1:
            continue
        if not (len(g.out_dict[u]) >= 1 and len(g.in_dict[u]) >= 1 and len(g.in_dict[v]) >= 1):
            continue
        non_edges.append((u, v))
        us.add(u)
    yield np.array(edges + positive_list + non_edges), np.hstack([np.ones(n), np.zeros(n)])
    for u, v in edges:
        g.add_edge(u, v)
    #for e in positive_list:
    #    g.remove_edge(e[0], e[1])

def gen_naive_classif_data(g, n):
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
    choosable = list(set(u for u, vs in g.out_dict.items() if vs) & set(u for u, vs in g.in_dict.items() if vs))
    while len(non_edges) < n:
        u = random.choice(choosable)
        v = random.choice(choosable)
        if v not in g.out_dict[u]:
            non_edges.append((u, v))
    yield np.array(edges + non_edges), np.hstack([np.ones(n), np.zeros(n)])
    for u, v in edges:
        g.add_edge(u, v)



def score(name, y, probs, classes_):
    print('{} auc:\t\t{:.4f}'.format(name, roc_auc_score(y, probs[:, list(classes_).index(1)])))
    print('{} accuracy:\t{:.4f}'.format(name, accuracy_score(y, classes_[np.argmax(probs, axis=1)])))


def dev(g, estimator):
    # generate results for the dev set
    with gen_classif_data(g, 1000) as (dev_edges, dev_y):
        with gen_classif_data(g, 8000) as (train_edges, train_y):
            estimator.fit((g, train_edges), train_y)
            score('train', train_y, estimator.predict_proba((g, train_edges)), estimator.classes_)
        score('dev', dev_y, estimator.predict_proba((g, dev_edges)), estimator.classes_)


def test(g, estimator):
    # generate results for the test set
    with gen_classif_data(g, 8000) as (train_edges, train_y):
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
                #('degrees', Degrees()),
                ('kNN', kNN()),
                ('common_neighbors', CommonNeighbors()),
                ('adamic_adar', AdamicAdar()),
                ('katz', Katz(5, 0.5)),
            ])),
            ('logreg', LogisticRegression()),
            # ('svm', SVC(kernel='rbf', probability=True))
        ])
        pipeline = GraphBaggingClassifier(pipeline, 4)
        if args.mode == 'dev':
            dev(g, pipeline)
        elif args.mode == 'test':
            test(g, pipeline)

if __name__ == '__main__':
    main()
