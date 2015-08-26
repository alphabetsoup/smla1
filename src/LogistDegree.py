__author__ = 'ldavies'

import numpy as np
import scipy.sparse as sp
import re
from collections import defaultdict
from collections import deque
from sklearn import linear_model
import matplotlib.pyplot as plt
from itertools import *
import json

class EdgePredictor():

    def __init__(self):
        self.edge_to = defaultdict(list)
        print("Loading training data into dictionary")
        with open("../data/train.txt","r") as traindatafile:
            for line in traindatafile:
                s = [int(i) for i in line.split()]
                k = s.pop(0)
                self.edge_to[k].extend(sorted(s))

    def writeTrainDataSorted(self):
        keys = sorted([v for v in self.edge_to.keys()])
        print("Writing sorted train data file")
        print("Keys:")
        #print(keys)
        with open("../data/train-sorted.txt",'w') as trainsort:
            for k in keys:
                print(str(k)+" ",end="",file=trainsort)
                print(*self.edge_to[k],sep=' ',file=trainsort)
            
            

    def extractNodeDegree(self):
        # extract features
        print("Calculating node degree")
        self.NodeDegree = defaultdict(int)
        for k in self.edge_to.keys():
            for j in self.edge_to[k]:
                self.NodeDegree[j] += 1
       
        #self.keys = sorted(self.NodeDegree.keys())

        print("Writing out node degree")
        with open('../data/node_degree.txt','w') as outfile:
            for pair in self.NodeDegree.items():
                print(pair, file=outfile)

        print("Creating a list of all node ids")
        self.node_keys = sorted(list(set(self.edge_to.keys()) | set(self.NodeDegree.keys())))

    def computeNodeDegreeDistribution():
        print("Computing distribution of node degree")
        vals = self.NodeDegree.values()
        bins = [vals.count(v) for v in range(0,max(vals)+1)]
        print("Writing distribution of node degree")
        with open('../data/node_degree_bins_cumulative_sum.txt','w') as outfile:
            for i,v in enumerate(bins):
                print(str(i)+"\t"+str(sum(bins[i:])), file=outfile)

        

    def extractNodeInvDegree(self):
        print("Computing node inv degree")
        # 2: Node inv degree. Number of nodes this node follows
        self.NodeInvDegree = {k: 0 if k not in self.edge_to else len(self.edge_to[k]) for k in self.node_keys}
        
        print("Writing node inv degree")
        with open('../data/node_inv_degree.txt','w') as outfile:
            for pair in self.NodeInvDegree.items():
                print(pair, file=outfile)

    def train(self): 
        print("Training with a sample of the first 100 nodes")
        #node_keys100 = self.node_keys[0:100]
        node_keys100 = [v for v in self.edge_to.keys()][:100]
        
        # try to fit in logistic recression
        #X = [[self.NodeDegree[j],self.NodeDegree[k],self.NodeInvDegree[j],self.NodeInvDegree[k]] for j in node_keys100 for k in chain(self.edge_to[j], self.edge_to[j+100])]
        #Y = [self.edge_to[j].count(k) for j in node_keys100 for k in chain(self.edge_to[j], self.edge_to[j+100])]
        X = [[self.NodeDegree[j],self.NodeDegree[k],self.NodeInvDegree[j],self.NodeInvDegree[k]] for j in node_keys100 for k in self.edge_to[j]]
        Y = [1 for j in node_keys100 for k in self.edge_to[j]]
        X.extend([[self.NodeDegree[j],self.NodeDegree[k],self.NodeInvDegree[j],self.NodeInvDegree[k]] for j in node_keys100 for k in self.edge_to[j+100]])
        Y.extend([1 if k in self.edge_to[j] else 0 for j in node_keys100 for k in self.edge_to[j+100]])
        #print("Testing self.edge_to chain")
        #for j in node_keys100:
        #    print("test row: "+str(j))
        #    print(self.edge_to[j].values())
        #    for k in chain(self.edge_to[j], self.edge_to[j+100]):
        #        print(k, end=", ")
        print("Data to fit X: len = " + str(len(X)))
        for feat in X:
            print(feat)
        print("Data to fit Y: len = " + str(len(Y)))
        for feat in Y:
            print(feat)
        print("Attempting logreg fit")
        self.logreg = linear_model.LogisticRegression(C=1e5)
        self.logreg.fit(X, Y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        #x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        #y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        #xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        #Z = self.logreg.predict(np.c_[xx.ravel(), yy.ravel()])
        
        # Put the result into a color plot
        #Z = Z.reshape(xx.shape)
        #plt.figure(1, figsize=(4, 3))
        #plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
        
        # Plot also the training points
        #plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
        #plt.xlabel('Degree into node')
        #plt.ylabel('Degree out of node')
        
        #plt.xlim(xx.min(), xx.max())
        #plt.ylim(yy.min(), yy.max())
        #plt.xticks(())
        #plt.yticks(())
        
        #plt.show()

    def predict(self):
        # predict
        print("Predicting the test data...")
    
        test = np.zeros(shape=(2000,4))
        with open("../data/test.txt") as testfile:
            for line in testfile:
                a = [int(x) for x in line.split()]
                test[a[0]-1] = [self.NodeDegree[a[1]],self.NodeDegree[a[2]],self.NodeInvDegree[a[1]],self.NodeInvDegree[a[2]]]
        C = self.logreg.predict(test)
        with open("../data/test-p.txt",'w') as testout:
            for p in C:
                print(p,file=testout)



def LogistDegree():
    self.NodeDegree = defaultdict(int)
    self.NodeInvDegree = defaultdict(int)
    with open("../data/node_degree.txt",'r') as degfile:
        for line in degfile:
            tup = make_tuple(line)
            self.NodeDegree[tup[0]] = tup[1]
    with open("../data/node_inv_degree.txt",'r') as invdegfile:
        for line in invdegfile:
            tup = make_tuple(line)
            self.NodeInvDegree[tup[0]] = tup[1]
    

ep = EdgePredictor()
ep.writeTrainDataSorted()
ep.extractNodeDegree()
ep.extractNodeInvDegree()
ep.train()
ep.predict()

