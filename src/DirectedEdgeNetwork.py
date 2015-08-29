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

def notList(l, r, s, n):
    # return a list of length n containing unique integers from list s that are not in list l or r
    count = 0
    for i in s:
        if i not in l and i != r and count < n:
            count += 1
            yield i

class DirectedEdgeNetwork():
    def __init__(self, trainpath):
        self.edge_to = defaultdict(list)
        print("Loading training data into dictionary")
        with open(trainpath,"r") as traindatafile:
            for line in traindatafile:
                s = [int(i) for i in line.split()]
                k = s.pop(0)
                self.edge_to[k].extend(sorted(s))
        self.all_nodes = set(self.edge_to.keys())
        for row in self.edge_to.values():
            self.all_nodes.union(row)

    def writeTrainDataSorted(self,trainsortpath):
        keys = sorted([v for v in self.edge_to.keys()])
        print("Writing sorted train data file")
        with open(trainsortpath,'w') as trainsort:
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
       
    def writeNodeDegree(self,nodedegpath):
        print("Writing out node degree")
        with open(nodedegpath,'w') as outfile: #'../data/node_degree.txt'
            for pair in self.NodeDegree.items():
                print(pair, file=outfile)

        print("Creating a list of all node ids")
        self.node_keys = sorted(list(set(self.edge_to.keys()) | set(self.NodeDegree.keys())))

    def generateInvAdjDict(self):
        print("Generating inverse graph adjacency dictionary")
        self.edge_from = defaultdict(list)
        for k, l in self.edge_to.items():
            for n in l:
                self.edge_from[n].append(k)
        
    def writeTrainDataAdjInvSorted(self,traininvsortpath):
        keys = sorted([v for v in self.edge_from.keys()])
        print("Writing sorted train data inv adjacency list file")
        with open(traininvsortpath,'w') as trainsort:
            for k in keys:
                self.edge_from[k].sort()
                print(str(k)+" ",end="",file=trainsort)
                print(*self.edge_from[k],sep=' ',file=trainsort)        

    def loadTrainDataAdjInv(self,traininvpath):
        self.edge_from = defaultdict(list)
        print("Loading training data into dictionary")
        with open(traininvpath,"r") as traindatafile:
            for line in traindatafile:
                s = [int(i) for i in line.split()]
                k = s.pop(0)
                self.edge_from[k].extend(sorted(s))

    def generateAdamicAdarFeature(self,followerpath, followingpath):
        # for each pair of nodes, compute common followers of those nodes
        with open(followerpath,'w') as followers:
            for k1,l1 in self.edge_from.items():
                for k2,l2 in self.edge_from.items():
                    print("hello")



    def loadNodeDegree(degpath,invpath):
        self.NodeDegree = defaultdict(int)
        self.NodeInvDegree = defaultdict(int)
        with open(degpath,'r') as degfile:
            for line in degfile:
                tup = make_tuple(line)
                self.NodeDegree[tup[0]] = tup[1]
        with open(invpath,'r') as invdegfile:
            for line in invdegfile:
                tup = make_tuple(line)
                self.NodeInvDegree[tup[0]] = tup[1]
            
    def computeNodeDegreeDistribution(self,degdistpath):
        print("Computing distribution of node degree")
        vals = self.NodeDegree.values()
        bins = [vals.count(v) for v in range(0,max(vals)+1)]
        print("Writing distribution of node degree") #'../data/node_degree_bins_cumulative_sum.txt'
        with open(degdistpath,'w') as outfile:
            for i,v in enumerate(bins):
                print(str(i)+"\t"+str(sum(bins[i:])), file=outfile)

    def extractNodeInvDegree(self):
        print("Computing node inv degree")
        # 2: Node inv degree. Number of nodes this node follows
        self.NodeInvDegree = {k: 0 if k not in self.edge_to else len(self.edge_to[k]) for k in self.node_keys}
        
    def writeNodeInvDegree(self,nodedeginvpath):
        print("Writing node inv degree")
        with open(nodedeginvpath,'w') as outfile: #'../data/node_inv_degree.txt'
            for pair in self.NodeInvDegree.items():
                print(pair, file=outfile)

    def train(self): 
        print("Training with a sample of the first 100 nodes")
        #node_keys100 = self.node_keys[0:100]
        node_keys100 = [v for v in self.edge_to.keys()][:50]
        
        X = []
        Y = []
        
        for j in node_keys100:
            for k in chain(self.edge_to[j],notList(self.edge_to[j],j,self.all_nodes,100)):
                X.append( [self.NodeDegree[j],self.NodeDegree[k],self.NodeInvDegree[j],self.NodeInvDegree[k]] )
                Y.append( 1 if k in self.edge_to[j] else 0 )

        X = np.matrix(X)

        with open("../data/Xfeats.txt",'w') as xout:
            for feat in X:
                print(feat, file=xout)
        with open("../data/Yfeats.txt",'w') as yout:
            for feat in Y:
                print(feat, file=yout)
        print("Attempting logreg fit")
        self.logreg = linear_model.LogisticRegression(C=1e5)
        self.logreg.fit(X, Y)
        self.X = X
        self.Y = Y

    def plot(self):
        print("Plotting not implemented yet")
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
    
    def predict(self,testpath,predictpath):
        # predict
        print("Predicting the test data...")
    
        test = np.zeros(shape=(2000,4))
        with open(testpath) as testfile:
            for line in testfile:
                a = [int(x) for x in line.split()]
                test[a[0]-1] = [self.NodeDegree[a[1]],self.NodeDegree[a[2]],self.NodeInvDegree[a[1]],self.NodeInvDegree[a[2]]]
        C = self.logreg.predict(test)
        with open(predictpath,'w') as testout:
            for p in C:
                print(p,file=testout)



    

