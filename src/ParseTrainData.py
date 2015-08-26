__author__ = 'ldavies'

import numpy as np
import re
from collections import defaultdict
from collections import deque
from sklearn import linear_model
import matplotlib.pyplot as plt
import json

edge_to = defaultdict(deque)
with open("../data/train.txt","r") as traindatafile:
    for line in traindatafile:
        s = deque([int(i) for i in line.split()])
        k = s.popleft()
        edge_to[k].extend(sorted(s))
#       print("Assigning " + str(k) + " to k")
#       print(edge_to[k])

#for k in edge_to:
    #print("Node "+str(k)+" follows: ")
    #print(edge_to[k], sep=', ', end='\n')


# extract features


# 1: Node degree. Number of followers
#flat_g = sorted([x for v in edge_to.values() for x in v])
#X_1 = defaultdict({k:flat_g.count(k) for k in edge_to.keys()})
#X_1 = defaultdict(int).fromkeys(edge_to.keys())
X_1 = defaultdict(int)
for k in edge_to.keys():
    for j in edge_to[k]:
        X_1[j] += 1

with open('../data/node_degree.txt','w') as outfile:
    for pair in X_1.items():
        print(pair, file=outfile)
#    json.dump(X_1, outfile)

vals = sorted(X_1.values())
bins = [vals.count(v) for v in range(0,max(vals)+1)]
#print("Max degree is " + str(max(vals)))

#X_1_cum_sum = {i:sum(bins[i:]) for i in bins.keys()}
#    json.dump(X_1_cum_sum, outfile)

with open('../data/node_degree_bins_cumulative_sum.txt','w') as outfile:
    for i,v in enumerate(bins):
        print(str(i)+"\t"+str(sum(bins[i:])), file=outfile)

#plt.bar(bins.keys(),bins.values())
#plt.show()

# 2: Node inv degree. Number of nodes this node follows
X_2 = {k: len(edge_to[k]) for k in edge_to.keys()}

with open('../data/node_inv_degree.txt','w') as outfile:
    for pair in X_2.items():
        print(pair, file=outfile)



#logreg = linear_model.LogisticRegression(c=1e5)


#logreg.fit()
