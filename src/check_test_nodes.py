from DirectedEdgeNetwork import DirectedEdgeNetwork as den
from contextlib import contextmanager, suppress


edge_from_keys = []

ep = den("../data/train-sorted.txt")
#ep.loadTrainDataAdjInv("../data/train-inv-sorted.txt")

# now create a list of edgefrom keys
for k,l in ep.edge_to.items():
    edge_from_keys.extend(l)

edge_from_keys = list(set(edge_from_keys))

with open("edgefromkeys.txt",'w') as efk:
    print(edge_from_keys,sep="\n",file=efk)

with open('test-mauled.txt','r') as testpub:
    with open('test-node-report.txt','w') as rep:
    #    with suppress(ValueError):
        for line in testpub:
            a = line.split()
            print(a[0])
            if int(a[1]) not in ep.edge_to.keys():
                print("From node " + a[1] + " is not a from node in train data (i.e. not an index in the adj list): " + line, file=rep) 
            if int(a[1]) not in edge_from_keys:
                print("From node " + a[1] + " is not a to node in train data (i.e. not an index in reversadj list): " + line, file=rep) 
            if int(a[2]) not in ep.edge_to.keys():
                print("To node "+a[2]+" is not a from node in train data (i.e. not an index in the adj list):   " + line, file=rep) 
            if int(a[2]) not in edge_from_keys:
                print("To node "+a[2]+" is not a to node in train data (i.e. not an index in reverse adj list): " + line, file=rep)
