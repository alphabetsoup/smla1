from DirectedEdgeNetwork import DirectedEdgeNetwork as den

ep = den("../data/train-sorted.txt")
ep.loadTrainDataAdjInv("../data/train-inv-sorted.txt")
ep.loadNodeDegree("../data/node_degree.txt","../data/node_inv_degree.txt")
ep.train()
ep.plot()
ep.predict('../data/test-public.txt','../data/test-prediction.txt')
