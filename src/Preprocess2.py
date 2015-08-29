from DirectedEdgeNetwork import DirectedEdgeNetwork as den


ep = den("../data/train-sorted.txt")
ep.extractNodeDegree()
ep.writeNodeDegree('../data/node_degree.txt')
ep.extractNodeInvDegree()
ep.writeNodeInvDegree('../data/node_inv_degree.txt')
