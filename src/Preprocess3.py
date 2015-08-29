from DirectedEdgeNetwork import DirectedEdgeNetwork as den


ep = den("../data/train-sorted.txt")
ep.loadTrainDataAdjInv("../data/train-inv-sorted.txt")
ep.generateAdamicAdarFeature('../data/adamic-adar-followers.txt','../data/adamic-adar-following.txt')
