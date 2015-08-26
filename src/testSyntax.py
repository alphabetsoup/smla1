__author__ = 'ldavies'

import numpy as np
import re
from collections import defaultdict
from collections import deque
import matplotlib.pyplot as plt

bins = [v*v for v in range(0,100)]

for i,v in enumerate(bins):
    print(str(i)+" "+str(sum(bins[i:100])))
