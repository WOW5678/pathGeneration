# -*- coding:utf-8 -*-
"""
@Time: 2019/09/27 10:51
@Author: Shanshan Wang
@Version: Python 3.7
@Function:
"""
import numpy  as np
import torch
import random
from torch.autograd import Variable
#from itertools import ifilter
a=[[1,2,3,4,5,0,0,0],[2,2,4,0,0]]
[row.reverse() for row in a]
print(a)

print([next(x for x in row if x >0) for row in a])