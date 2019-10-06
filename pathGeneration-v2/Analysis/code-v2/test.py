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
a=[[1,2,3],[4,5,6],[7,8,9]]
a=[torch.Tensor(row) for row in a]
print('a:',a)
a=[row.data.numpy() for row  in a]
print('aa:',a)
action=Variable(torch.LongTensor(a))
print(action)