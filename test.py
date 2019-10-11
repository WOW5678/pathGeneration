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
from keras.utils import to_categorical

data=[1,2,3,2,1,0]
# 去重
data=list(set(data))
encoded=to_categorical(data,num_classes=5)
print(sum(encoded))
