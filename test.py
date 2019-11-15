# -*- coding:utf-8 -*-
"""
@Time: 2019/09/03 10:56
@Author: Shanshan Wang
@Version: Python 3.7
@Function: 实现分层的DQN网络
"""
import torch
import copy

import random
import math
import numpy as np

a=[[1],[2]]
for row in a:
    if [1] in row:
        print(row)