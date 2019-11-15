# -*- coding:utf-8 -*-
"""
@Time: 2019/09/10 21:04
@Author: Shanshan Wang
@Version: Python 3.7
@Function:
"""
import torch
import torch.nn  as nn
from collections import deque
import numpy as np
import random
import torch.autograd as autograd
import torch.nn.functional as F
from gcn import GCNNet

# USE_CUDA=torch.cuda.is_available()
# device = ("cuda:5" if torch.cuda.is_available() else 'cpu')
#


class ActionSelection(nn.Module):
    def __init__(self,args):
        super(ActionSelection, self).__init__()
        self.args=args
        self.layers=nn.Sequential(
            nn.Linear(args.state_size,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,args.max_children_num)
        )

    def forward(self,x):
        return self.layers(x)

    def act(self,state,action_space,children_len,k,epsilon):
        '''

        :param state:
        :param action_space: 为[64,229]
        :param children_len: [64]
        :param eval_flag:
        :return:
        '''
        # with torch.no_grad():
        q_value = self.layers(state) # [64,229]
        # 测试过程 只选择q_value最大的数据 没有随机性
        # 只从孩子节点中选择最大的
        actionInds=[]
        actions = []
        if random.random()>epsilon:

            # 开始采样环节  因此需要对这个batch中的单个样本进行采样
            # 直接选择最大的那个概率
            for i in range(self.args.batch_size):
                if children_len[i]>k:
                    actionInd=[torch.argsort(q_value[i][:children_len[i]])[k].item()]
                    action = [action_space[i][actionInd[0]].item()]
                else:
                    actionInd=[0]
                    action=[0]
                actionInds.append(actionInd)
                actions.append(action)
            return actions,actionInds

        else:
            # 从孩子节点中随机选择
            for i in range(self.args.batch_size):
                if children_len[i]>0:
                    #print('children_len:',children_len[i])
                    actionInd=[torch.randperm(children_len[i])[0].item()]
                    action = [action_space[i][actionInd[0]].item()]
                else:
                    actionInd=[0]
                    action=[0]
                #print('actionId:{},action:{}'.format(actionInd,action))
                actionInds.append(actionInd)
                actions.append(action)
            return actions,actionInds

    def evalAct(self,state,action_space,children_len):

        q_value = self.layers(state)  # [1,229]
        with torch.no_grad():
            if children_len[0]>0:
                actionInd=[torch.argsort(q_value[0][:children_len[0]])[0].item()]
                action = [action_space[0][actionInd[0]].item()]
                value=q_value[0][actionInd[0]]
            else:
                actionInd=[0]
                action=[0]
                value = q_value[0][actionInd[0]]
        return  actionInd,value


class ReplayBuffer(object):
    def __init__(self,capacity):
        self.buffer=deque(maxlen=capacity)

    def push(self,ehrs, actionLists, selected_actions, next_actionLists, actionIndexs,rewards, done):
        for ehr, actionList, selected_action, next_actionList, actionIndex,reward in zip(ehrs, actionLists, selected_actions, next_actionLists, actionIndexs,rewards):
            self.buffer.append((ehr, actionList, selected_action, next_actionList, actionIndex,reward, done))

    def sample(self,batch_size):
        ehrs, actionLists, selected_actions, next_actionLists,actionIndexs,rewards, dones=zip(*random.sample(self.buffer,batch_size))
        return ehrs,actionLists,selected_actions,next_actionLists,actionIndexs,rewards,dones

    def __len__(self):
        return len(self.buffer)

class VanillaConv(nn.Module):
    def __init__(self,args,vocab_size):
        super(VanillaConv, self).__init__()
        self.args=args
        chanel_num = 1
        filter_num = self.args.num_filter_maps
        filter_sizes = [3,5,7]

        self.embedding = nn.Embedding(vocab_size,args.cnn_embedding_size)
        self.conv1 = nn.ModuleList([nn.Conv2d(chanel_num, filter_num, (filter_size,self.args.cnn_embedding_size)) for filter_size in filter_sizes])
        self.dropout1 = nn.Dropout(self.args.dropout_rate)

    def forward(self, x):
        # 将每个电子病历转化成Tensor对象
        x = self.embedding(x)
        #print('x:',x.shape)
        x = x.unsqueeze(1) #[batch_size,1,200,emb_size]
        #print('x_unsequeeze:',x.shape)
        #print(F.relu(self.convs[0](x)).shape) # 每个不同的filter卷积之后为：[batch_size,32,198,1],[batch_size,32,197,1],[batch_size,32,196,1]
        #print('x:',x.shape) #[49, 1, 100]
        # 多通道的图卷积操作（单层卷积）
        x=[F.relu(conv(x),inplace=True) for conv in self.conv1]

        x_ = [item.squeeze(3) for item in x]
        x_ = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x_]
        x_ = torch.cat(x_, 1)
        x_ = self.dropout1(x_)
        return x_

# 对选择出的路径进行编码，返回路径的表示
# 目前的做法是：将当前的节点以及上一步的节点以及中间的边作为当前时刻的路径
class PathEncoder(nn.Module):
    def __init__(self,args,g):
        self.args=args
        super(PathEncoder, self).__init__()

        self.embedding=nn.Embedding(len(args.node2id),args.node_embedding_size)
        self.gcn=GCNNet()
        self.w_path=nn.Linear(args.node_embedding_size*2,args.node_embedding_size)
        self.embedding.weight = nn.Parameter(self.gcn(g, self.embedding.weight))

    def forward(self,actionList,teacher_force=False):
        #print('actionList:',actionList)
        current_node = [sample[-1] for sample in actionList]
        # print('current_node:',current_node)
        current_node = torch.Tensor(current_node).long().to(self.args.device)
        # print('current_node:',current_node)
        current_node_emb = self.embedding(current_node)  # [64,1,100]

        if len(actionList[0])>1: #说明不是第一个hop
            last_node=[sample[-2] for sample in actionList]
            last_node=torch.Tensor(last_node).long().to(self.args.device)
            last_node_emb=self.embedding(last_node) #[64,1,100]
            current_node_emb=current_node_emb*last_node_emb
        #print('self embedding weights:',self.embedding.weight)
        return current_node_emb


class PathDecoder(nn.Module):
    def __init__(self,args):
        self.args=args
        super(PathDecoder, self).__init__()
        self.hidden_size=args.path_hidden_size
        self.max_length=args.hops

        self.gru=nn.GRU(self.hidden_size,self.hidden_size)
        self.hidden=self.initHidden(args)

    # 其中x是对路径编码之后的表示 所以不需要再进行embedding
    def forward(self,input): # 其中input为decoder_input,hidden为decoder_hidden
        # print('before-self.hidden:',self.hidden.shape)
        output,self.hidden=self.gru(input,self.hidden)
        return output


    def initHidden(self,args):
        hidden = torch.zeros((1,self.args.k,args.path_hidden_size))
        hidden = torch.Tensor(hidden).to(args.device)
        return hidden



