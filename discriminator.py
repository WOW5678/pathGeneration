# -*- coding:utf-8 -*-
"""
@Time: 2019/10/09 8:55
@Author: Shanshan Wang
@Version: Python 3.7
@Function:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

class Discriminator(nn.Module):
    def __init__(self,args):
        super(Discriminator, self).__init__()

        self.args = args
        self.hidden_size = args.d_hidden_size

        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.hidden = self.initHidden(args)
        self.transform=nn.Linear(self.hidden_size,1)

        self.optimizer=optim.Adam(self.parameters(),self.args.lr)


    # 其中x是对路径编码之后的表示 所以不需要再进行embedding
    def forward(self, states,paths):
        batchRewards=[]
        # 将states,path进行拼接
        for state,path in zip(states,paths): # state,path是针对每个样本的，元素是一个list
            # 将state 与path 进行拼接 得到输入向量
            self.hidden=self.initHidden(self.args)

            example_reward=[]
            for s,p in zip(state,path): # 针对样本中的每个hop
                #print('s.shape,p.shape:',s.shape,p.shape)
                p=torch.sum(p,dim=0).unsqueeze(dim=0)
                #print('p.shape:',p.shape)
                input=torch.cat((s,p),dim=1).unsqueeze(dim=0)
                output, self.hidden = self.gru(input, self.hidden)
                #对hidden进行变换，得到reward
                reward=F.sigmoid(self.transform(self.hidden))
                example_reward.append(reward)
            batchRewards.append(example_reward)

        return batchRewards

    def forward2(self,states,paths):
        #将state 与path 进行拼接 得到输入向量
        inputs=torch.cat((states,paths),dim=-1) #[64,3,400]
        output,hidden=self.gru(inputs,None)
        # 对hidden进行变换，得到reward
        batchRewards=F.sigmoid(self.transform(output).squeeze(-1))
        return batchRewards

    def initHidden(self, args):
        hidden = torch.zeros((1, 1, args.d_hidden_size))
        hidden = torch.Tensor(hidden).to(args.device)
        return hidden

def train_d_model(d_model,batchStates_n,batchPaths_n,batchStates_p,batchPaths_p):

    # 首先对这个batch的数据集进行规整
    #需要先对batchStates_n,batchPaths_n进行处理

    batchStates_n_=torch.zeros((len(batchStates_n),3,d_model.args.path_hidden_size*3)).to(d_model.args.device)
    batchPaths_n_ = torch.zeros((len(batchPaths_n), 3, d_model.args.path_hidden_size)).to(d_model.args.device)

    for i in range(len(batchStates_n)):
        sample_s=torch.zeros((3,d_model.args.path_hidden_size*3))
        sample_p = torch.zeros((3, d_model.args.path_hidden_size))
        for j in range(len(batchStates_n[i])):
            sample_s[j]=batchStates_n[i][j].squeeze(0)
            sample_p[j] = torch.sum(batchPaths_n[i][j],dim=0)

        batchStates_n_[i]=sample_s
        batchPaths_n_[i]=sample_p

    states=batchStates_n_+batchStates_p
    paths=batchPaths_n_+batchPaths_p

    labels=torch.ones((len(paths),3))
    for i in range(len(batchStates_n)):
        labels[i]=torch.zeros(3)

    labels=labels.float().to(d_model.args.device)
    # 得到模型预测出的label
    # pred_labels是个长度为2*batch_size的list,每个元素都是一个list,list中包含三个元素，每个元素都是一个tensor
    pred_labels=d_model.forward2(states,paths)
    loss=F.binary_cross_entropy_with_logits(pred_labels, labels)
    d_model.optimizer.zero_grad()
    loss.backward(retain_graph=True)
    d_model.optimizer.step()
