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
        self.transform1=nn.Linear(self.hidden_size,self.hidden_size)
        self.transform2=nn.Linear(self.hidden_size,1)

        self.optimizer=optim.Adam(self.parameters(),self.args.lr)

    def forward(self,states,hiddens):

        states = states.to(self.args.device)
        paths= hiddens.to(self.args.device)

        #将state 与path 进行拼接 得到输入向量
        inputs=torch.cat((states,paths),dim=-1) #[64,400]
        # 对hidden进行变换，得到reward
        hidden=F.relu(self.transform1(inputs),inplace=True)

        # No sigmoid
        batchRewards=self.transform2(hidden) #[64,1]
        return batchRewards


def train_d_model(d_model,batchStates_n,batchHiddens_n,batchStates_p,batchHiddens_p,mode='BCE'):


    batchStates_p = torch.Tensor(batchStates_p).float().to(d_model.args.device) #[20,batch_size,300]
    batchHiddens_p = torch.Tensor(batchHiddens_p).float().to(d_model.args.device) #[20,batch_size,100]
    batchStates_n = torch.Tensor(batchStates_n).float().to(d_model.args.device)   #[20,batch_size,300]
    batchHiddens_n = torch.Tensor(batchHiddens_n).float().to(d_model.args.device) #[20,batch_size,100]

    batchStates_p=batchStates_p.view(-1,batchStates_p.shape[-1])    #[batch_size*hops*k,300]
    batchHiddens_p=batchHiddens_p.view(-1,batchHiddens_p.shape[-1]) #[batch_size*hops*k,100]
    batchStates_n=batchStates_n.view(-1,batchStates_n.shape[-1])    #[batch_size*hops*k,300]
    batchHiddens_n=batchHiddens_n.view(-1,batchHiddens_n.shape[-1]) #[batch_size*hops*k,100]


    if mode=='BCE':
        batchStates=torch.cat((batchStates_p,batchStates_n),dim=0)
        batchHiddens=torch.cat((batchHiddens_p,batchHiddens_n),dim=0)
        labels=torch.Tensor([1]*len(batchHiddens_p)+[0]*len(batchHiddens_n)).float().unsqueeze(dim=-1).to(d_model.args.device)
        pred=d_model(batchStates,batchHiddens)
        loss=F.binary_cross_entropy_with_logits(pred,labels)

    elif mode=='MSE':
        pred_labels_p=d_model(batchStates_p,batchHiddens_p)  #[batch_size*hops*k,1]
        pred_labels_n=d_model(batchStates_n,batchHiddens_n)  #[batch_size*hops*k,1]
        #loss=F.binary_cross_entropy_with_logits(pred_labels, labels)

        loss=0.5*(torch.mean((pred_labels_p-1) ** 2+torch.mean(pred_labels_n+1)**2))

    d_model.optimizer.zero_grad()

    loss.backward()
    d_model.optimizer.step()

    del batchStates_p,batchHiddens_p,batchStates_n,batchHiddens_n
    torch.cuda.empty_cache()

    return loss.item()

def g_loss(d_model,batchStates_n,batchHiddens_n,mode):
    batchStates_n=torch.tensor(batchStates_n).to(d_model.args.device)
    batchHiddens_n=torch.tensor(batchHiddens_n).to(d_model.args.device)

    batchStates_n=batchStates_n.view(-1,batchStates_n.shape[-1])
    batchHiddens_n=batchHiddens_n.view(-1,batchHiddens_n.shape[-1])

    pred_labels = d_model(batchStates_n, batchHiddens_n)

    if mode=='MSE':
        loss = 0.5* torch.mean((pred_labels-1) ** 2)

    elif mode=='BCE':
        labels=torch.tensor([1]*len(batchStates_n)).float().unsqueeze(dim=-1).to(d_model.args.device)
        loss = F.binary_cross_entropy_with_logits(pred_labels, labels)

    del batchStates_n,batchHiddens_n,pred_labels
    torch.cuda.empty_cache()

    return loss
