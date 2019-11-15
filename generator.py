# -*- coding:utf-8 -*-
"""
@Time: 2019/09/25 17:01
@Author: Shanshan Wang
@Version: Python 3.7
@Function:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy

from modelUtils import ActionSelection,ReplayBuffer,VanillaConv,PathEncoder,PathDecoder
import torch.optim as optim
import numpy as np
import math
from discriminator import *

class Generator(nn.Module):
    def __init__(self,args,data,g,level2_parients):
        super(Generator, self).__init__()
        self.args=args
        self.data=data
        self.g=g
        self.level2_parients = level2_parients
        print('self.level2_parients:',self.level2_parients)
        self.cnn = VanillaConv(args,vocab_size=data.size())
        self.pathEncoder = PathEncoder(args,self.g)
        self.pathDecoder = PathDecoder(args)
        self.pathHist = []  # 保存着已经选择的路径（只保留最后的一个ICD）

        self.attn = nn.Linear(args.node_embedding_size * 4, args.node_embedding_size * 3)
        #self.attn_combine = nn.Linear(args.node_embedding_size * 2, args.node_embedding_size)


        # 与强化学习有关的模块
        #self.buffer =
        self.ActionSelection = ActionSelection(args)
        #self.loss_func = nn.MSELoss()

        self.learn_step_counter = 0  # for target_net updating
        self.gamma = 0.9  # 计算未来奖励的折扣率
        self.epsilon = 0.95  # agent最初探索环境选择action的探索率
        self.epsilon_min = 0.05  # agent控制随机探索的阈值
        self.epsilon_decay = 0.995  # 随着agent做选择越来越好，降低探索率

        self.optimizer=optim.Adam(self.parameters(),lr=args.lr)


    def forward(self,d_model,ehrs,hier_labels,buffer,eval_flag):
        # 针对batch 中的每个样本(每个电子病历)进行单独的取样
        # 1.得到电子病历的表示
        ehrRrep = self.cnn(ehrs) #[64,300]
        actionLists = [[[self.args.node2id.get('ROOT')] for i in range(self.args.k)] for j in
                       range(self.args.batch_size)]
        batchStates=[]
        batchHiddens=[]
        batchRewards=0.0
        for k in range(self.args.k):
            paths = [row[k] for row in actionLists] #[[**],[**].....]
            actions = [[self.args.node2id.get('ROOT')] for i in range(self.args.batch_size)]

            for hop in range(2):

                path = self.pathEncoder(paths)  # [64,100]
                # hidden = self.pathDecoder(path)  # [64,10,100]
                hidden = path.squeeze(1)  # [64,100]
                state = F.relu(self.attn(torch.cat((hidden, ehrRrep), 1)), inplace=True)  # [64,300]


                children, children_len = action_space(actions, self.args)
                if eval_flag:
                    actions,actionIndexs = self.ActionSelection.act(state, children, children_len,k,0)  # [64,229],[64]
                else:
                    actions,actionIndexs = self.ActionSelection.act(state, children, children_len,k,
                                                                     self.epsilon)  # [64,229],[64,1]

                next_paths=[row+action for row,action in zip(paths,actions)]
                #print('next_paths:',next_paths)
                # 5. 根据当前选择的action 计算reward值
                #reward=d_model(state,hidden) # 对当前选择出的action 进行评估 #[batch_size,1]
                if eval_flag:
                    pass
                else:
                    reward = self.getReward(next_paths, hier_labels, hop)
                    batchRewards += sum([sum(r) for r in reward])
                    # print('k,reward:',k, reward)
                    if hop == 1:
                        done = True
                    else:
                        done = False
                    #done=True
                    # 6. 将这次尝试加入到buffer中
                    buffer.push(ehrs.detach().cpu().numpy(),paths,actions,next_paths,actionIndexs,reward,done)

                paths=next_paths # 更新state

            #print('paths:',paths)
            for i in range(len(actionLists)):
                #print('paths[i]:',paths[i])
                actionLists[i][k]=paths[i]
        print('batchRewards:',batchRewards)
        return actionLists,batchStates,batchHiddens


    def getReward(self,paths,hier_labels,hop):
        # 通过层次计算是否给予奖励值
        #处理hier_labels
        #print('before hier_labels:',hier_labels)
        new_hier_labels=[]
        for sample in hier_labels:
            new_hier_labels.append([row[:hop+1] for row in sample])
        #print('hier_labels:',new_hier_labels)
        # 处理paths
        paths = [row[1:] for row in paths]
        rewards = []
        for p, h_ in zip(paths, new_hier_labels):
            flag = False
            for h in h_:
                if p == h:
                    rewards.append([10])
                    flag = True
                    #break

            if flag == False:
                rewards.append([-1])
        return rewards


    def calcuateValue_target(self,ehrs,paths,eval_flag=True):

        # 针对batch 中的每个样本(每个电子病历)进行单独的取样
        # 1.得到电子病历的表示
        ehrRrep = self.cnn(ehrs)  # [64,300]
        # 重新计算历史路径的hidden(否则要设置retain_graph=True)
        batch_action_values=torch.zeros((self.args.batch_size,1)).to(self.args.device)
        # 因为paths 可能是不同hop得到的，长度不一致，因此需要每个样本每个样本的计算

        for i in range(self.args.batch_size):
            #print('paths[i]:',paths[i])
            actions=[[paths[i][-1]]]
            #print('actions:',actions)
            # 2.得到路径的表示
            path = self.pathEncoder([paths[i]])  # [1,1,100]
            #hidden = self.pathDecoder(path)  # [64,10,100]
            hidden = path.squeeze(1)  # [1,100]
            state = F.relu(self.attn(torch.cat((hidden, ehrRrep[i].unsqueeze(0)), 1)), inplace=True)  # [1,300]

            children, children_len = action_space(actions, self.args)  # children:[1,2,229],children_len:[64]
            # 通过target 网络得到最大q值对应的actionIds
            actionId,action_value= self.ActionSelection.evalAct(state,children, children_len)

            # # 找出该actionId对应的eval计算出的next_q_value
            # action_value = gen_model_target.ActionSelection(state)[:, actionId[0]]  # [1,228]
            batch_action_values[i] = action_value

        return batch_action_values

    def calcuateValue_eval(self, ehrs, paths, actionIds,eval_flag=True):

        # 针对batch 中的每个样本(每个电子病历)进行单独的取样
        # 1.得到电子病历的表示
        ehrRrep = self.cnn(ehrs)  # [64,300]
        # 重新计算历史路径的hidden(否则要设置retain_graph=True)
        batch_action_values = torch.zeros((self.args.batch_size, 1)).to(self.args.device)
        # 因为paths 可能是不同hop得到的，长度不一致，因此需要每个样本每个样本的计算

        for i in range(self.args.batch_size):
            # 2.得到路径的表示
            path = self.pathEncoder([paths[i]])  # [1,1,100]
            # hidden = self.pathDecoder(path)  # [64,10,100]
            hidden = path.squeeze(1)  # [1,100]
            state = F.relu(self.attn(torch.cat((hidden, ehrRrep[i].unsqueeze(0)), 1)), inplace=True)  # [1,300]
            #print('actionIds:',actionIds[i])
            action_value = self.ActionSelection(state)[:,actionIds[i]] #[1,228]

            batch_action_values[i] = action_value

        return batch_action_values

    # 传入真实的action 得到真实的（state,path）对
    def teacher_force(self,ehrs,hier_labels,buffer):

        # 初始的actionList 全为root节点 （若K为5 则为5个root节点）
        actionLists = [[[self.args.node2id.get('ROOT')] for i in range(self.args.k)] for j in
                       range(self.args.batch_size)]
        # 1.得到电子病历的表示
        ehrRrep = self.cnn(ehrs)  # [64,300]

        batchStates = []
        batchHiddens = []

        for k in range(self.args.k):
            paths = [row[k] for row in actionLists] #[[**],[**].....]
            actions=[[self.args.node2id.get('ROOT')] for i in range(self.args.batch_size)]
            for hop in range(1):

                # 2.得到路径的表示
                path = self.pathEncoder(paths)  # [64,100]
                #hidden = self.pathDecoder(path)  # [64,10,100]
                hidden = path.squeeze(1)    # [64,100]
                state = F.relu(self.attn(torch.cat((hidden, ehrRrep), 1)), inplace=True)  # [64,300]
                # 3. 直接从层次化的true labels中找见对应hop的表示 作为action 并更新actionLists
                if hop==0:
                    children, children_len = action_space(actions, self.args)
                    actions,actionIndexs=getHopAction(hier_labels,children,hop)
                    # 更新 paths
                    #paths=actions

                else:

                    children, children_len = action_space(actions, self.args)
                    # 找见上一个hop的action对应的孩子节点
                    actions,actionIdexs=getHopActionNext(hier_labels,children,paths,hop)

                #更新next_paths
                next_paths = [row + action for row, action in zip(paths, actions)]

                #print('next_paths:',next_paths)
                #4. 直接给出reward值

                reward=[[10]]*self.args.batch_size
                if hop == 1:
                    done = True
                else:
                    done = False

                #6. 将这次尝试加入到buffer中
                buffer.push(ehrs.detach().cpu().numpy(), paths, actions, next_paths,actionIndexs,reward, done)

                paths = next_paths  # 更新state

                batchStates.append(state.detach().cpu().numpy())
                batchHiddens.append(hidden.detach().cpu().numpy())

            for i in range(len(actionLists)):
                actionLists[i][k]=paths[i]

        return batchStates, batchHiddens

    def updateEpision(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def getHopAction(hier_labels,children, hop):

    selected_actions = [[row[hop] for row in sample] for sample in hier_labels]  #selected_actions:[[**,**,**],[**,**,**,**],..]
    selected_actions=[list(set(row)) for row in selected_actions]
    actions=[]
    actionIndexs=[]
    for i in range(len(selected_actions)): # i 为每个样本的索引

        # 从样本中选择出一个并得到索引
        #print('selected_actions[i]:',selected_actions[i])
        if 0 in selected_actions[i]:
            selected_actions[i].remove(0)

        action = random.choice(selected_actions[i]) # 从可选的actions中选择出一个action
        # 查找该action对应的actionId(即childrenId)
        actionId=children[i].cpu().numpy().tolist().index(action)
        actionIndexs.append([actionId])
        actions.append([action])
    return actions,actionIndexs

def getHopActionNext(hier_labels,children,selected_actions,hop):

    new_hier_labels = []
    for sample in hier_labels:
        new_hier_labels.append([row[:hop + 1] for row in sample])

    actionIndexs=[]
    actions=[]
    #print('new_hier_labels:',new_hier_labels)
    for i in range(len(selected_actions)): # 找见以每一条selected_actions为开头的路径（注：可能有多条）
        #print('selected_actions:',selected_actions[i])
        paths=[row for row in new_hier_labels[i] if row[:hop]==selected_actions[i][1:]]
        #print('paths:',paths)
        action=random.choice(paths)[hop]
        # 找见对应的actionId
        #print(children[i])
        actionId = children[i].cpu().numpy().tolist().index(action)
        actionIndexs.append([actionId])
        actions.append([action])

    return actions,actionIndexs

def action_space(parients,args):

    childrens=[]
    children_lens=[]
    for sample_i in range(len(parients)):
        children=[torch.nonzero(row) for row in args.adj[parients[sample_i]]]
        children_len = [len(row) for row in children]
        for i in range(len(children)):
            children[i]=[item[0].cpu().item() for item in children[i]]
            #mask
            children[i]=children[i]+[0]*(args.max_children_num-len(children[i]))

        childrens.append(children)
        children_lens.append(children_len)
    childrens=torch.Tensor(childrens).long().squeeze(dim=1).to(args.device) #[64,229]
    children_lens=torch.Tensor(children_lens).long().squeeze(dim=-1).to(args.device) #[64]
    return childrens,children_lens



def run_pre_train_step(gen_model,current_batch):
    ehrs=[example.ehr for example in current_batch]
    ehrs=torch.Tensor(ehrs).long().to(gen_model.args.device)
    labels=[example.labels for example in current_batch]
    hier_labels=[example.hier_labels for example in current_batch]

    ehrRrep = gen_model.cnn(ehrs)  # [64,200]
    actionLists = [[[gen_model.args.node2id.get('ROOT')] for i in range(gen_model.args.k)] for j in
                   range(gen_model.args.batch_size)]
    batchStates = []
    batchHiddens = []
    for k in range(gen_model.args.k):
        actions = [[gen_model.args.node2id.get('ROOT')] for i in range(gen_model.args.batch_size)]
        paths = [row[k] for row in actionLists]  # [[**],[**].....]
        # print('k:{},paths:{}',paths)
        for hop in range(gen_model.args.hops):
            if hop == 0:
                state = ehrRrep
                children, children_len = action_space(actions, gen_model.args)
                actionIndexs, actions =gen_model.ActionSelection.act(state, children, children_len, k,
                                                                     0)  # [64,229],[64]
            else:
                path = gen_model.pathEncoder(paths)  # [64,100]
                # hidden = self.pathDecoder(path)  # [64,10,100]
                hidden = path.squeeze(1)  # [64,100]
                state = F.relu(gen_model.attn(torch.cat((hidden, ehrRrep), 1)), inplace=True)  # [64,300]

                # 3. action selection模块通过state 从孩子空间中选择要执行的action
                # 首先获得当前跳的action_space空间 然后DQN根据state在该空间中选择action
                # selected_action作为父节点 选择对应的孩子节点
                children, children_len = action_space(actions, gen_model.args)  # children:[64,229],children_len:[64]
                actionIndexs, actions = gen_model.ActionSelection.act(state, children, children_len, 0,
                                                                 0)  # [64,229],[64]
            # 4.根据beam_search 进一步筛选可能的action，并更新actionList（环境状态的一部分）
            next_paths = [row + action for row, action in zip(paths, actions)]
            paths = next_paths  # 更新state

        # print('paths:',paths)
        for i in range(len(actionLists)):
            # print('paths[i]:',paths[i])
            actionLists[i][k] = paths[i]
    # print('actionLists:',actionLists)
    return actionLists, batchStates, batchHiddens


def generated_negative_samples(gen_model,d_model,ehrs,hier_labels,buffer):
    # 从g_model中生成state,hidden(路径的表示)
    actionLists,batchStates,batchHiddens= gen_model(d_model,ehrs,hier_labels,buffer,eval_flag=False)

    # 释放内存和显存
    del ehrs
    #torch.cuda.empty_cache()
    return actionLists,batchStates,batchHiddens

def generated_positive_samples(gen_model,ehrs,hier_labels,buffer):

    batchStates,batchPaths = gen_model.teacher_force(ehrs, hier_labels,buffer)

    del ehrs
    #torch.cuda.empty_cache()

    return batchStates,batchPaths

def run_eval_step(gen_model,d_model, buffer,current_batch):
    ehrs = [example.ehr for example in current_batch]
    ehrs=torch.Tensor(ehrs).long().to(gen_model.args.device)

    #labels = [example.labels for example in current_batch]
    hier_labels = [example.hier_labels for example in current_batch]
    for i in range(len(hier_labels)):  # i为样本索引
        for j in range(len(hier_labels[i])):  # j为每个样本的每条路径索引
            if len(hier_labels[i][j]) < 2:
                hier_labels[i][j] = hier_labels[i][j] + [0] * (2 - len(hier_labels[i][j]))

    labels=[]
    for sample in hier_labels:
        labels.append([row[1] for row in sample])

    # 从g_model中生成state,action
    predPaths,_,_ = gen_model(d_model,ehrs,hier_labels,buffer,eval_flag=True)
    # 只需要返回预测的label路径

    del ehrs
    #torch.cuda.empty_cache()

    return labels, predPaths





