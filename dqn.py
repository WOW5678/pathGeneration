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

USE_CUDA=torch.cuda.is_available()
device = ("cuda:2" if torch.cuda.is_available() else 'cpu')

Variable=lambda *args,**kwargs:autograd.Variable(*args,**kwargs).to(device)


class Net(nn.Module):
    def __init__(self,args):
        super(Net, self).__init__()

        self.layers=nn.Sequential(
            nn.Linear(args.state_size,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,args.max_children_num)
        )

    def forward(self,x):
        return self.layers(x)


class DQN(object):
    def __init__(self,args):
        self.args=args
        self.eval_net,self.target_net=Net(args),Net(args)
        self.learn_step_counter=0 # for target_net updating
        self.buffer=ReplayBuffer(capacity=5000)
        self.loss_func=nn.MSELoss()
        self.epsilon=0.5
        self.eval_net.to(args.device)
        self.target_net.to(args.device)

    def act(self,state,action_space,children_len,eval_flag=False):
        # 针对的是一个样本
        action_values=[] # q值用来训练模型 score值用来选择路径
        action_scores=[]
        actions_k=[]
        action_indexs=[]

        q_value = self.eval_net(state)  #[B,72] # 第一个hops时是[1,101]，后面每个hop都是[5,101]

        # 测试过程 只选择q_value最大的数据 没有随机性
        if eval_flag:
            self.epsilon=99999
        else:
            self.epsilon=0.5
        if random.random()<self.epsilon:
            # action_value:[1,5],actionInds:[1,5]
            for i in range(len(action_space)):
                # action shape,actionIds shape:[1,5]
                if children_len[i]>self.args.k:
                    # print('children_len[i]:',children_len[i])
                    # print('q_value[0][:children_len[i]]:',q_value[0][:children_len[i]])
                    score=F.softmax(q_value[0][:children_len[i]]) # 在所有孩子节点上得到一个概率分布
                    action_value,actionInds=torch.topk(q_value[0][:children_len[i]],self.args.k)

                    # print('action_value:',action_value) #[0.1316, 0.1185, 0.1068, 0.1000, 0.0992]
                    # print('actionInds:', actionInds)  # [94,  6, 55, 16, 73]
                    actions = action_space[i][actionInds]
                    # print('actions:', actions)  # [8190, 4546, 8179, 8190, 5937]
                    action_score,_=torch.topk(score,self.args.k)

                elif children_len[i]==0: # 说明父节点就是叶子节点 没有孩子节点 则直接选择第一个孩子（PAD） (使用action==0进行填充)
                    score = F.softmax(q_value[0])
                    action_value=[q_value[0][-1]] #  取最后的一个action
                    actions = torch.Tensor([self.args.node2id.get('PAD')]).long().to(self.args.device)
                    actionInds = torch.Tensor([self.args.max_children_num - 1]).long().to(self.args.device)
                    action_score = [score[-1]]
                    #continue

                else: # 孩子个数不足beam search的宽度
                    score = F.softmax(q_value[0][:children_len[i]])
                    action_value,actionInds=torch.topk(q_value[0][:children_len[i]],children_len[i])
                    actions=action_space[i][actionInds]
                    action_score, _ = torch.topk(score, children_len[i])

                action_values.append(action_value)
                action_scores.append(action_score)
                actions_k.append(actions)
                action_indexs.append(actionInds.data.cpu().numpy())
            return action_values,action_scores,actions_k,action_indexs

        else:
            # 随机选择出K个action(不重复)
            for i in range(len(action_space)):
                if children_len[i]==0: #没有孩子节点的话 直接选择第一个pad的节点
                    score = F.softmax(q_value[0])
                    action_value =[q_value[0][-1]]
                    actions = torch.Tensor([self.args.node2id.get('PAD')]).long().to(self.args.device)
                    actionInds = torch.Tensor([self.args.max_children_num-1]).long().to(self.args.device)
                    action_score = [score[-1]]
                    #continue

                else: # 从孩子节点中随机选择
                    score = F.softmax(q_value[0][:children_len[i]])
                    actionInds=torch.randperm(children_len[i])[:self.args.k]
                    actions=action_space[i][actionInds].to(self.args.device)
                    action_value=q_value[0][actionInds].to(self.args.device)
                    action_score = score[actionInds].to(self.args.device)

                action_values.append(action_value)
                action_scores.append(action_score)
                actions_k.append(actions)
                action_indexs.append(actionInds.data.cpu().numpy())
            return action_values,action_scores,actions_k,action_indexs

    def teacher_forcing(self,state,action_space,children_len,correctAction):
        # 针对的是一个样本
        q_value = self.eval_net(state)  #[B,72] # 第一个hops时是[1,101]，后面每个hop都是[5,101]

        # 直接选择出对的action
        # 先找出孩子的index
        actionInd=action_space[0].index(correctAction)
        action_value=q_value[0][actionInd]
        return actionInd,action_value

    # 同步current policy 与target net
    def update_target(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())


    def update(self):

        # 取出batch_size个样本
        state, action, reward, next_state, done = self.buffer.sample(self.args.batch_size)

        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
        #print('action:',action)
        action = Variable(torch.LongTensor(action)).to(self.args.device) #[5,5]
        reward = Variable(torch.FloatTensor(np.float32(reward))) #[5]
        done = Variable(torch.FloatTensor(done))

        # q_values,next_q_values,next_q_state_values:[5,1,72]-->[5,72]
        q_values = self.eval_net(state).squeeze(1)
        next_q_values = self.eval_net(next_state).squeeze(1)
        #print('next_q_values:', next_q_values)
        next_q_state_values = self.target_net(next_state).squeeze(1)
        #print('next_q_state_values:',next_q_state_values)


        #q_value:[5]
        q_value = q_values.gather(dim=1, index=action.unsqueeze(1)).squeeze(1) #[5]


        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1) #[5]
        expected_q_value = reward + 0.95 * next_q_value * (1 - done)

        loss = (q_value - Variable(expected_q_value)).pow(2).mean()
        return loss

class ReplayBuffer(object):
    def __init__(self,capacity):
        self.buffer=deque(maxlen=capacity)

    def push(self,state,action,reward,next_state,done):
        state=np.expand_dims(state,0)
        next_state=np.expand_dims(next_state,0)

        self.buffer.append((state,action,reward,next_state,done))

    def sample(self,batch_size):
        state,action,reward,next_state,done=zip(*random.sample(self.buffer,batch_size))
        return np.concatenate(state),action,reward,np.concatenate(next_state),done

    def __len__(self):
        return len(self.buffer)



