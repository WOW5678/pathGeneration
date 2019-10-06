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

USE_CUDA=torch.cuda.is_available()
device = ("cuda:1" if torch.cuda.is_available() else 'cpu')

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

    def act(self,state,action_space,children_len):
        # 针对的是一个样本
        action_values=[]
        actions_k=[]
        action_indexs=[]

        q_value = self.eval_net(state)  #[1,101] # 第一个hops时是[1,101]，后面每个hop都是[5,101]
        if random.random()<self.epsilon:
            # action_value:[1,5],actionInds:[1,5]
            for i in range(len(action_space)):
                # action_value shape:[1,5],actionIds shape:[1,5]
                #x=q_value[0][:children_len[i]]
                #print('x:',x.shape)

                # action shape,actionIds shape:[1,5]
                if children_len[i]>5:
                    print('children_len[i]:',children_len[i])
                    print('q_value[0][:children_len[i]]:',q_value[0][:children_len[i]])
                    action_value,actionInds=torch.topk(q_value[0][:children_len[i]],self.args.k)

                    print('action_value:',action_value) #[0.1316, 0.1185, 0.1068, 0.1000, 0.0992]
                    print('actionInds:', actionInds)  # [94,  6, 55, 16, 73]
                    actions = action_space[i][actionInds]
                    # print('actions:', actions)  # [8190, 4546, 8179, 8190, 5937]


                elif children_len[i]==0: # 说明就是叶子节点 则不需要选择action (使用action==0进行填充)
                    action_value=torch.zeros(1).float().to(self.args.device)
                    actions=torch.Tensor([0]).long().to(self.args.device)
                    actionInds=torch.Tensor([0]).long().to(self.args.device)

                else:
                    print('children_len[i]:', children_len[i])
                    print('q_value[0]:', q_value[0])
                    action_value,actionInds=torch.topk(q_value[0][:children_len[i]],children_len[i])

                    print('action_value:',action_value) #[0.1316, 0.1185, 0.1068, 0.1000, 0.0992]
                    print('actionInds:',actionInds) #[94,  6, 55, 16, 73]
                    actions=action_space[i][actionInds]
                    #print('actions:',actions) #[8190, 4546, 8179, 8190, 5937]

                action_values.append(action_value)
                actions_k.append(actions)
                action_indexs.append(actionInds.data.cpu().numpy())
            return action_values,actions_k,action_indexs

        else:
            # 随机选择出K个action(不重复)
            for i in range(len(action_space)):
                if children_len[i]==0:
                    action_value = torch.zeros(1).float().to(self.args.device)
                    actions = torch.Tensor([0]).long().to(self.args.device)
                    actionInds = torch.Tensor([0]).long().to(self.args.device)
                else:
                    actionInds=torch.randperm(children_len[i])[:self.args.k]
                    actions=action_space[i][actionInds].to(self.args.device)
                    action_value=q_value[0][actionInds].to(self.args.device)

                action_values.append(action_value)
                actions_k.append(actions)
                action_indexs.append(actionInds.data.cpu().numpy())
            return action_values,actions_k,action_indexs

    # 同步current policy 与target net
    def update_target(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def update(self):

        # 取出batch_size个样本
        state, action, reward, next_state, done = self.buffer.sample(self.args.batch_size)

        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
        print('action:',action)
        action = Variable(torch.LongTensor(action)).to(self.args.device) #[5,5]
        reward = Variable(torch.FloatTensor(np.float32(reward))) #[5]
        done = Variable(torch.FloatTensor(done))

        # q_values,next_q_values,next_q_state_values:[5,1,72]-->[5,72]
        q_values = self.eval_net(state).squeeze(1)
        next_q_values = self.eval_net(next_state).squeeze(1)
        print('next_q_values:', next_q_values)
        next_q_state_values = self.target_net(next_state).squeeze(1)
        print('next_q_state_values:',next_q_state_values)
        x = torch.max(next_q_values, dim=1)
        print('x:', x)

        #q_value:[5]
        index=action.unsqueeze(1)
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



