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
from gcn import GCNNet
from dqn import DQN,ReplayBuffer
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

        self.DQN = DQN(args)

        self.pathHist = []  # 保存着已经选择的路径（只保留最后的一个ICD）

        self.attn = nn.Linear(args.node_embedding_size * 4, args.node_embedding_size * 3)
        self.attn_combine = nn.Linear(args.node_embedding_size * 2, args.node_embedding_size)

        #self.atten=selfAttention(hidden_dim=args.node_embedding_size*4)

        # Attentional affine transformation
        # self.r_x = nn.Linear(args.node_embedding_size, args.node_embedding_size * 3)
        # nn.init.normal_(self.r_x.weight, mean=0, std=1)
        # self.b_x = nn.Linear(args.node_embedding_size, args.node_embedding_size * 3)
        # nn.init.normal_(self.b_x.weight, mean=0, std=1)

        self.r_x = nn.Parameter(torch.FloatTensor(args.node_embedding_size, args.node_embedding_size * 3))
        # 使用xaview_uniform_方法初始化权重
        nn.init.xavier_uniform_(self.r_x.data)  # (2,8285,16)
        self.b_x = nn.Parameter(torch.FloatTensor(args.node_embedding_size, args.node_embedding_size * 3))
        # nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_uniform_(self.b_x.data)  # (95,2)


        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    # 传入的是一个batch的数据量
    # K表示针对每个hop 选择出前K个最有可能的action
    def forward(self,ehrs,hier_labels,eval_flag):
        predPaths=[]
        batchStates=[]
        batchActionIndexs=[]
        batchPaths=[]
        batchDones=[]

        # 针对batch 中的每个样本(每个电子病历)进行单独的取样
        for i in range(len(ehrs)):
            example_states=[]
            example_done=[]
            example_actionIndexs=[]
            example_paths=[]

            # 首先初始化hidden
            hidden = torch.Tensor(np.zeros((1,self.args.path_hidden_size))).to(self.args.device)
            # 1.得到电子病历的表示
            ehrRrep = self.cnn(ehrs[i])        # 放在此处是为了每个样本都重置下环境
            self.sequences = [[[self.args.node2id.get('ROOT')], 1.0]] # 根节点

            for hop in range(self.args.hops):  # 从根节点到叶子节点的最大hop次数
                # 输入EHR的，得到G网络生成的路径

                if hop!=0:
                    hidden=hidden.sum(dim=1)

                # 2.得到attentive的EHR表示
                #atten_weights = F.softmax(self.attn(torch.cat((hidden, ehrRrep), 1))) #[1,300]
                # attn_ehrRrep = torch.mul(atten_weights, ehrRrep)  # [1,300]
                #attn_ehrRrep=self.atten(torch.cat((hidden,ehrRrep),1))
                # print('hidden:',hidden)
                state=F.relu(self.attn(torch.cat((hidden, ehrRrep), 1)))
                # print('ehrRrep:',ehrRrep)
                # print('attn_ehrRrep:', attn_ehrRrep)
                #
                # # 3.得到融合了EHR和路径信息的表示,即state的表示
                # #state = self.r_x(hidden) * attn_ehrRrep + self.b_x(hidden)  # [32, 300]， state:[1,300]
                # state = torch.mm(hidden,self.r_x)+ attn_ehrRrep
                #print('state:',state)

                # 4.首先获得当前跳的action_space空间 然后DQN根据state在该空间中选择action
                if hop==0:
                    children = torch.Tensor(self.level2_parients).long().to(self.args.device)
                    children_len=torch.Tensor([self.args.max_children_num]).long().to(self.args.device)
                else:
                    # selected_action作为父节点 选择对应的孩子节点
                    children,children_len = action_space(selected_action, self.args)   # action:[32]

                # print('children.shape:',children.shape)        #[1, 101]
                # 在选择action 之前 也要执行以下判断，如果children_len中包含有0 说明选择出了叶子节点

                action_values,action_scores,actions,actionIndexs = self.DQN.act(state,children,children_len,eval_flag)
                #print('hop:',hop)
                # print('actions:',actions)

                # 使用action_scores进行路径解码（因为是概率值） 而不是Q_value值
                selected_action, actionList=self.beam_search(action_scores,actions)


                # 4.将当前选择的节点和上一步选择的节点（保存在self.actionList中） 输入到path encoder中得到路径的表示
                path = self.pathEncoder(selected_action,actionList)

                # 5.将路径表示输入到path Decoder中以更新hidden的表示

                output= self.pathDecoder( path.unsqueeze(0))
                hidden=self.pathDecoder.hidden

                if hop==3:
                    done=True
                else:
                    done=False

                example_states.append(state) # state:[1,300]

                example_done.append(done)
                example_actionIndexs.append(actionIndexs)
                example_paths.append(path) # path:[10,100]

            # 最后一个hop之后得到的state(训练时要使用)
            hidden = hidden.sum(dim=1)
            state = F.relu(self.attn(torch.cat((hidden, ehrRrep), 1)))
            example_states.append(state)


            predPaths.append(actionList)
            batchStates.append(example_states)
            batchActionIndexs.append(example_actionIndexs)
            batchPaths.append(example_paths)
            batchDones.append(example_done)

        return predPaths,batchStates,batchActionIndexs,batchPaths,batchDones


    # 传入真实的action 得到真实的（state,path）对
    def teacher_force(self,ehrs,hier_labels):
        # 对hier_labels进行填充
        for i in range(len(hier_labels)):
            for j in range(len(hier_labels[i])):
                if len(hier_labels[i][j])<4:
                    hier_labels[i][j]=hier_labels[i][j]+[0]*(4-len(hier_labels[i][j]))
            # if len(hier_labels[i])<self.args.k: # 标签的个数小于K
            #     for i in range(self.args.k-len(hier_labels[i])):
            #         hier_labels[i].append([0]*4)

        print('hier_labels:',hier_labels)

        batchStates=torch.zeros((len(ehrs),3,self.args.path_hidden_size*3)).to(self.args.device)
        batchPaths=torch.zeros((len(ehrs),3,self.args.path_hidden_size)).to(self.args.device)
        # 针对batch 中的每个样本(每个电子病历)进行单独的取样
        for i in range(len(ehrs)):
            example_states = torch.zeros((3,self.args.path_hidden_size*3))
            example_paths = torch.zeros((3,self.args.path_hidden_size))

            # 首先初始化hidden
            hidden = torch.Tensor(np.zeros((1, self.args.path_hidden_size))).to(self.args.device)
            # 1.得到电子病历的表示
            ehrRrep = self.cnn(ehrs[i])  # 放在此处是为了每个样本都重置下环境
            self.sequences = [[[self.args.node2id.get('ROOT')], 1.0]]  # 根节点

            for hop in range(self.args.hops):  # 从根节点到叶子节点的最大hop次数
                # 输入EHR的，得到G网络生成的路径

                if hop != 0:
                    hidden = hidden.sum(dim=1)

                # 2.得到attentive的EHR表示
                # atten_weights = F.softmax(self.attn(torch.cat((hidden, ehrRrep), 1))) #[1,300]
                # attn_ehrRrep = torch.mul(atten_weights, ehrRrep)  # [1,300]
                # attn_ehrRrep=self.atten(torch.cat((hidden,ehrRrep),1))
                # print('hidden:',hidden)
                state = F.relu(self.attn(torch.cat((hidden, ehrRrep), 1)))

                # 4.首先获得当前跳的action_space空间 然后DQN根据state在该空间中选择action

                if hop>0:
                    actionList=selected_action
                else:
                    actionList=[self.args.node2id.get('ROOT') for i in range(self.args.k)]

                selected_action = [row[hop] for row in hier_labels[i]]
                if len(selected_action) < self.args.k:
                    selected_action = selected_action + [0] * (self.args.k - len(selected_action))
                else:
                    selected_action=selected_action[:self.args.k]
                # 其实截断是不应该的 应该所有正确的标签都应该考虑进去

                # 4.将当前选择的节点和上一步选择的节点（保存在self.actionList中） 输入到path encoder中得到路径的表示
                # print('selected_action:',selected_action)
                # print('actionList:',actionList)
                path = self.pathEncoder(selected_action, actionList,teacher_force=True)

                # 5.将路径表示输入到path Decoder中以更新hidden的表示

                output = self.pathDecoder(path.unsqueeze(0))
                hidden = self.pathDecoder.hidden

                example_states[hop]=state.squeeze(0)  # state:[1,300]
                example_paths[hop]=torch.sum(path,dim=0)  # path:[10,100]
            batchStates[i]=example_states
            batchPaths[i]=example_paths

        return batchStates,batchPaths

    def beam_search(self,data,actions_store):
        all_candidates=list()
        for sequence,row, actions in zip(self.sequences,data,actions_store):
            seq,score=sequence

            for j in range(len(row)):
                # if actions[j].item() in self.args.leafNodes:
                #     continue
                candidate=[seq+[actions[j].item()],score *-math.log(row[j].item())]
                all_candidates.append(candidate)

            # order all candidates by scores
            ordered=sorted(all_candidates,key=lambda tup:tup[1],reverse=True)
            # select k best
            self.sequences=ordered[:self.args.k]
            #print('self.sequences:',self.sequences)
        selected_action=[row[0][-1] for row in self.sequences]
        #print('selected_action:',selected_action)
        selected_path=[row[0] for row in self.sequences]
        return selected_action,selected_path

    def step(self,actionList,hier_label,hop):
        # 对比当前预测出的路径以及真实的路径 设定对应的reward
        hop_tures=[]
        for row in hier_label:
            hop_tures.extend(row)

        if hop==3:
            done=True
        else:
            done=False
        for row in actionList:
            if row[hop+1] in hop_tures:
                reward=1
                return reward,done
            else:
                reward=-1
                return reward,done

    def update_buffer(self,batch_states,batch_actionIndexs,batch_rewards,batch_done):
        for example_states,example_actionIndexs,example_rewards,example_done in zip(batch_states,batch_actionIndexs,batch_rewards,batch_done):
            # 将这些数据转换成（state,action,reward,next_state,done）保存到memory中
            for i in range(len(example_states)):
                example_states[i] = example_states[i].data.cpu().numpy()

            for i in range(len(example_rewards)):
                for j in range(len(example_actionIndexs[i])):
                    self.DQN.buffer.push(example_states[i], example_actionIndexs[i][j][0], example_rewards[i],
                                         example_states[i + 1], example_done[i])


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
        x=torch.Tensor([x]).long().to(self.args.device)
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

    def forward(self,current_node,actionList,teacher_force=False):
        #通过两个节点的表示得到路径的表示
        # print(type(self.embedding.weight)) #'torch.nn.parameter.Parameter'
        # print(type(self.gcn(g,self.embedding.weight))) # 'torch.Tensor'
        #print('g:', g)
        # 从node embedding列表中选择出对应节点的embedding
        # print('current_node:',current_node) #[32]
        # print([row[-1] for row in action_list])
        if teacher_force:
            last_node=torch.Tensor(actionList).long().to(self.args.device)
        else:
            last_node=torch.Tensor([row[-2] for row in actionList]).long().to(self.args.device)

        #print('last_node_id:', last_node)
        current_node=torch.Tensor(current_node).long().to(self.args.device)
        #print('current_node_id:', current_node)
        current_node=self.embedding(current_node)
        last_node=self.embedding(last_node)

        # print('current_node:',current_node.shape) # [1,5,100]
        # print('last_node:',last_node.shape) #[5,1,100]
        #通过两个节点最终得到输入路径的表示
        path=self.w_path(torch.cat((current_node,last_node),dim=1))
        return path

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
        # print('input:',input.shape)
        output,self.hidden=self.gru(input,self.hidden)
        #print('after-self.hidden:', self.hidden.shape)
        return output

    def initHidden(self,args):
        hidden = torch.zeros((1,self.args.k,args.path_hidden_size))
        hidden = torch.Tensor(hidden).to(args.device)
        return hidden

class selfAttention(nn.Module):
    def __init__(self,hidden_dim):
        super(selfAttention, self).__init__()
        self.hidden_dim=hidden_dim
        self.projection=nn.Sequential(
            nn.Linear(hidden_dim,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,1)
        )
    def forward(self,encoder_outputs):

        #(L,H)-->(L,1)
        energy=self.projection(encoder_outputs)
        weights=F.softmax(energy.squeeze(-1),dim=-1)
        #(L,1)*(L,1)--> (B,H)
        outputs=(encoder_outputs*weights.unsqueeze(-1)).sum(dim=1)
        return outputs

def action_space(parients,args):
    # 根据parient 找到属于他的孩子节点 注意是一个batch的样本
    # 直接从adj中找出 该parient对应的行
    # parient:[1448,1881,5284,3002,4694]
    #adj=torch.Tensor(args.adj).to(args.device) #[9140,9140]
    children=[torch.nonzero(row) for row in args.adj[parients]]
    children_len = [len(row) for row in children]
    for i in range(len(children)):
        children[i]=[item[0].cpu().item() for item in children[i]]
        #mask
        children[i]=children[i]+[0]*(args.max_children_num-len(children[i]))
    #print('children:',children)
    #转换成tensor对象
    children=torch.Tensor(children).long()
    children_len=torch.Tensor(children_len).long()
    return children,children_len



def run_pre_train_step(gen_model,current_batch):
    ehrs=[example.ehr for example in current_batch]
    labels=[example.labels for example in current_batch]
    hier_labels=[example.hier_labels for example in current_batch]
    batchPathes=gen_model(ehrs,hier_labels)
    return labels,batchPathes



def run_train_step(gen_model,current_batch,d_model):
    ehrs = [example.ehr for example in current_batch]
    labels = [example.labels for example in current_batch]
    hier_labels = [example.hier_labels for example in current_batch]

    # 从g_model中生成state,action
    predPaths, batchStates, batchActionIndexs, batchPaths, batchDones= gen_model(ehrs, hier_labels,eval_flag=False)

    # 从d_model中得到rewards值
    batchStates_=[row[:-1] for row in batchStates]
    batchRewards=d_model.forward(batchStates_, batchPaths)
    # 将五元组信息加入到buffer中 供g_model的训练使用

    # 将agent的尝试放入到buffer中
    gen_model.update_buffer(batchStates, batchActionIndexs, batchRewards, batchDones)

    return labels,predPaths,batchStates_,batchPaths

def run_eval_step(gen_model, current_batch):
    ehrs = [example.ehr for example in current_batch]
    labels = [example.labels for example in current_batch]
    hier_labels = [example.hier_labels for example in current_batch]

    # 从g_model中生成state,action
    predPaths, batchStates, batchActionIndexs, batchPaths, batchDones = gen_model(ehrs, hier_labels,eval_flag=True)
    # 只需要返回预测的label路径
    return labels, predPaths


def generated_positive_samples(gen_model,current_batch):
    ehrs = [example.ehr for example in current_batch]
    hier_labels = [example.hier_labels for example in current_batch]

    # 从g_model中生成positive  state,path对
    batchStates,batchPaths = gen_model.teacher_force(ehrs, hier_labels)
    return batchStates,batchPaths





