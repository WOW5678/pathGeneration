# -*- coding:utf-8 -*-
"""
@Time: 2019/09/25 15:02
@Author: Shanshan Wang
@Version: Python 3.7
@Function: main file
"""

import torch
import numpy as np
import random
import argparse
import os
import data
from batcher import GenBatcher,DisBatcher
import generator
from generator import Generator
import full_eval
from discriminator import Discriminator
import discriminator
import utils
import csv
import matplotlib.pyplot as plt
import pandas as pd


from keras.utils import to_categorical
from torch import autograd
import random
from modelUtils import ReplayBuffer
import  torch.nn.functional as F
# 指定运行的显卡
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

#设定随机种子
seed=2019
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# 设定一些参数
PARSER=argparse.ArgumentParser(description='the code of path generation')
PARSER.add_argument('-data_dir','--data_dir',default='data',type=str)
PARSER.add_argument('-vocab_size', '--vocab_size', default=50000, type=int,
                    help='Size of vocabulary')

PARSER.add_argument('-num_epochs', '--num_epochs', default=200, type=int, help='num_epochs')
PARSER.add_argument('-batch_size', '--batch_size', default=64, type=int, help='batch size')
PARSER.add_argument('-lr', '--lr', default=0.001, type=float, help='learning rate')
PARSER.add_argument('-hops', '--hops', default=4, type=int, help='number of hops')
PARSER.add_argument('-max_children_num', '--max_children_num', default=129, type=int, help='max number of children')
PARSER.add_argument('-padded_len_ehr', '--padded_len_ehr', default=200, type=int, help='padded length of ehr')


# CNN模块参数
PARSER.add_argument('-cnn_embedding_size', '--cnn_embedding_size', default=100, type=int, help='the embedding size of CNN Network')
PARSER.add_argument('-num_filter_maps', '--num_filter_maps', default=100, type=int, help='the num of filters of CNN Network')
PARSER.add_argument('-dropout_rate', '--dropout_rate', default=0.5, type=float, help='the dropout rate of  CNN Network')

# pathEncoder模块中的参数
PARSER.add_argument('-node_embedding_size', '--node_embedding_size', default=100, type=int, help='the embedding size of each node (GNN)')

# pathDecoder 模块中的参数
PARSER.add_argument('-path_hidden_size', '--path_hidden_size', default=100, type=int, help='the hidden size of each path (LSTM)')

# DNQ模块中的参数
PARSER.add_argument('-state_size', '--state_size', default=300, type=int, help='the size of state')
PARSER.add_argument('-k', '--k', default=10, type=int, help='the top k paths')

# D 模块中的参数
PARSER.add_argument('-d_hidden_size','--d_hidden_size',default=400,type=int,help='the hidden size of D model')
PARSER.add_argument('-mode','--mode',default='MSE',type=str,help='the type of loss')

args=PARSER.parse_args()
print(args)

args.device=('cuda:3' if torch.cuda.is_available() else 'cpu')
Variable=lambda *args,**kwargs:autograd.Variable(*args,**kwargs).to(args.device)

# 预训练 G模型
# 使用启发式reward训练G网络
def pre_train_generator(gen_model,d_model,batcher,max_run_epoch):
    epoch=0
    counter=0
    while epoch <max_run_epoch:
        batches=batcher.get_batches(mode='train')
        step=0
        print('batches number:',len(batches))
        while step<len(batches):
            current_batch=batches[step]
            step+=1
            labels, batchPathes=generator.run_pre_train_step(gen_model,current_batch)
            #print('batchPathes:',batchPathes)
            #计算出评估指标
            predicted_labels=[]
            for sample,label in zip(batchPathes,labels):
                #print('sample:',[args.id2node.get(item) for row in sample for item in row])
                [row.reverse() for row in sample]
                # next(x for x in row if x >0) for row in sample)
                pred=[next(x for x in row if x >0) for row in sample]
                #print('sample:',sample)
                # print('pred:', [args.id2node.get(item) for item in pred])
                #
                # print('label:',[args.id2node.get(item) for item in label])
                predicted_labels.append(pred)

            batchJaccard=[full_eval.jaccrad(pred,label) for pred,label in zip(predicted_labels,labels)]
            avgJaccard=sum(batchJaccard)*1.0/len(batchJaccard)
            print('avgJaccard:',avgJaccard)

            if len(gen_model.DQN.buffer)>args.batch_size:
                #print('updating model............')
                #训练模型
                loss=gen_model.DQN.update()+d_model()
                print('loss:',loss.item())
                gen_model.optimizer.zero_grad()
                loss.backward()
                gen_model.optimizer.step()
                counter+=1

            # 当达到训练到一定次数之后 更新target DQN
            if counter%20==0:
                gen_model.DQN.update_target()

# 预训练 D模型
def pre_train_discriminator(d_model,d_batcher,max_run_epoch):
    pass

# 训练 G模型
def train_generator(gen_model_eval,gen_model_target,d_model,batchStates,batchHiddens,buffer,mode):

    if len(buffer)>args.batch_size:
        #print('updating model............')
        #训练模型
        #+discriminator.g_loss(d_model, batchStates, batchHiddens,mode)
        loss=RLLoss(gen_model_eval,gen_model_target,buffer)
        #loss = RLLoss(gen_model_eval, gen_model_target, buffer)
        gen_model_eval.learn_step_counter += 1

        #当达到训练到一定次数之后 更新target DQN
        if gen_model_eval.learn_step_counter%20==0:
            update_target(gen_model_eval,gen_model_target)
        #每100轮之后降低explore rate
        if gen_model_eval.learn_step_counter%1==0:
            gen_model_eval.updateEpision()
            print('gen_model_eval.epsilon:',gen_model_eval.epsilon)
            print(len(buffer))

        clean_grad(gen_model_eval, d_model)
        loss.backward()
        gen_model_eval.optimizer.step()

        # for name, param in gen_model_eval.named_parameters():
        #     print('name', name)
        #     print('param.grad:', param.grad)
        return loss.item()


# 训练 D模型
def train_discriminator(d_model,batchStates_n,batchPaths_n,batchStates_p,batchPaths_p,mode):

    loss=discriminator.train_d_model(d_model,batchStates_n,batchPaths_n,batchStates_p,batchPaths_p,mode)
    return loss

def evaluate(g_batcher,gen_model,d_model,buffer,writer,flag='valid'):
    #加载验证集中所有的数据
    batches = g_batcher.get_batches(mode=flag)
    batch_micro_p,batch_macro_p,batch_micro_r,batch_macro_r,batch_micro_f1,batch_macro_f1,batch_micro_auc_roc,batch_macro_auc_roc=[],[],[],[],[],[],[],[]

    for batch_num in range(len(batches)):
        current_batch=batches[batch_num]

        labels, predHierLabels=generator.run_eval_step(gen_model, d_model,buffer,current_batch)
        # 对labels 和predPaths进行预处理 整理成full_eval中需要的格式 直接得到需要的评估指标

        predicted_labels = []
        onehot_labels=[]
        for sample, label in zip(predHierLabels, labels):
            #print('sample:',[args.id2node.get(item) for row in sample for item in row])
            pred = [[i for i in row if i > 0][-1] for row in sample]
            pred=list(set(pred))
            label=list(set(label))
            print('pred:',pred)
            # print('label:',label)
            pred=sum(to_categorical(pred,num_classes=len(args.node2id)))
            label=sum(to_categorical(label,num_classes=len(args.node2id)))
            predicted_labels.append(pred.tolist())
            onehot_labels.append(label.tolist())

        #计算这一个batch的评估指标
        micro_p, macro_p, micro_r, macro_r, micro_f1, macro_f1, micro_auc_roc, macro_auc_roc=full_eval.full_evaluate(predicted_labels,onehot_labels)

        print('batch_number:{}, micro_p:{:.4f}, macro_p:{:.4f}, micro_r:{:.4f}, macro_r:{:.4f}, micro_f1:{:.4f}, macro_f1:{:.4f}, micro_auc_roc:{:.4f}, macro_auc_roc:{:.4f}'.format(batch_num,micro_p, macro_p, micro_r, macro_r, micro_f1, macro_f1, micro_auc_roc, macro_auc_roc))
        batch_micro_p.append(micro_p)
        batch_macro_p.append(macro_p)
        batch_micro_r.append(micro_r)
        batch_macro_r.append(macro_r)
        batch_micro_f1.append(micro_f1)
        batch_macro_f1.append(macro_f1)
        batch_micro_auc_roc.append(micro_auc_roc)
        batch_macro_auc_roc.append(macro_auc_roc)


    avg_micro_p=sum(batch_micro_p)*1.0/len(batch_micro_p)
    avg_macro_p=sum(batch_macro_p)*1.0/len(batch_macro_p)
    avg_micro_r=sum(batch_micro_r)*1.0/len(batch_micro_r)
    avg_macro_r=sum(batch_macro_r)*1.0/len(batch_macro_r)
    avg_micro_f1=sum(batch_micro_f1)*1.0/len(batch_micro_f1)
    avg_macro_f1=sum(batch_macro_f1)*1.0/len(batch_macro_f1)
    avg_micro_auc_roc=sum(batch_micro_auc_roc)*1.0/len(batch_micro_auc_roc)
    avg_macro_auc_roc=sum(batch_macro_auc_roc)*1.0/len(batch_macro_auc_roc)

    # 取平均值为最终的结果
    print( 'avg_micro_p:{:.4f},avg_macro_p:{:.4f}, avg_micro_r:{:.4f}, avg_macro_r:{:.4f}, avg_micro_f1:{:.4f}, avg_macro_f1:{:.4f}, avg_micro_auc_roc:{:.4f}, avg_macro_auc_roc:{:.4f}'.format(
        avg_micro_p, avg_macro_p,avg_micro_r,
        avg_macro_r, avg_micro_f1,avg_macro_f1,
        avg_micro_auc_roc, avg_macro_auc_roc))
    writer.writerow([avg_micro_p, avg_macro_p,avg_micro_r,avg_macro_r, avg_micro_f1,avg_macro_f1,avg_micro_auc_roc, avg_macro_auc_roc])
    return avg_micro_f1

# 同步current policy 与target net
def update_target(eval_net,target_net):
    #print('update target net.....')
    target_net.load_state_dict(eval_net.state_dict())

def RLLoss(gen_model_eval,gen_model_target,buffer):

    # 取出batch_size个样本
    # state:[64,300],action:64个元素的元组，reward:[64,3],done:64个元素的元组 每个元素是个list
    ehrs, paths, selected_action, next_paths, actionIndexs,reward, done= buffer.sample(args.batch_size)


    ehrs = torch.Tensor(ehrs).long().to(args.device) #[64,200]
    reward =torch.Tensor(reward).float().squeeze(-1).to(args.device) #[64]
    done = torch.Tensor(done).float().to(args.device) #[64]

    actionIndexs=torch.Tensor(actionIndexs).long().to(args.device) #[64,1]
    q_value = gen_model_eval.calcuateValue_eval(ehrs,paths,actionIndexs,eval_flag=True).squeeze(-1) #[64,1]
    next_q_value = gen_model_target.calcuateValue_target(ehrs,next_paths,eval_flag=True).squeeze(-1) #[64,1]

    expected_q_value = reward + 0.9 * next_q_value * (1 - done)  # [64]
    # 对每个值都计算一个权重  因为
    #print(q_value.shape,expected_q_value.shape)
    loss=F.smooth_l1_loss(q_value,expected_q_value)
    return loss

def clean_grad(g_model,d_model):
    g_model.optimizer.zero_grad()
    d_model.optimizer.zero_grad()

def main():
    ################################
    ## 第一模块：数据准备工作
    data_=data.Data(args.data_dir, args.vocab_size)

    # 对ICD tree 处理
    parient_children, level2_parients,leafNodes, adj,node2id, hier_dicts= utils.build_tree(os.path.join(args.data_dir,'note_labeled_v2.csv'))
    graph = utils.generate_graph(parient_children, node2id)
    args.node2id=node2id
    args.id2node={id:node for node,id in node2id.items()}
    args.adj=torch.Tensor(adj).long().to(args.device)
    # args.leafNodes=leafNodes
    args.hier_dicts=hier_dicts
    # args.level2_parients=level2_parients
    #print('836:',args.id2node.get(836),args.id2node.get(0))

    # TODO batcher对象的细节
    g_batcher=GenBatcher(data_,args)

    #################################
    ## 第二模块： 创建G模型，并预训练 G模型
    # TODO Generator对象的细节
    gen_model_eval=Generator(args,data_,graph,level2_parients)
    gen_model_target=Generator(args,data_,graph,level2_parients)
    gen_model_target.eval()
    print(gen_model_eval)

    # for name,param in gen_model_eval.named_parameters():
    #     print(name,param.size(),type(param))
    buffer=ReplayBuffer(capacity=100000)
    gen_model_eval.to(args.device)
    gen_model_target.to(args.device)

    # TODO generated 对象的细节

    # 预训练 G模型
    #pre_train_generator(gen_model,g_batcher,10)

    #####################################
    ## 第三模块： 创建 D模型，并预训练 D模型
    d_model=Discriminator(args)
    d_model.to(args.device)


    # 预训练 D模型
    #pre_train_discriminator(d_model,d_batcher,25)

    ########################################
    ## 第四模块： 交替训练G和D模型

    #将评估结果写入文件中
    f=open('valid_result.csv','w')
    writer=csv.writer(f)
    writer.writerow(['avg_micro_p', 'avg_macro_p','avg_micro_r,avg_macro_r', 'avg_micro_f1','avg_macro_f1','avg_micro_auc_roc', 'avg_macro_auc_roc'])
    epoch_f=[]
    for epoch in range(args.num_epochs):
        batches=g_batcher.get_batches(mode='train')
        print('number of batches:',len(batches))
        for step in range(len(batches)):
            #print('step:',step)
            current_batch = batches[step]
            ehrs = [example.ehr for example in current_batch]
            ehrs = torch.Tensor(ehrs).long().to(args.device)

            hier_labels = [example.hier_labels for example in current_batch]

            true_labels = []


            # 对hier_labels进行填充
            for i in range(len(hier_labels)):  # i为样本索引
                for j in range(len(hier_labels[i])):  # j为每个样本的每条路径索引
                    if len(hier_labels[i][j]) < 4:
                        hier_labels[i][j] = hier_labels[i][j] + [0] * (4 - len(hier_labels[i][j]))
                # if len(hier_labels[i]) < args.k:
                #     for time in range(args.k - len(hier_labels[i])):
                #         hier_labels[i].append([0] * args.hops)

            for sample in hier_labels:
                #print('sample:',sample)
                true_labels.append([row[1] for row in sample])

            predHierLabels,batchStates_n, batchHiddens_n = generator.generated_negative_samples(gen_model_eval,d_model,ehrs,hier_labels,buffer)

            #true_labels = [example.labels for example in current_batch]

            _, _, avgJaccard=full_eval.process_labels(predHierLabels,true_labels,args)


            # G生成训练D的positive samples
            batchStates_p, batchHiddens_p = generator.generated_positive_samples(gen_model_eval,ehrs,hier_labels,buffer)


            # 训练 D网络
            #d_loss=train_discriminator(d_model,batchStates_n,batchHiddens_n,batchStates_p,batchHiddens_p,mode=args.mode)


            # 训练 G模型
            #for g_epoch in range(10):
            g_loss=train_generator(gen_model_eval, gen_model_target, d_model, batchStates_n, batchHiddens_n,buffer,mode=args.mode)

            print('batch_number:{}, avgJaccard:{:.4f}, g_loss:{:.4f}'.format(step, avgJaccard,g_loss))

        # #每经过一个epoch 之后分别评估G 模型的表现以及D模型的表现（在验证集上的表现）
        avg_micro_f1=evaluate(g_batcher,gen_model_eval,d_model,buffer,writer,flag='valid')
        epoch_f.append(avg_micro_f1)

    # 画图
    # plot results
    window = int(args.num_epochs / 20)
    print('window:',window)
    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9]);
    rolling_mean = pd.Series(epoch_f).rolling(window).mean()
    std = pd.Series(epoch_f).rolling(window).std()
    ax1.plot(rolling_mean)
    ax1.fill_between(range(len(epoch_f)), rolling_mean - std, rolling_mean + std, color='orange',
                     alpha=0.2)
    ax1.set_title('Episode Length Moving Average ({}-episode window)'.format(window))
    ax1.set_xlabel('Epoch Number');
    ax1.set_ylabel('F1')

    ax2.plot(epoch_f)
    ax2.set_title('Performance on valid set')
    ax2.set_xlabel('Epoch Number');
    ax2.set_ylabel('F1')

    fig.tight_layout(pad=2)
    plt.show()
    fig.savefig('results.png')

    f.close()
if __name__ == '__main__':
    main()
