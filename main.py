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


from keras.utils import to_categorical
# 指定运行的显卡
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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
PARSER.add_argument('-vocab_size', '--vocab_size', default=5000, type=int,
                    help='Size of vocabulary')

PARSER.add_argument('-num_epochs', '--num_epochs', default=10, type=int, help='num_epochs')
PARSER.add_argument('-batch_size', '--batch_size', default=64, type=int, help='batch size')
PARSER.add_argument('-lr', '--lr', default=0.0001, type=float, help='learning rate')
PARSER.add_argument('-hops', '--hops', default=3, type=int, help='number of hops')
PARSER.add_argument('-max_children_num', '--max_children_num', default=228, type=int, help='max number of children')
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
args=PARSER.parse_args()
print(args)

args.device=('cuda:2' if torch.cuda.is_available() else 'cpu')

# 预训练 G模型
# 使用启发式reward训练G网络
def pre_train_generator(gen_model,batcher,max_run_epoch):
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
                loss=gen_model.DQN.update()
                print('loss:',loss.item())
                gen_model.optimizer.zero_grad()
                loss.backward()
                gen_model.optimizer.step()
                counter+=1

            # 当达到训练到一定次数之后 更新target DQN
            if counter%100==0:
                gen_model.DQN.update_target()


# 预训练 D模型
def pre_train_discriminator(d_model,d_batcher,max_run_epoch):
    pass

# 训练 G模型
def train_generator(gen_model,d_model,current_batch):
    counter=0
    for epoch in range(2):
        labels, predHierLabels,batchStates,batchPaths=generator.run_train_step(gen_model,current_batch,d_model)
        #print('batchPathes:',batchPathes)
        #计算出评估指标
        predicted_labels=[]
        for sample,label in zip(predHierLabels,labels):
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


        if len(gen_model.DQN.buffer)>args.batch_size:
            #print('updating model............')
            #训练模型
            loss=gen_model.DQN.update()
            print('train G model, epoch:{},loss:{},avgJaccard:{}'.format(epoch,loss.item(),avgJaccard))
            gen_model.optimizer.zero_grad()
            loss.backward()
            gen_model.optimizer.step()
            counter+=1

        # 当达到训练到一定次数之后 更新target DQN
        if counter%500==0:
            gen_model.DQN.update_target()

    return batchStates, batchPaths

# 训练 D模型
def train_discriminator(d_model,batchStates_n,batchPaths_n,batchStates_p,batchPaths_p):
    for epoch in range(2):
        print('train D model,epoch：',epoch)
        discriminator.train_d_model(d_model,batchStates_n,batchPaths_n,batchStates_p,batchPaths_p)

    # 统计D_model的表现
    # TODO

def evaluate(g_batcher,gen_model,flag='valid'):
    #加载验证集中所有的数据
    batches = g_batcher.get_batches(mode=flag)
    batch_micro_p,batch_macro_p,batch_micro_r,batch_macro_r,batch_micro_f1,batch_macro_f1,batch_micro_auc_roc,batch_macro_auc_roc=[],[],[],[],[],[],[],[]
    for batch_num in range(len(batches)):
        current_batch=batches[batch_num]
        labels, predHierLabels=generator.run_eval_step(gen_model, current_batch)
        # 对labels 和predPaths进行预处理 整理成full_eval中需要的格式 直接得到需要的评估指标

        predicted_labels = []
        onehot_labels=[]
        for sample, label in zip(predHierLabels, labels):
            # print('sample:',[args.id2node.get(item) for row in sample for item in row])
            [row.reverse() for row in sample]
            # next(x for x in row if x >0) for row in sample)
            pred = [next(x for x in row if x > 0) for row in sample]
            pred=list(set(pred))
            label=list(set(label))
            pred=sum(to_categorical(pred,num_classes=len(args.node2id)))
            label=sum(to_categorical(label,num_classes=len(args.node2id)))
            predicted_labels.append(pred.tolist())
            onehot_labels.append(label.tolist())

        #计算这一个batch的评估指标
        micro_p, macro_p, micro_r, macro_r, micro_f1, macro_f1, micro_auc_roc, macro_auc_roc=full_eval.full_evaluate(predicted_labels,onehot_labels)

        print('batch_number:{},micro_p:{},macro_p:{}, micro_r:{}, macro_r:{}, micro_f1:{}, macro_f1:{}, micro_auc_roc:{}, macro_auc_roc:{}'.format(batch_num,micro_p, macro_p, micro_r, macro_r, micro_f1, macro_f1, micro_auc_roc, macro_auc_roc))
        batch_micro_p.append(micro_p)
        batch_macro_p.append(macro_p)
        batch_micro_r.append(micro_r)
        batch_macro_r.append(macro_r)
        batch_micro_f1.append(micro_f1)
        batch_macro_f1.append(macro_f1)
        batch_micro_auc_roc.append(micro_auc_roc)
        batch_macro_auc_roc.append(macro_auc_roc)

    # 取平均值为最终的结果
    print( 'avg_micro_p:{},avg_macro_p:{}, avg_micro_r:{}, avg_macro_r:{}, avg_micro_f1:{}, avg_macro_f1:{}, avg_micro_auc_roc:{}, avg_macro_auc_roc:{}'.format(
        sum(batch_micro_p)*1.0/len(batch_micro_p), sum(batch_macro_p)*1.0/len(batch_macro_p),sum(batch_micro_r)*1.0/len(batch_micro_r),
        sum(batch_macro_r)*1.0/len(batch_macro_r), sum(batch_micro_f1)*1.0/len(batch_micro_f1),sum(batch_macro_f1)*1.0/len(batch_macro_f1),
        sum(batch_micro_auc_roc)*1.0/len(batch_micro_auc_roc), sum(batch_macro_auc_roc)*1.0/len(batch_macro_auc_roc)))

def main():
    ################################
    ## 第一模块：数据准备工作
    data_=data.Data(args.data_dir, args.vocab_size)

    # 对ICD tree 处理
    parient_children, level2_parients,leafNodes, adj,node2id, hier_dicts= utils.build_tree(os.path.join(args.data_dir,'note_labeled.csv'))
    graph = utils.generate_graph(parient_children, node2id)
    args.node2id=node2id
    args.id2node={id:node for node,id in node2id.items()}
    args.adj=torch.Tensor(adj).long().to(args.device)
    args.leafNodes=leafNodes
    args.hier_dicts=hier_dicts


    # TODO batcher对象的细节
    g_batcher=GenBatcher(data_,args)

    
    #################################
    ## 第二模块： 创建G模型，并预训练 G模型
    # TODO Generator对象的细节
    gen_model=Generator(args,data_,graph,level2_parients)

    gen_model.to(args.device)
    # TODO generated 对象的细节

    # 预训练 G模型
    #pre_train_generator(gen_model,g_batcher,10)

    # 利用G 生成一些negative samples
    # generated.generator_train_negative_samples()
    # generated.generator_test_negative_samples()

    #####################################
    ## 第三模块： 创建 D模型，并预训练 D模型
    d_model=Discriminator(args)
    d_model.to(args.device)
    d_batcher=DisBatcher(data_,args)

    # 预训练 D模型
    pre_train_discriminator(d_model,d_batcher,25)

    ########################################
    ## 第四模块： 交替训练G和D模型
    for epoch in range(args.num_epochs):
        batches=g_batcher.get_batches(mode='train')
        print('number of batches:',len(batches))
        for step in range(len(batches)):
            print('step:',step)
            current_batch = batches[step]
            #训练 G模型
            batchStates_n,batchPaths_n=train_generator(gen_model,d_model,current_batch)

            # 生成训练D的negative samples

            # 生成训练D的positive samples
            batchStates_p,batchPaths_p=generator.generated_positive_samples(gen_model,current_batch)
            # 生成测试样本

            # 训练 D网络
            train_discriminator(d_model,batchStates_n,batchPaths_n,batchStates_p,batchPaths_p)
        #每经过一个epoch 之后分别评估G 模型的表现以及D模型的表现（在验证集上的表现）
        evaluate(g_batcher,gen_model,flag='valid')

if __name__ == '__main__':
    main()
