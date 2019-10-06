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
from batcher import GenBatcher
import generator
from generator import Generator
from generated_example import Generated_example

import utils
import time


# 指定运行的显卡
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
PARSER.add_argument('-batch_size', '--batch_size', default=32, type=int, help='batch size')
PARSER.add_argument('-lr', '--lr', default=0.0001, type=float, help='learning rate')
PARSER.add_argument('-hops', '--hops', default=3, type=int, help='number of hops')
PARSER.add_argument('-max_children_num', '--max_children_num', default=72, type=int, help='max number of children')
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
PARSER.add_argument('-k', '--k', default=5, type=int, help='the top k paths')

args=PARSER.parse_args()
print(args)

args.device=('cuda:1' if torch.cuda.is_available() else 'cpu')

# 预训练 G模型
# 使用启发式reward训练G网络
def pre_train_generator(gen_model,batcher,max_run_epoch):
    epoch=0
    while epoch <max_run_epoch:
        batches=batcher.get_batches(mode='train')
        step=0
        print(len(batches))
        while step<len(batches):
            current_batch=batches[step]
            print('current_batch:',len(current_batch))
            step+=1
            batchPathes=generator.run_pre_train_step(gen_model,current_batch)

            if len(gen_model.DQN.buffer)>args.batch_size:
                print('updating model............')
                #训练模型
                loss=gen_model.DQN.update()
                print('loss:',loss)
                gen_model.optimizer.zero_grad()
                loss.backward()
                gen_model.optimizer.step()


# 预训练 D模型
def pre_train_discriminator(d_model,d_batcher,max_run_epoch):
    pass

# 训练 G模型
def train_generator(gen_model,d_model,g_batcher,d_batcher,batches,generated):
    pass

# 训练 D模型
def train_discriminator(d_model,max_epoch,d_batcher,batches):
    pass



def main():
    ################################
    ## 第一模块：数据准备工作
    data_=data.Data(args.data_dir, args.vocab_size)

    # 对ICD tree 处理
    parient_children, level2_parients,leafNodes, adj,node2id, hier_dicts= utils.build_tree(os.path.join(args.data_dir,'note_labeled.csv'))
    graph = utils.generate_graph(parient_children, node2id)
    args.node2id=node2id
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
    generated=Generated_example(gen_model,data_,g_batcher)
    # 预训练 G模型
    pre_train_generator(gen_model,g_batcher,10)

    # 利用G 生成一些negative samples
    generated.generator_train_negative_samples()
    generated.generator_test_negative_samples()


    #####################################
    ## 第三模块： 创建 D模型，并预训练 D模型
    d_model=Discriminator(args,data_)

    d_batcher=DisBatcher(data_,args)

    # 预训练 D模型
    pre_train_discriminator(d_model,d_batcher,25)

    ########################################
    ## 第四模块： 交替训练G和D模型
    for epoch in range(args.num_epochs):
        batches=g_batcher.get_batches(mode='train')
        for step in range(int(len(batches)/1000)):

            #训练 G模型
            train_generator(gen_model,d_model,g_batcher,d_batcher,batches[step*1000:(step+1)*1000],generated)

            # 生成训练D的negative samples
            generated.generator_samples("train_sample_generated/"+str(epoch)+"epoch_step"+str(step)+"_temp_positive", "train_sample_generated/"+str(epoch)+"epoch_step"+str(step)+"_temp_negative", 1000)

            # 生成测试样本
            generated.generator_test_samples()

            # TODO: 评估 G模型的表现

            # 创建训练D的batch(即包含 negative samples和positive samples)
            d_batcher.train_batch=d_batcher.create_batches(mode='train',shuffleis=True)

            # 训练 D网络
            train_discriminator(d_model,5,d_batcher,dis_batcher.get_batches(mode="train"))

if __name__ == '__main__':
    main()
