# -*- coding:utf-8 -*-
"""
@Time: 2019/09/11 20:48
@Author: Shanshan Wang
@Version: Python 3.7
@Function:
"""
import csv
import numpy as np

def findrange(icd,level,parient_child,level2_parient):

    for item in level:
        if '-' in item:
            tokens=item.split('-')
            if icd.startswith('E') or icd.startswith('V'):
                if int(icd[1:]) in  range(int(tokens[0][1:]),int(tokens[1][1:])+1):
                    parient_child.append((item,icd))
                    return item
            else:
                # 不是以E或者V开头的
                if int(icd) in  range(int(tokens[0]),int(tokens[1])+1):
                    parient_child.append((item,icd))
                    return item
        else:
            if icd.startswith('E') or icd.startswith('V'):
                if int(icd[1:])==int(item[1:]):
                    parient_child.append((item,icd))
                    level2_parient.append(item)
                    return item
            else:
                # 不是以E或者V开头的
                if int(icd)==int(item):
                    parient_child.append((item,icd))
                    level2_parient.append(item)
                    return item

def build_tree(filepath):

    level2 = ['001-009', '010-018', '020-027', '030-041', '042', '045-049', '050-059', '060-066', '070-079', '080-088',
              '090-099', '100-104', '110-118', '120-129', '130-136', '137-139', '140-149', '150-159', '160-165',
              '170-176',
              '176', '179-189', '190-199', '200-208', '209', '210-229', '230-234', '235-238', '239', '240-246',
              '249-259',
              '260-269', '270-279', '280-289', '290-294', '295-299', '300-316', '317-319', '320-327', '330-337', '338',
              '339', '340-349', '350-359', '360-379', '380-389', '390-392', '393-398', '401-405', '410-414', '415-417',
              '420-429', '430-438', '440-449', '451-459', '460-466', '470-478', '480-488', '490-496', '500-508',
              '510-519',
              '520-529', '530-539', '540-543', '550-553', '555-558', '560-569', '570-579', '580-589', '590-599',
              '600-608',
              '610-611', '614-616', '617-629', '630-639', '640-649', '650-659', '660-669', '670-677', '678-679',
              '680-686',
              '690-698', '700-709', '710-719', '720-724', '725-729', '730-739', '740-759', '760-763', '764-779',
              '780-789',
              '790-796', '797-799', '800-804', '805-809', '810-819', '820-829', '830-839', '840-848', '850-854',
              '860-869',
              '870-879', '880-887', '890-897', '900-904', '905-909', '910-919', '920-924', '925-929', '930-939',
              '940-949',
              '950-957', '958-959', '960-979', '980-989', '990-995', '996-999']
    level2_E=['E000-E899', 'E000', 'E001-E030', 'E800-E807', 'E810-E819', 'E820-E825', 'E826-E829', 'E830-E838',
              'E840-E845', 'E846-E849', 'E850-E858', 'E860-E869', 'E870-E876', 'E878-E879', 'E880-E888', 'E890-E899',
              'E900-E909', 'E910-E915', 'E916-E928', 'E929', 'E930-E949', 'E950-E959', 'E960-E969', 'E970-E978',
              'E980-E989', 'E990-E999']
    level2_V=['V01-V91', 'V01-V09', 'V10-V19','V20-V29','V30-V39', 'V40-V49', 'V50-V59', 'V60-V69', 'V70-V82', 'V83-V84',
              'V85', 'V86', 'V87', 'V88', 'V89','V90','V91']

    allICDS=[] # 保存所有的icds
    with open(filepath,'r') as f:
        reader=csv.reader(f)
        next(reader)
        data=[row[-1] for row in reader]
        for row in data:
            icds=row.split(';')
            allICDS.extend([icd for icd in icds if len(icd)>0])

    allICDS_=list(set(allICDS))
    allICDS_.sort(key=allICDS.index)
    print('write to file:')
    with open('data/allICDs.csv','w') as f:
        writer=csv.writer(f)
        for item in allICDS_:
            writer.writerow([item])

    #针对EHR中出现的每个icd code 找到它的所有父节点，以(parient,child)形式保存
    parient_child=[]
    level2_parient=[]
    hier_icds={}
    for icd in allICDS_:
        hier_icd=[icd]

        # 先判断icd中是否包含E ,例如：E939.58 or E824.1
        if icd.startswith('E'):
            # 先判断是否包含小数点：
            if '.' in icd:
                tokens = icd.split('.')
                # 再判断小数点后有几位
                if len(tokens[1])==1:
                    # 去掉小数点之后会得到第一个父节点 （E824,E824.1）
                    parient_child.append((tokens[0],icd))
                    hier_icd.insert(0,tokens[0])
                    # 找到E824 对应的范围
                    parient=findrange(tokens[0],level2_E,parient_child,level2_parient)
                    hier_icd.insert(0, parient)

                elif len(tokens[1])==2:
                    #去掉小数点后得到会得到三层的父节点
                    parient_child.append((icd[:-1],icd)) #（E939.5，E939.58）
                    hier_icd.insert(0, icd[:-1])
                    parient_child.append((tokens[0],icd[:-1])) #（E939，E939.5）
                    hier_icd.insert(0, tokens[0])
                    parient=findrange(tokens[0],level2_E,parient_child,level2_parient)
                    hier_icd.insert(0, parient)

        # 先判断icd中是否包含V ,例如：V85.54 or V86.0
        elif icd.startswith('V'):
            # 先判断是否包含小数点：
            if '.' in icd:
                tokens = icd.split('.')
                # 再判断小数点后有几位
                if len(tokens[1]) == 1:
                    # 去掉小数点之后会得到第一个父节点 （V86.0,V86）
                    parient_child.append((tokens[0], icd))
                    hier_icd.insert(0, tokens[0])
                    # 找到E824 对应的范围
                    parient=findrange(tokens[0], level2_V, parient_child,level2_parient)
                    hier_icd.insert(0, parient)

                elif len(tokens[1]) == 2:
                    # 去掉小数点后得到会得到三层的父节点
                    parient_child.append((icd[:-1], icd))  # （E939.5，E939.58）
                    hier_icd.insert(0, icd[:-1])
                    parient_child.append((tokens[0], icd[:-1]))  # （E939，E939.5）
                    hier_icd.insert(0, tokens[0])
                    parient=findrange(tokens[0], level2_V, parient_child,level2_parient)
                    hier_icd.insert(0, parient)
        else:
            # 先判断是否包含小数点：
            if '.' in icd:
                tokens = icd.split('.')
                # 再判断小数点后有几位
                if len(tokens[1]) == 1:
                    # 去掉小数点之后会得到第一个父节点 （E824,E824.1）
                    parient_child.append((tokens[0], icd))
                    hier_icd.insert(0, tokens[0])
                    # 找到E824 对应的范围
                    parient=findrange(tokens[0], level2, parient_child,level2_parient)
                    hier_icd.insert(0, parient)
                elif len(tokens[1]) == 2:
                    # 去掉小数点后得到会得到三层的父节点
                    parient_child.append((icd[:-1], icd))  # （E939.5，E939.58）
                    hier_icd.insert(0, icd[:-1])
                    parient_child.append((tokens[0], icd[:-1]))  # （E939，E939.5）
                    hier_icd.insert(0, tokens[0])
                    parient=findrange(tokens[0], level2, parient_child,level2_parient)
                    hier_icd.insert(0, parient)
        if icd not in hier_icds:
            hier_icds[icd]=hier_icd


    # print(parient_child)
    # print('level2_parients:',level2_parient)
    # 把所有相关的节点转换成id的形式

    nodes = []
    for row in parient_child:
        if row[0] not in nodes:
            nodes.append(row[0])
        if row not in nodes:
            nodes.append(row[1])

    nodes=nodes + allICDS_
    nodes_=list(set(nodes))
    nodes_.sort(key=nodes.index)

    node2id = {node: (id+1) for id, node in enumerate(nodes_)}
    node2id['PAD']=0 # 0的位置进行PADDing
    node2id['ROOT']=len(node2id)

    #根据parient_child 生成一个邻接矩阵，用来方便寻找每个子节点的孩子
    parient_child_new=[]
    adj=np.zeros((len(node2id),len(node2id)))
    # print('len:',len(node2id))
    for row in parient_child:
        # print(row[0],row[1])
        if row[0]!=row[1]:
            adj[node2id.get(row[0])][node2id.get(row[1])]=1
            parient_child_new.append([node2id.get(row[0]),node2id.get(row[1])])

    level2_parient_new = []
    for parient in level2_parient:
        level2_parient_new.append(node2id.get(parient))

    level2_parient=list(set(level2_parient_new))
    level2_parient.sort(key=level2_parient_new.index)
    # 统计最大的孩子个数
    children_num=[len(np.argwhere(row)) for row in adj]
    max_children_num=max(len(level2_parient),max(children_num))
    min_children_num=min(len(level2_parient),min(children_num))
    print('max_childeren_num:',max_children_num) # 72
    print('level2_parient:',len(level2_parient))  # 0 (有些没有孩子节点，它本身就是叶子节点)

    level2_parient=level2_parient+[0]*(max_children_num-len(level2_parient))
    leafNode=[node2id.get(item) for item in allICDS_]


    # 将层级的labels转换为id
    hier_dicts={}
    for icd,hier_icd in hier_icds.items():
        # print('icd:',icd)
        # print('hier_icd:',hier_icd)
        icdId=node2id.get(icd)
        hier_icdIds=[node2id.get(item) for item in hier_icd]
        hier_dicts[icdId]=hier_icdIds
    # print('hier_dicts：',hier_dicts)
    return parient_child_new,[level2_parient],leafNode,adj,node2id,hier_dicts

def generate_graph(parient_child,node2id):
    import networkx as nx
    # 将parient-child中的每条边转换成id

    # 根据关系创建图
    G = nx.Graph()
    G.add_nodes_from(node2id.values())
    G.add_edges_from(parient_child)
    print('number of nodes:', G.number_of_nodes())
    print('number of edges:', G.number_of_edges())

    return G


