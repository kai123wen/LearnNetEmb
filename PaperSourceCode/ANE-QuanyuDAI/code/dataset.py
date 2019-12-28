# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import scipy.io as sio
from collections import defaultdict as dd
import random
import networkx as nx
import node2vec
from negative_sampling import UnigramTable


def load_data(file):
    # load data
    data_mat = sio.loadmat(file)
    data_mat = data_mat['network']
    data_mat = data_mat.toarray()
    return data_mat


# From node2vec source code
# 读取含有边信息的文件，将其转为图
def read_graph(args):
    '''
    Reads the input network in networkx.
    '''
    # 判断是都是带权图
    if args.weighted:
        G = nx.read_edgelist(args.input_edgelist, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input_edgelist, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    # 判断是否是无向图
    if not args.directed:
        G = G.to_undirected()

    return G


# walk_generator = dataset.contextSampling(walks, adjMat, args)
def contextSampling(walks, network, args):
    '''
    batch generator
    '''
    """
    walks 大约的形式为：
    [ [a,b,c..]
      [b,c,a..]
      [a,c,b..]
      [b,d,c..]
      .
      .
      .
            ]
    """
    walks_num = walks.shape[0]
    walks_len = walks.shape[1]

    # 得到每个节点的度
    degree = list(np.sum(network, 1))

    # 得到用于负采样 有了这张表以后，每次去我们进行负采样时，
    # 只需要在0-1亿范围内生成一个随机数，然后选择表中索引号为这个随机数的那个单词作为我们的negative word即可
    table = UnigramTable(degree)

    count = 0
    l_nodes = []
    r_nodes = []
    labels = []
    while True:
        for i in range(walks_num):
            for l in range(walks_len):
                # 以l位置为中心，左右各 args.window_size 大小
                for m in range(l - args.window_size, l + args.window_size + 1):
                    if m < 0 or m >= walks_len: continue
                    # 当 l == 10 的时候，m >= 0 开始
                    """
                    l_nodes 含有 l 点左边 window_size 大小的节点
                    r_nodes 含有 l 点右边 window_size 大小的节点
                    """
                    l_nodes.append(walks[i, l] - 1)
                    r_nodes.append(walks[i, m] - 1)
                    labels.append(1.0)
                    # negative samples corresponding to the current positive pair
                    for k in range(args.K):
                        n_neg = table.sample(1)[0]
                        while n_neg == walks[i, l] - 1 or n_neg == walks[i, m] - 1:
                            n_neg = table.sample(1)[0]

                        l_nodes.append(walks[i, l] - 1)
                        r_nodes.append(n_neg)
                        labels.append(-1.0)
                    count = count + 1
                    if count >= args.batch_size:
                        yield np.array(l_nodes, dtype=np.int32), np.array(r_nodes, dtype=np.int32), np.array(labels,
                                                                                                             dtype=np.float32)
                        l_nodes = []
                        r_nodes = []
                        labels = []
                        count = 0
