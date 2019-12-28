# -*- coding: UTF-8 -*-
import numpy as np
import networkx as nx
import random


# From node2vec source code

class Graph():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        从一个初始节点计算一个随机游走
        walk_length :随机游走序列长度,这里初始的时候传递的是输入的args.length
        start_node ：初始节点
        :return 列表，随机游走序列
        """
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
                                               alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), '/', str(num_walks))
            # 将序列的所有元素随机排序
            random.shuffle(nodes)
            for node in nodes:
                # 向数组中append添加得到的随机游走序列
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return np.array(walks)

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G
        is_directed = self.is_directed

        # 存储 每个节点的邻居节点，其权重占比的列表 例如 alias_nodes['a'] = [1/6,1/3,1/2]
        alias_nodes = {}
        # G.nodes 返回图的节点列表
        # 对于每一个节点
        for node in G.nodes():
            # 得到当前结点的邻居结点(有直连关系)的权值列表，[1,1,1,1...]
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            # 权重求和
            norm_const = sum(unnormalized_probs)
            # 求每个权重的占的比重，权重大的占的比重就大，将其形成一个列表
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        """
        这里目前看不懂，应该是给边赋值
        """
        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        # alias_nodes形式为{1:(J, q), 2:(J,q)...},1和2代表结点id
        # alias_edges形式为{(1,2):(J,q), (2,1):(J,q),(1,3):(J,q)...} (1,2)代表一条边

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


# 这里使用了别名采样法： Alias method
# 参见博客 https://blog.csdn.net/haolexiao/article/details/65157026

"""
这个函数的作用还是不明白
"""


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    :param probs: 节点之间的权重所占比例向量，是一个列表
    :return : 输入概率，得到对应的两个列表
            一个是原始的 probs 数组,如[0.4,0.8,0.6]
            另外就是在上面补充的Alias数组，其值代表填充的那一列的序号索引
            具体的可以参见博客 https://blog.csdn.net/haolexiao/article/details/65157026
            方便后面的抽样调用
    '''
    # J和q数组和probs数组大小一致
    # probs长度由当前结点的邻居节点数量决定
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    # 将数据分类为具有概率的结果 大于或者小于1 / K.
    # 这两个列表里存放的是结点的下标
    smaller = []
    larger = []

    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    # 先列出下标 在列出数据，即 kk 是下标，prob 是数据
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        # 这里就是判断每一个概率 prob 大于还是小于 1/k
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    # 然后循环并创建少量二元混合分布
    # 在整个均匀混合分布中适当地分配更大的结果。
    # 假如每条边权重都为1，实际上这里的while循环不会执行，因为每条边概率都是一样的，相当于不需要采样
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
