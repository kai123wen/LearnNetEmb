# -*- coding: UTF-8 -*-
import matplotlib as mpl
import time

import argparse
import numpy as np
import scipy.io as sio
import dataset
import AIDW
import mcc_liblinear


def parse_args():
    '''
    Parses the AIDW arguments.
    '''
    parser = argparse.ArgumentParser(description="Run Adversarial Inductive DeepWalk.")
    # for input
    parser.add_argument('--input_edgelist', nargs='?', default='input/citeseer-edgelist.txt', help='Input graph path')
    parser.add_argument('--input_ppmi', nargs='?', default='input/citeseer-PPMI-4.mat', help='Input PPMI')
    parser.add_argument('--input_adj', nargs='?', default='input/citeseer-undirected.mat', help='Input adjMat')
    parser.add_argument('--rep', nargs='?', default='output/citeseer-rep.mat', help='Embeddings path')
    # for random walk
    # Walk_length: How many nodes are in each random walk
    # num_walks : Number of random walks to be generated from each node in the graph
    parser.add_argument('--walk_length', type=int, default=10, help='Length of walk per source.')
    parser.add_argument('--num_walks', type=int, default=10, help='Number of walks per source.')
    parser.add_argument('--window_size', type=int, default=10, help='Context size for optimization.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
    parser.add_argument('--K', type=int, default=5, help='Number of negative pairs for each positive pair.')

    #
    # --返回概率参数（Return parameter）p，对应BFS，p控制回到原来节点的概率，从节点t跳到节点v以后，有1 / p的概率在节点v处再跳回到t。
    #
    # --离开概率参数（Inout parameter）q，对应DFS，q控制跳到其他节点的概率。
    parser.add_argument('--p', type=float, default=1, help='Return hyperparameter.')
    parser.add_argument('--q', type=float, default=1, help='Inout hyperparameter.')
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)
    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)
    # for GANs
    parser.add_argument('--hidden_layers', type=int, default=1, help='Number of hidden layers.')
    parser.add_argument('--hidden_neurons', nargs='?', default='128', help='Hidden neurons')
    # for training
    parser.add_argument('--T0', type=int, default=1, help='Context loss iteration times.')
    parser.add_argument('--T1', type=int, default=1, help='Discriminator iteration times.')
    parser.add_argument('--T2', type=int, default=1, help='Generator iteration times.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    # for test
    parser.add_argument('--resultTxt', nargs='?', default='data/AIDW-citeseer.txt')

    return parser.parse_args()


def main(args):
    localtime = time.asctime(time.localtime(time.time()))
    print('Begining Time :', localtime)

    # 这里 args.k 初始是 5 ,因此K = [5]
    # 这里K代表的是对于每一个positive word ，设置多少个negative word
    # https://www.jianshu.com/p/ed15e2adbfad
    # 这里涉及到了负采样的知识，在关于word2vec 的论文中，作者提出了对于小规模数据集，建议选择 5-20 个 negative words，
    # 对于大规模数据集选择 2-5个 negative words.
    K = [args.K]
    # 学习率，也就是步长
    learning_rate = [args.lr]

    for k in K:
        args.K = k
        for lr in learning_rate:
            resultFile = open(args.resultTxt, 'a')
            args.lr = lr

            print('-----------------New settings------------------')
            resultFile.write('==============================\n')
            resultFile.write('==============================\n')
            # num_walks：nodes采样一次为一个epoch，那此处就是num_walks个epoch,这里 num_walks 初始化为 10 ，也就是进行10个epoch

            # num_walks : Number of random walks to be generated from each node in the graph
            # walk_length : How many nodes are in each random walk
            # window_size : 这里的window_size 类似于 word2vec 中的
            # batch_size : 每批处理数据的大小
            # lr ： 学习率
            settings = 'num_walks-{}-walk_length-{}-window_size-{}-K-{}-batch_size-{}-lr-{}\n'.format(
                str(args.num_walks), str(args.walk_length), str(args.window_size), str(args.K), str(args.batch_size),
                str(args.lr))
            resultFile.write(settings)
            # input_edgelist : 边集合，即有两列id
            # ppmi 这里是有疑惑的
            # input_adj 应该是邻接矩阵
            # hidden_neurons：Number of hidden layers
            settings = 'input-{}\ninput_ppmi-{}\ninput_adj-{}\nhidden_neurons-{}\n'.format(args.input_edgelist,
                                                                                           args.input_ppmi,
                                                                                           args.input_adj,
                                                                                           args.hidden_neurons)
            resultFile.write(settings)
            resultFile.write('Leak-0.2\n')
            resultFile.write('==============================\n')
            resultFile.write('==============================\n')

            resultFile.close()

            AIDW.AIDW(args)

    localtime = time.asctime(time.localtime(time.time()))
    print('Endding Time :', localtime)


if __name__ == "__main__":
    args = parse_args()
    main(args)
