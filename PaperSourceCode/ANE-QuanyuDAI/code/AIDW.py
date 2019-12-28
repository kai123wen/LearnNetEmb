# -*- coding: UTF-8 -*-
import matplotlib as mpl
import time

# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')

from keras.layers import Dense, Reshape, Flatten, Input, merge, noise, Embedding
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l1, L1L2
import keras.backend as K
from keras.layers import LeakyReLU, Activation
from keras.layers.merge import concatenate, multiply
from keras.layers.normalization import BatchNormalization
from keras.utils.generic_utils import Progbar

import os
import scipy.io as sio
import pandas as pd
import numpy as np
import networkx as nx

import sampler
import dataset
import node2vec
import mcc_liblinear


# customed loss function to capture network structure properties
def cross_entropy_loss(y_true, y_pred):
    return - K.mean(K.log(K.sigmoid(K.clip(K.sum(y_pred, axis=1) * y_true, -6, 6))))


def context_preserving(latent_dim):
    node_rep = Input(shape=(latent_dim,), name='node_rep')
    context_rep = Input(shape=(latent_dim,), name='context_rep')
    sim = multiply([node_rep, context_rep])
    return Model(inputs=[node_rep, context_rep], outputs=sim, name='context_aware')


def encoder(node_num, hidden_layers, hidden_neurons):
    """
    :param node_num: 节点数量
    :param hidden_layers: 隐藏层数量
    :param hidden_neurons: 隐藏层神经元个数
    :return: 训练模型
    """
    x = Input(shape=(node_num,))
    # 高斯噪声 正则化层 缓解过拟合
    encoded = noise.GaussianNoise(0.2)(x)
    for i in range(hidden_layers):
        encoded = Dense(hidden_neurons[i])(encoded)
        encoded = LeakyReLU(0.2)(encoded)
        BatchNormalization()
        encoded = noise.GaussianNoise(0.2)(encoded)
    return Model(inputs=x, outputs=encoded)


def model_discriminator(latent_dim, output_dim=2, hidden_dim=512, reg=lambda: L1L2(1e-7, 1e-7)):
    z = Input((latent_dim,))
    h = Dense(hidden_dim, kernel_regularizer=reg())(z)
    # LeakyRelU是修正线性单元（Rectified Linear Unit，ReLU）的特殊版本，
    # 当不激活时，LeakyReLU仍然会有非零输出值，从而获得一个小梯度，避免ReLU可能出现的神经元“死亡”现象。
    h = LeakyReLU(0.2)(h)
    # 能够保证权重的尺度不变，因为BatchNormalization在激活函数前对输入进行了标准化
    # 该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1
    BatchNormalization()
    h = Dense(hidden_dim, kernel_regularizer=reg())(h)
    h = LeakyReLU(0.2)(h)
    BatchNormalization()
    y = Dense(output_dim, activation="softmax", kernel_regularizer=reg())(h)
    return Model(z, y)


# args 执行时输入的参数
def AIDW(args):
    # read data and define batch generator

    # 根据 citeseer-edgelist 文件，将其转化为 图
    nx_G = dataset.read_graph(args)

    # 构建 node2vec 的 Graph 类对象。 node2vec 源码讲解：https://blog.csdn.net/wen_fei/article/details/82690530
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)

    # Preprocessing of transition probabilities for guiding the random walks.
    # 这里应该就是对node2vec中离开概率参数进行预处理，具体的作用还是不明确

    G.preprocess_transition_probs()

    #   对每个结点，根据num_walks得出其多条随机游走路径
    walks = G.simulate_walks(args.num_walks, args.walk_length)

    # 这里加载 PPMI 矩阵
    '''
    这里对于 PPMI 矩阵的作用还是不明白
    '''

    network = dataset.load_data(args.input_ppmi)

    # 加载邻接矩阵
    adjMat = dataset.load_data(args.input_adj)

    # walks 是通过对图中的每个节点进行 num_walks 次遍历，每一次遍历，都会遍历到所有点，对所有点获取其本次的随机游走序列

    """
    这个函数的作用目前还不是很清楚
    """
    walk_generator = dataset.contextSampling(walks, adjMat, args)

    # -------------------------------read parameters-------------------------------#
    # 得到名字 citeseer-PPMI-4
    dataset_name = args.input_ppmi.split('/')[1].split('.')[0]

    # 隐藏层层数设置为 1
    hidden_layers = args.hidden_layers

    # 神经元个数设置为 128
    neurons = args.hidden_neurons

    neurons = neurons.split('/')
    hidden_neurons = []
    for i in range(len(neurons)):
        hidden_neurons.append(int(neurons[i]))
    print('******* hidden_neurons : ', hidden_neurons)
    # 得到节点数量
    node_num = network.shape[0]
    # latent_dim = 128
    latent_dim = hidden_neurons[-1]
    # ------------------------------------------------------------------------------#

    # -----------------------------build the graph----------------------------------#
    # embeddings (x ->z)
    # encoder_node, encoder_context = encoder_NN(node_num, hidden_layers, hidden_neurons)
    # 通过编码器 简化信息
    encoder_node = encoder(node_num, hidden_layers, hidden_neurons)
    encoder_context = encoder(node_num, hidden_layers, hidden_neurons)

    # context preserving
    context_prediction = context_preserving(latent_dim)

    # constructing context preserving model
    node = encoder_node.inputs[0]
    context = encoder_context.inputs[0]
    node_rep = encoder_node(node)
    context_rep = encoder_context(context)
    sim = context_prediction([node_rep, context_rep])
    context_model = Model(inputs=[node, context], outputs=sim)
    context_model.compile(optimizer=RMSprop(lr=args.lr), loss=cross_entropy_loss)

    # discriminator (z -> y)
    discriminator = model_discriminator(latent_dim)
    discriminator.compile(optimizer=RMSprop(lr=args.lr), loss='categorical_crossentropy')

    # train generator
    x = encoder_node.inputs[0]
    z = encoder_node(x)
    y_fake = discriminator(z)
    gan = Model(inputs=x, outputs=y_fake)
    gan.compile(optimizer=RMSprop(lr=args.lr), loss='mse')
    # ------------------------------------------------------------------------------#

    # ---------------------------print summary of models----------------------------#
    print('encoder_node:')
    encoder_node.summary()
    print('encoder_context:')
    encoder_context.summary()
    print('context_prediction:')
    context_prediction.summary()
    print('context_model:')
    context_model.summary()
    print('Discriminator:')
    discriminator.summary()
    print('GAN:')
    gan.summary()
    # ------------------------------------------------------------------------------#
    epoch_gen_loss = []
    epoch_disc_loss = []
    epoch_context_loss = []
    index = 0

    while True:
        index = index + 1
        # -------------------------batch data sampling for context--------------------------------------#
        l_nodes, r_nodes, labels = next(walk_generator)
        batchsize = l_nodes.shape[0]

        left_batch = network[l_nodes]
        right_batch = network[r_nodes]
        data_batch = np.concatenate([left_batch, right_batch], axis=0)

        for t in range(args.T0):
            epoch_context_loss.append(context_model.train_on_batch([left_batch, right_batch], labels))

        # the updating of the discriminator
        # noise = np.random.normal(0, 1, (2*batchsize, latent_dim)).astype(np.float32)
        noise = np.random.uniform(-1.0, 1.0, [2 * batchsize, latent_dim])
        # noise = np.random.power(1, (2*batchsize, latent_dim))
        z_batch = encoder_node.predict(data_batch)
        X = np.concatenate((noise, z_batch))
        y_dis = np.zeros([4 * batchsize, 2])
        y_dis[0:2 * batchsize, 1] = 1
        y_dis[2 * batchsize:, 0] = 1

        for t in range(args.T1):
            # clip weights
            discriminator.trainable = True
            weights = [np.clip(w, -0.01, 0.01) for w in discriminator.get_weights()]
            discriminator.set_weights(weights)
            epoch_disc_loss.append(discriminator.train_on_batch(X, y_dis))

        # the updating of the generator
        y_fake = np.zeros([2 * batchsize, 2])
        y_fake[:, 1] = 1
        for t in range(args.T2):
            discriminator.trainable = False
            epoch_gen_loss.append(gan.train_on_batch(data_batch, y_fake))

        if index % 50 == 0:
            print('\nTraining loss for index {}:'.format(index))
            context_loss = np.mean(np.array(epoch_context_loss[-50:]), axis=0)
            dis_loss = np.mean(np.array(epoch_disc_loss[-50:]), axis=0)
            gen_loss = np.mean(np.array(epoch_gen_loss[-50:]), axis=0)
            print('AutoE-{} Dis-{} Gen-{}'.format(context_loss, dis_loss, gen_loss))

        if (index) % (200) == 0:
            rep = encoder_node.predict(network)

            rep_file = 'output/{}-rep-{}.mat'.format(dataset_name, str((index) / 200))
            sio.savemat(rep_file, {'rep': rep})

            sio.savemat(args.rep, {'rep': rep})
            resultFile = open(args.resultTxt, 'a')
            resultFile.write('index-{}\n'.format(index))
            resultFile.close()
            results = mcc_liblinear.mcc_liblinear_one_file(args)
            mcc_liblinear.save_results(results, args.resultTxt)
