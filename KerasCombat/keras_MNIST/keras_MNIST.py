"""
感知机 keras 手写数字识别
"""
# 在开头加上from __future__ import print_function这句之后，即使在python2.X，使用print就得像python3.X那样加括号使用。
# python2.X中print不需要括号，而在python3.X中则需要。
from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
# 加载全连接层和激活函数
from keras.layers.core import Dense, Activation
# 加载随机梯度下降优化器
from keras.optimizers import SGD
# Keras 改造的 numpy 的一个函数 np_utils.to_categorical
from keras.utils import np_utils

# 设定随机数种子，保证结果的可重现性
np.random.seed(1671)
NB_EPOCH = 200
#  共分为10类
NB_CLASSES = 10
BATCH_SIZE = 128
# 优化器选用 随机梯度下降
OPTIMIZER = SGD()
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2  # 训练集中用作验证集的数据比例
# 数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train 是60000行28*28的数据，变形为60000*784
# print(X_test)
# 28*28 图片像素数
RESHAPED = 784
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
# 转化为GPU便于计算的float32类型
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# 归一化
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# to_categorical就是将类别向量转换为二进制（只有0和1）的矩阵类型表示。
# 其表现为将原有的类别向量转换为独热编码的形式。先上代码看一下效果：
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

model = Sequential()
#  定义一层网络 ，该层中含有 NB_CLASSES 个神经元，预计有 RESHAPED 个输入
#  注意，这里 input dim= A 与 input_shape=(RESHAPED,) 作用是一样的
model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,)))
# 为网络层添加激活函数
model.add(Activation('softmax'))
#  模型概括打印
model.summary()
# 定义好模型之后我们需要通过编译（compile）来对学习过程进行配置，
# 我们可以为模型的编译指定各类参数包括：优化器optimizer，损失函数loss，评估指标metrics。
# http://www.ywk.space/2017/08/03/%E5%8E%9F%E5%88%9Bkeras%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B02%EF%BC%9A%E9%A1%BA%E5%BA%8F%E6%A8%A1%E5%9E%8B/
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
# verbose 是打印日志的选项
# 进行训练，首先是样本组成方式，keras样本是以numpy中的array类型作为载体的。训练模型一般使用fit函数
# epochs为迭代次数，batch_size为批次大小
# fit 中的 verbose
# verbose：日志显示
# verbose = 0 为不在标准输出流输出日志信息
# verbose = 1 为输出进度条记录
# verbose = 2 为每个epoch输出一行记录
# 注意： 默认为 1
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=2)
# 训练后的模型，我们需要对其性能进行评估，以此来确定训练效果是否达到了我们的预期。
# loss, accuracy = model.evaluate(X, y)
# evaluate方法的参数X,y与fit方法的数据类型是一样的，一般会选择用测试数据进行评估
score = model.evaluate(X_test, Y_test, verbose=2)
print('Test score: ', score[0])
print('tets accuracy :', score[1])
