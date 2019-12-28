"""
和之前不同的是添加了两个隐藏层
添加 dropout 来进一步改进简单网络
dropout 可以缓解过拟合的状况
"""
# 在开头加上from __future__ import print_function这句之后，即使在python2.X，使用print就得像python3.X那样加括号使用。
# python2.X中print不需要括号，而在python3.X中则需要。
from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
# 加载全连接层和激活函数
from keras.layers.core import Dense, Activation, Dropout
# 加载随机梯度下降优化器
from keras.optimizers import SGD
# Keras 改造的 numpy 的一个函数 np_utils.to_categorical
from keras.utils import np_utils
#  引入 tensorBoard
from keras.callbacks import TensorBoard

# 设定随机数种子，保证结果的可重现性
np.random.seed(1671)
NB_EPOCH = 200
#  共分为10类
NB_CLASSES = 10
BATCH_SIZE = 128
# 优化器选用 随机梯度下降
OPTIMIZER = SGD()
N_HIDDEN = 128
DROPOUT = 0.3
VALIDATION_SPLIT = 0.2  # 训练集中用作验证集的数据比例
# 数据
# y_train 是因为数字识别的训练集中含有每张图片的标签
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

# to_categorical就是将类别向量转换为二进制（只有0和1）的矩阵类型表示。即one-hot编码
# 其表现为将原有的类别向量转换为独热编码的形式。先上代码看一下效果：
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))

model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))

model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=
OPTIMIZER, metrics=['accuracy'])

#  在最后的训练中才划分了训练集和测试集
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=2,
                    validation_split=VALIDATION_SPLIT)
score = model.evaluate(X_test, Y_test, verbose=2)
print("Test score : ", score[0])
print("Test accuracy : ", score[1])
json_string = model.to_json()
with open('json_string.json', 'w') as f:
    f.write(json_string)
