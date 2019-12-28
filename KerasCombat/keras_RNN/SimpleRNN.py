from __future__ import print_function
from keras.layers import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
import numpy as np

# 读取文件，并将文件内容解码为ASCII码
fin = open('alice_in_wonderland.txt', 'rb')
lines = []  #
for line in fin:  # 遍历每行数据
    line = line.strip().lower()  # 去除每行两端空格
    line = line.decode('ascii', 'ignore')  # 解码为ASCII码
    if len(line) == 0:
        continue
    lines.append(line)
fin.close()
text = ' '.join(lines)

chars = set([c for c in text])  # 获取字符串中不同字符组成的集合
nb_chars = len(chars)  # 获取集合中不同元素数量
char2index = dict((c, i) for i, c in enumerate(chars))  # 创建字符到索引的字典
index2char = dict((i, c) for i, c in enumerate(chars))  # 创建索引到字符的字典

SEQLEN = 10  # 超参数，输入字符串长度
STEP = 1  # 输出字符串长度
input_chars = []  # 输入字符串列表
label_chars = []  # 标签列表
for i in range(0, len(text) - SEQLEN, STEP):
    input_chars.append(text[i: i + SEQLEN])
    label_chars.append(text[i + SEQLEN])

# 将输入文本和标签文本向量化: one-hot编码
X = np.zeros((len(input_chars), SEQLEN, nb_chars), dtype=np.bool)  # 输入文本张量
Y = np.zeros((len(input_chars), nb_chars), dtype=np.bool)  # 标签文本张量
for i, input_char in enumerate(input_chars):  # 遍历所有输入样本
    for j, ch in enumerate(input_char):  # 对于每个输入样本
        X[i, j, char2index[ch]] = 1
    Y[i, char2index[label_chars[i]]] = 1

# 构建模型
HIDDEN_SIZE = 128
BATCH_SIZE = 128
NUM_ITERATIONS = 25
NUM_EPOCH_PER_ITERATIONS = 1
NUM_PREDS_PER_EPOCH = 100

model = Sequential()
model.add(SimpleRNN(HIDDEN_SIZE, return_sequences=False, input_shape=(SEQLEN, nb_chars), unroll=True))
model.add(Dense(nb_chars))
model.add(Activation('softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

for iteration in range(NUM_ITERATIONS):
    print('=' * 50)
    print('Iteration #: %d' % (iteration))  # 打印迭代次数
    # 训练模型
    model.fit(X, Y, batch_size=BATCH_SIZE, epochs=NUM_EPOCH_PER_ITERATIONS)  # 每次迭代训练一个周期
    # 使用模型进行预测
    test_idx = np.random.randint(len(input_chars))  # 随机抽取样本
    test_chars = input_chars[test_idx]
    print('Generating from seed: %s' % (test_chars))
    print(test_chars, end='')  # 不换行输出
    for i in range(NUM_PREDS_PER_EPOCH):  # 评估每一次迭代后的结果
        Xtest = np.zeros((1, SEQLEN, nb_chars))
        for i, ch in enumerate(test_chars):
            Xtest[0, i, char2index[ch]] = 1  # 样本张量
        # pred = model.predict(Xtest, verbose=0)[0]
        # ypred = index2char[np.argmax(pred)]		# 找对类别标签对应的字符
        ypred = index2char[model.predict_classes(Xtest)[0]]
        print(ypred, end='')
        # 使用test_chars + ypred继续
        test_chars = test_chars[1:] + ypred
print()
