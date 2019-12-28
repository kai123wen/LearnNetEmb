from keras.models import Model
from keras.layers import Input, Dense, Reshape, Subtract
from keras.layers import merge as m
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

import urllib.request
import collections
import os
import zipfile

import numpy as np
import tensorflow as tf


def maybe_download(filename, url, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        # filename, _ = urllib.urlretrieve(url + filename, filename)
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


# Read the data into a list of strings.
# 得到单词的列表
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    # UNK - > unknown
    count = [['UNK', -1]]
    # >>> c = Counter('abracadabra')
    # >>> c.most_common()
    # [('a', 5), ('r', 2), ('b', 2), ('c', 1), ('d', 1)]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    # dictionary = ('cat',1)... 类似形式
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    """
    data 这个数组的作用就是根据 dictionary ，将 words 单词序列转化为对应的索引序列，
    比如：word = [b,a,a,b,a]
    -> count = [['UNK', -1],['a',3],['b',2]]
    -> dictionary = dict(('UNK',0),('a',1),('b',2))
    -> data = [2,1,1,2,1]
    上面的步骤概括一下：
    1. 将单词序列按照出现频次来进行降序排列
    2. 为得到的降序排列的单词添加id
    3. 将原来的单词序列用id来进行替换
    """
    # 更新 count 数组
    count[0][1] = unk_count
    # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    # 将 dictionary 翻转，一遍以后通过 id 找到对应的单词
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def collect_data(vocabulary_size=10000):
    url = 'http://mattmahoney.net/dc/'
    filename = maybe_download('text8.zip', url, 31344016)
    vocabulary = read_data(filename)
    print(vocabulary[:7])
    """
    -> data 就是将原来单词列表表示为id列表
    -> count 就是根据单词的出现频率对单词进行降序排列得到的数组，数组的每一行是单词与其出现次数
    -> dictionary 包含的是 单词与id 的对应关系
    -> reverse_dictionary 包含的是 id 与 单词的对应关系
    """
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                                vocabulary_size)
    del vocabulary  # Hint to reduce memory.
    return data, count, dictionary, reverse_dictionary


vocab_size = 10000  # 单词数量
data, count, dictionary, reverse_dictionary = collect_data(vocabulary_size=vocab_size)
print(data[:7])

window_size = 3  # 目标单词周围能作为上下文的窗口大小
vector_dim = 300  # 词向量的尺寸
epochs = 200000  # epoch 次数

valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

# 这里的采样表的作用是获取到 size 为 10000 的列表，
# 该函数用以产生skipgrams中所需要的参数sampling_table。这是一个长为size的向量，
# sampling_table[i]代表采样到数据集中第i常见的词的概率（为平衡期起见，对于越经常出现的词，要以越低的概率采到它）
sampling_table = sequence.make_sampling_table(vocab_size)
# print("sum：",sum(sampling_table))
# print("sampling_table", sampling_table)
# window_size ： 整数，正样本对之间的最大距离，
couples, labels = skipgrams(data, vocab_size, window_size=window_size, sampling_table=sampling_table)
# 将couples 分解成word_target 和 word_context
word_target, word_context = zip(*couples)
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")

print(couples[:10], labels[:10])

# create some input variables
# 首先我们需要指定什么样的张量（以及大小）将要输入到模型中。
# 在本例中，我们会输入单个目标词和上下文相关词，所以每个输入变量大小为(1,)
input_target = Input((1,))
input_context = Input((1,))

# 构造嵌入层
embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')
target = embedding(input_target)
# print("target:", target)
target = Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)

# setup a cosine similarity operation which will be output in a secondary model
# similarity = merge([target, context], mode='cos', dot_axes=0)
similarity = Subtract()([target, context])

# now perform the dot product operation to get a similarity measure
# dot_product = merge([target, context], mode='dot', dot_axes=1)
dot_product = m.dot([target, context], axes=1)
dot_product = Reshape((1,))(dot_product)
# add the sigmoid output layer
output = Dense(1, activation='sigmoid')(dot_product)
# create the primary training model
model = Model(input=[input_target, input_context], output=output)
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

# create a secondary validation model to run our similarity checks during training
validation_model = Model(input=[input_target, input_context], output=similarity)


class SimilarityCallback:
    def run_sim(self):
        for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            print('valid_word: ', valid_word)
            top_k = 8  # number of nearest neighbors
            # valid_size = 16  # Random set of words to evaluate similarity on.
            # valid_window = 100  # Only pick dev samples in the head of the distribution.
            # valid_examples = np.random.choice(valid_window, valid_size, replace=False)
            sim = self._get_sim(valid_examples[i])
            print('sim: ', sim)
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def _get_sim(valid_word_idx):
        sim = np.zeros((vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        for i in range(vocab_size):
            in_arr1[0,] = valid_word_idx
            in_arr2[0,] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            print("*** ", out)
            sim[i] = out
        return sim


sim_cb = SimilarityCallback()

arr_1 = np.zeros((1,))
arr_2 = np.zeros((1,))
arr_3 = np.zeros((1,))
for cnt in range(epochs):
    print("**")
    idx = np.random.randint(0, len(labels) - 1)
    arr_1[0,] = word_target[idx]
    arr_2[0,] = word_context[idx]
    arr_3[0,] = labels[idx]
    # train_on_batch 函数在一个 batch 的数据上进行一次参数更新，函数返回训练误差的标量值或标量值的 list，与 evaluate 的情形相同。
    loss = model.train_on_batch([arr_1, arr_2], arr_3)
    if cnt % 100 == 0:
        print("Iteration {}, loss={}".format(cnt, loss))
    if cnt % 10000 == 0:
        sim_cb.run_sim()
