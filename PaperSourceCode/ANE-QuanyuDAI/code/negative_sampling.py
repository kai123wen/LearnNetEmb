# -*- coding: UTF-8 -*-
'''
Refer to https://github.com/deborausujono/word2vecpy/blob/master/word2vec.py
'''

'''
使用了轮盘赌的原理
如果是采用负采样的方法，此时还需要初始化每个词被选中的概率。
在所有的词构成的词典中，每一个词出现的频率有高有低，我们希望，对于那些高频的词，被选中成为负样本的概率要大点
'''

import math
import numpy as np


class UnigramTable:
    """
    Using degree list to initialize the drawing 
    """

    # 这里应该是使用了负采样的方法
    # 参考
    # https://zhuanlan.zhihu.com/p/27234078

    """
    我们使用“一元模型分布（unigram distribution）”来选择“negative words”。

    要注意的一点是，一个单词被选作negative sample的概率跟它出现的频次有关，出现频次越高的单词越容易被选作negative words。

    在word2vec的C语言实现中，你可以看到对于这个概率的实现公式。每个单词被选为“negative words”的概率计算公式与其出现的频次有关。
    
    """

    def __init__(self, vocab):
        # vocab 是每个节点的度的列表
        vocab_size = len(vocab)
        power = 0.75
        # 这里 t 应该就是代表的是频次
        norm = sum([math.pow(t, power) for t in vocab])  # Normalizing constant

        table_size = int(1e8)  # Length of the unigram table
        table = np.zeros(table_size, dtype=np.uint32)

        # unigram table 有一个包含了一亿个元素的数组，
        # 这个数组是由词汇表中每个单词的索引号填充的，并且这个数组中有重复，也就是说有些单词会出现多次。
        print('Filling unigram table')
        p = 0  # Cumulative probability
        i = 0
        for t in range(vocab_size):
            p += float(math.pow(vocab[t], power)) / norm
            # 负采样概率*1亿=单词在表中出现的次数。
            while i < table_size and float(i) / table_size < p:
                table[i] = t
                i += 1
        self.table = table
        print('Finish filling unigram table')

    '''
    有了这张表以后，每次去我们进行负采样时，只需要在0-1亿范围内生成一个随机数，
    然后选择表中索引号为这个随机数的那个单词作为我们的negative word即可。
    一个单词的负采样概率越大，那么它在这个表中出现的次数就越多，它被选中的概率就越大。
    '''

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]
