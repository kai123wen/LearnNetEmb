# coding:utf-8

from sklearn.datasets import fetch_20newsgroups
from gensim.models import word2vec
from bs4 import BeautifulSoup
import re
import nltk
import time

start = time.time()

news = fetch_20newsgroups(subset='all')
X, y = news.data, news.target


def news_to_sentences(news):
    news_text = BeautifulSoup(news).get_text()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(news_text)
    sentences = []
    for sent in raw_sentences:
        sentences.append(re.sub('[^a-zA-Z]', ' ', sent.lower().strip()).split())
    return sentences


# 句子词语列表化
sentences = []
for x in X:
    sentences.extend(news_to_sentences(x))

# 设置词语向量维度
num_featrues = 300
# 保证被考虑词语的最低频度
min_word_count = 20
# 设置并行化训练使用CPU计算核心数量
num_workers = 2
# 设置词语上下午窗口大小
context = 5
downsampling = 1e-3

model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_featrues, min_count=min_word_count, window=context,
                          sample=downsampling)

model.init_sims(replace=True)

# 输入一个路径，保存训练好的模型，其中./data/model目录事先要存在
model.save("word2vec_model/word2vec_gensim")
# model.wv.save_word2vec_format("data/model/word2vec_org","data/model/vocabulary",binary=False)
