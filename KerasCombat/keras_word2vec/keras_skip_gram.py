"""
参考 keras深度学习实战 112 页的内容
"""

from keras.preprocessing.text import *
from keras.preprocessing.sequence import skipgrams

text = "I love green eggs and ham ."
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

word2id = tokenizer.word_index
id2word = {v: k for k, v, in word2id.items()}

wids = [word2id[w] for w in text_to_word_sequence(text)]
pairs, labels = skipgrams(wids, len(word2id))
print("pairs: ", pairs)
print("labels: ", labels)
for i in range(10):
    print(" {0} ({1}) , {2} ({3}) -> {4}".format(id2word[pairs[i][0]], pairs[i][0], id2word[pairs[i][1]],
                                                 pairs[i][1],
                                                 labels[i]))
