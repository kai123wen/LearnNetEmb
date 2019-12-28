from keras.layers import Concatenate, Dot
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Sequential

vocab_size = 5000  # 词典大小
embed_size = 300  # 输出向量的维度，Google在最新发布的基于Google news数据集训练的模型中使用的就是300个特征的词向量
word_model = Sequential()
"""
嵌入层将正整数（下标）转换为具有固定大小的向量，如[[4],[20]]->[[0.25,0.1],[0.6,-0.2]]
Embedding层只能作为模型的第一层

"""
word_model.add(Embedding(vocab_size, embed_size, embeddings_initializer="glorot_uniform", input_shape=1))
"""
Reshape层用来将输入shape转换为特定的shape
keras.layers.core.Reshape(target_shape)
target_shape：目标shape，为整数的tuple，不包含样本数目的维度（batch大小）
(batch_size,)+target_shape
"""
word_model.add(Reshape((embed_size,)))

context_model = Sequential()
context_model.add(Embedding(vocab_size, embed_size, embeddings_initializer='glorot_uniform', input_length=1))
context_model.add(Reshape((embed_size,)))

model = Sequential()
model.add(Dot([word_model, context_model]))
model.add(Dense(1, init='glorot_uniform', activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam')
