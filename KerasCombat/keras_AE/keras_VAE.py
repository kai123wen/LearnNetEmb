"""
变分自动编码器
参考网站：https://www.bilibili.com/video/av60862404?from=search&seid=11481053709271454780

- 该视频只是讲述了代码的编写过程，但是其中例如采样为什么是 均值 + （。。。那些就不写了） 没有说
    可以参考： https://www.cnblogs.com/nxf-rabbit75/p/10013568.html 讲的很详细

"""
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist

# 定义常数
batch_size = 100
# 下面会定义两个全连接层
# 第一个全连接层的输入是784，输出是256

original_dim = 784
intermediate_dim = 256
# 第二个全连接层的输入是 256，输出是2
# 为了平面的可视化
latent_dim = 2
epochs = 50

# encoder 部分
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_var / 2) * epsilon


z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# decoder 部分
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


def vae_loss(x, x_decoded_mean):
    # 交叉熵 表示的网络还原程度的损失函数
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    # 隐含变量与高斯分布相近程度的损失函数 。
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss


vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# shuffle：布尔值，表示是否在训练过程中每个epoch前随机打乱输入样本的顺序。
# validation_data：形式为（X，y）或（X，y，sample_weights）的tuple，是指定的验证集。此参数将覆盖validation_spilt。
vae.fit(x_train, x_train, shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(x_test, x_test))

encoder = Model(x, z_mean)
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
# plt.show()
plt.savefig('result.png')
