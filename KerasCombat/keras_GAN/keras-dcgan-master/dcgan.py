from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math

"""
学习网址：https://www.bilibili.com/video/av21255701?from=search&seid=4463323497592419170

卷积与反卷积 图解： https://www.jianshu.com/p/cba362d84c75
"""


# noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
def generator_model():
    """
    最开始输入的是一个 100 维的向量（噪声）
    :return:
    """
    model = Sequential()
    # 将 100 维 升高到 1024
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    # 将其做成图片 128 是通道大小，7*7 是图片大小
    # 之后通过两次 UpSampling2D 操作，来变成 28*28
    model.add(Dense(128 * 7 * 7))
    # 这里很重要，因为不加的话就会崩掉,虽然我还不知道是为什么
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
    # 经过 UpSampling2D 得到 14*14 的图像
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    # 经过 UpSampling2D 得到 28*28 的图像
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    """
    判决其实就是一个二分类问题
    :return:
    """
    model = Sequential()
    model.add(
        Conv2D(64, (5, 5),
               padding='same',
               input_shape=(28, 28, 1))
    )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(g, d):
    """
    将判别器固定，来训练生成器
    :param g: 生成器
    :param d: 判别器
    :return: 生成 d 固定，g可变的网络
    """
    model = Sequential()
    model.add(g)
    # 将判别器固定住。也就是说判别器的所有参数是不能动的
    d.trainable = False
    model.add(d)
    return model


def combine_images(generated_images):
    """
    合并图像
    :param generated_images:
    :return:
    """
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = \
            img[:, :, 0]
    return image


def train(BATCH_SIZE):
    # 预处理
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]
    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
    d = discriminator_model()
    g = generator_model()

    # 固定判别器，只更新生成器的网路
    d_on_g = generator_containing_discriminator(g, d)
    # 定义 优化器
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    # 开始训练，训练100个周期
    for epoch in range(100):
        print("Epoch is", epoch)  # 打印这是第几个训练周期
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))  # 打印训练多少频次
        # 一个周期训练多少批（等于总数除以一批的数量）
        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]  # 获得一批的真实图片
            generated_images = g.predict(noise, verbose=0)  # 生成一批虚假图片
            # 每隔多少次生成一次图片，并保存成png格式
            if index % 20 == 0:
                image = combine_images(generated_images)
                image = image * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    "result/" + str(epoch) + "_" + str(index) + ".png")

            #  将图像的数据和生成的数据拼接在一起，作为判别器的输入
            X = np.concatenate((image_batch, generated_images))
            # label,第一个 batch_size 是 1，第二个 batch_size 中是 0
            # 前一批真实图片的标签为1，后一批虚假图片的标签为0
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE

            # 步骤一 ： 首先是训练 d （判别器）
            d_loss = d.train_on_batch(X, y)

            print("batch %d d_loss : %f" % (index, d_loss))
            # 生成噪音
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))

            # 训练完之后，就设置判别器不能再次训练
            d.trainable = False

            # 步骤二 ：训练生成器，
            # [1] * BATCH_SIZE  这个参数表示的是 希望该网络的输出是啥
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            # 生成器完成，判别器继续
            # 打开判别器D的参数，为了新一轮的循环
            d.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            # 保存 g 和 d 的参数
            if index % 10 == 9:
                g.save_weights('generator', True)
                d.save_weights('discriminator', True)


def generate(BATCH_SIZE, nice=False):
    """
    使用得到的生成器
    :param BATCH_SIZE:
    :param nice:
    :return:
    """
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE * 20, 100))
        generated_images = g.predict(noise, verbose=1)
        # predict 和 predict_on_batch 是有区别的
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE * 20)
        index.resize((BATCH_SIZE * 20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        # 生成一个噪音
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        # 根据噪音得到图像
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
