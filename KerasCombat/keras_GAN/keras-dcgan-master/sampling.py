from keras.layers import UpSampling2D
import numpy as np
import tensorflow as tf

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
x = x.reshape(4, 4)
x = tf.convert_to_tensor(x)
y = UpSampling2D(size=(2, 2))(x)
with tf.Session() as sess:
    print(y.eval())
