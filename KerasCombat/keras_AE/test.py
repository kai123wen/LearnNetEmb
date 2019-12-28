import numpy as np

list = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]

x_train = np.array(list)

a = np.prod(x_train.shape[1:])
print(x_train.shape[1:])
print(a)

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
print(x_train)
