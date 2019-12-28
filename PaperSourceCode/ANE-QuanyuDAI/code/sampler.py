import numpy as np
import random
from math import *


# for generating samples from prior distribution

def one_hot_vector(label_batch, label_num):
    '''
    transform label_batch to one_hot_vector
    '''
    one_hot = np.zeros((label_batch.shape[0], label_num))
    for i in range(label_batch.shape[0]):
        if label_batch[i] == 100:
            one_hot[i, label_num - 1] = 1
        else:
            one_hot[i, int(label_batch[i])] = 1
    return one_hot


def uniform(batchsize, n_dim, minv=-1, maxv=1):
    # uniformly generate noise: (batchsize, n_dim)
    return np.random.uniform(minv, maxv, (batchsize, n_dim)).astype(np.float32)


def gaussian(batchsize, n_dim, mean=0, var=1):
    return np.random.normal(mean, var, (batchsize, n_dim)).astype(np.float32)


def gaussian_mixture_sampling(n_dim, label_indices, n_labels):
    if n_dim % 2 != 0:
        raise Exception("n_dim must be a multiple of 2.")

    def sample(x, y, label, n_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    batchsize = label_indices.shape[0]
    x_var = 0.5
    y_var = 0.05
    x = np.random.normal(0, x_var, (batchsize, n_dim / 2))
    y = np.random.normal(0, y_var, (batchsize, n_dim / 2))
    z = np.empty((batchsize, n_dim), dtype=np.float32)
    for batch in range(batchsize):
        if label_indices[batch] != 100:
            # for labeled nodes, sampling from a specific component
            for zi in range(n_dim / 2):
                z[batch, zi * 2:zi * 2 + 2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
        else:
            # for unlabeled nodes, sampling from full mixture distribution
            for zi in range(n_dim / 2):
                z[batch, zi * 2:zi * 2 + 2] = sample(x[batch, zi], y[batch, zi], random.randint(0, n_labels - 1),
                                                     n_labels)
    return z


def swiss_roll_sampling(batchsize, n_dim, label_indices, n_labels):
    def sample(label, n_labels):
        uni = np.random.uniform(0.0, 1.0) / float(n_labels) + float(label) / float(n_labels)
        r = sqrt(uni) * 3.0
        rad = np.pi * 4.0 * sqrt(uni)
        x = r * cos(rad)
        y = r * sin(rad)
        return np.array([x, y]).reshape((2,))

    z = np.zeros((batchsize, n_dim), dtype=np.float32)
    for batch in range(batchsize):
        if label_indices[batch] != 100:
            for zi in range(n_dim / 2):
                z[batch, zi * 2:zi * 2 + 2] = sample(label_indices[batch], n_labels)
        else:
            for zi in range(n_dim / 2):
                z[batch, zi * 2:zi * 2 + 2] = sample(random.randint(0, n_labels - 1), n_labels)
    return z


def gaussian_mixture(batchsize, n_dim, n_labels):
    if n_dim % 2 != 0:
        raise Exception("n_dim must be a multiple of 2.")

    def sample(x, y, label, n_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 0.5
    y_var = 0.05
    x = np.random.normal(0, x_var, (batchsize, n_dim / 2))
    y = np.random.normal(0, y_var, (batchsize, n_dim / 2))
    z = np.empty((batchsize, n_dim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(n_dim / 2):
            z[batch, zi * 2:zi * 2 + 2] = sample(x[batch, zi], y[batch, zi], random.randint(0, n_labels - 1), n_labels)
    return z


def supervised_gaussian_mixture(batchsize, n_dim, label_indices, n_labels):
    if n_dim % 2 != 0:
        raise Exception("n_dim must be a multiple of 2.")

    def sample(x, y, label, n_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 0.5
    y_var = 0.05
    x = np.random.normal(0, x_var, (batchsize, n_dim / 2))
    y = np.random.normal(0, y_var, (batchsize, n_dim / 2))
    z = np.empty((batchsize, n_dim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(n_dim / 2):
            z[batch, zi * 2:zi * 2 + 2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
    return z


def swiss_roll(batchsize, n_dim, n_labels):
    def sample(label, n_labels):
        uni = np.random.uniform(0.0, 1.0) / float(n_labels) + float(label) / float(n_labels)
        r = sqrt(uni) * 3.0
        rad = np.pi * 4.0 * sqrt(uni)
        x = r * cos(rad)
        y = r * sin(rad)
        return np.array([x, y]).reshape((2,))

    z = np.zeros((batchsize, n_dim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(n_dim / 2):
            z[batch, zi * 2:zi * 2 + 2] = sample(random.randint(0, n_labels - 1), n_labels)
    return z


def supervised_swiss_roll(batchsize, n_dim, label_indices, n_labels):
    def sample(label, n_labels):
        uni = np.random.uniform(0.0, 1.0) / float(n_labels) + float(label) / float(n_labels)
        r = sqrt(uni) * 3.0
        rad = np.pi * 4.0 * sqrt(uni)
        x = r * cos(rad)
        y = r * sin(rad)
        return np.array([x, y]).reshape((2,))

    z = np.zeros((batchsize, n_dim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(n_dim / 2):
            z[batch, zi * 2:zi * 2 + 2] = sample(label_indices[batch], n_labels)
    return z
