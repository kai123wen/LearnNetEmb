import scipy
import scipy.io as sio
import numpy as np
import argparse
import time
from liblinearutil import *


# for multi-class classification

def load_data(file):
    # load data
    net_dict = sio.loadmat(file)
    network = net_dict['network']
    network = network.toarray()
    label = net_dict['group']
    label = np.transpose(label)
    label = label[0]
    return network, label


def save_results(results, file):
    f = open(file, 'a')
    N, D = results.shape[0], results.shape[1]
    for i in range(N):
        for n in range(D):
            f.write(str(results[i, n]) + '\t')
            if n == (D - 1): f.write('\n')
    f.close()


def mcc_liblinear_one_file(args):
    netFile = args.input_adj
    resultTxt = args.resultTxt
    repFile = args.rep

    network, labels = load_data(netFile)
    N = network.shape[0]
    results = np.zeros((1, 9))

    rep = sio.loadmat(repFile)
    rep = rep['rep']
    tmp = np.zeros((10, 9))

    SEED = 123456
    np.random.seed(SEED)
    IDX = []
    for i in range(10):
        IDX.append(np.random.permutation(N))

    for i in range(9):
        for j in range(10):
            Ntr = int((i + 1.0) / 10 * N)

            IDXtr = IDX[j][0:Ntr]
            IDXts = IDX[j][Ntr:]

            Xtr = scipy.sparse.csr_matrix(rep[IDXtr])
            Xts = scipy.sparse.csr_matrix(rep[IDXts])

            Ytr = scipy.asarray(labels[IDXtr])
            Yts = scipy.asarray(labels[IDXts])

            # cmd = '-s 2 -c {} -q'.format((i+1.0)/200)
            cmd = '-s 2 -c 1 -q'
            m = train(Ytr, Xtr, cmd)
            p_label, p_acc, p_val = predict(Yts, Xts, m)  # accuracy in p_acc[0]
            tmp[j, i] = p_acc[0]

    results[0, :] = np.sum(tmp, axis=0) / 10
    return results
