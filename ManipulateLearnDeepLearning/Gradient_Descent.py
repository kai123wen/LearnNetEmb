"""
梯度下降
"""

"""
以 y = x^2 为例
"""

import matplotlib.pyplot as plt
import numpy as np


def gd(eta):
    x = 10
    result = [x]
    for i in range(10):
        x -= eta * 2 * x
        result.append(x)
    print('the final result is : ', x)
    return result


def show_trace(res):
    x = np.arange(-10, 10, 0.1)
    y = x ** 2
    plt.plot(x, y)
    plt.plot(res, [x * x for x in res], '-o')
    plt.savefig('result.jpg')


x = gd(0.2)
show_trace(x)
