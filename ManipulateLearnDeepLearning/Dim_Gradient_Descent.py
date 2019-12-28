"""
多维梯度下降
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

eta = 0.1  # 学习率


# 目标函数
def fun(x1, x2):
    return x1 ** 2 + 2 * x2 ** 2


# 两个变量梯度下降的函数
def gd_2dd(x1, x2):
    return x1 - eta * 2 * x1, x2 - eta * 4 * x2


def train_2d(trainer):
    x1, x2 = -30, 30
    result_one = [x1]
    result_two = [x2]
    for i in range(10):
        x1, x2 = trainer(x1, x2)
        result_one.append(x1)
        result_two.append(x2)
    print("the final x1 and x2 are : ", x1, x2)
    return result_one, result_two


def show_trace_2d(f, result):
    fig = plt.figure()
    ax = Axes3D(fig)
    x1 = np.arange(-10, 10, 0.1)
    x2 = np.arange(-10, 10, 0.1)
    X1, X2 = np.meshgrid(x1, x2)  # 网格的创建，这个是关键
    y = f(X1, X2)
    ax.plot_surface(X1, X2, y, rstride=1, cstride=1, cmap='rainbow')
    result_x1 = np.array(result[0])
    result_x2 = np.array(result[1])
    result_y = f(result_x1, result_x2)
    ax.plot(result_x1, result_x2, result_y,'-o',c='r')
    plt.show()


show_trace_2d(fun, train_2d(gd_2dd))
