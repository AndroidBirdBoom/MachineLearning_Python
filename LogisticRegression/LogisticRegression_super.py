import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def LogisticRegression():
    data = np.loadtxt('data2.txt', delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    plot_data(X, y)




# 显示二维图形
def plot_data(X, y):
    ho = np.where(y == 1)
    ve = np.where(y == 0)
    plt.plot(X[ho, 0], X[ho, 1], c='r')
    plt.plot(X[ve, 0], X[ve, 1], c='b')
    plt.show()


# 测试逻辑回归函数
def testLogisticRegression():
    LogisticRegression()


if __name__ == "__main__":
    testLogisticRegression()
