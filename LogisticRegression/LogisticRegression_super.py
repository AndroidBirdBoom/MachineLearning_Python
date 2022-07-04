import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import optimize

'''
逻辑回归算法：
           
跟线性回归差不多，其实就是在线性回归的基础上改的
线性回归是连续的函数 -> logistic函数 -> 限制连续函数只能在0-1中取值 -> 可以给出取0/1的可能性，进而做到分类  

逻辑回归步骤（归根结底还是求theta的值）：
h(theta) = 1/(1+e^-z) 
           z = theta^T*x
           
           -> 求theta最小值
           -> 先求代价函数最小值
           -> 利用梯度下降算法（正则化，防止过拟合）

'''


def LogisticRegression():
    data = np.loadtxt('data2.txt', delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    plot_data(X, y)

    # 创建多项式
    X = mapFeature(X[:, 0], X[:, 1])
    theta = np.zeros((X.shape[1], 1))
    initial_lambda = 0.1
    J = costFunction(theta, X, y, initial_lambda)

    result = optimize.fmin_bfgs(costFunction, theta, gradient, args=(X, y, initial_lambda))
    X = data[:, :-1]
    y = data[:, -1]
    plotDecisionBoundary(result, X, y)


# 画决策边界
def plotDecisionBoundary(theta, X, y):
    pos = np.where(y == 1)  # 找到y==1的坐标位置
    neg = np.where(y == 0)  # 找到y==0的坐标位置
    # 作图
    plt.figure(figsize=(15, 12))
    plt.plot(X[pos, 0], X[pos, 1], 'ro')  # red o
    plt.plot(X[neg, 0], X[neg, 1], 'bo')  # blue o
    # plt.title(u"决策边界", fontproperties=font)

    # u = np.linspace(30,100,100)
    # v = np.linspace(30,100,100)

    u = np.linspace(-1, 1.5, 50)  # 根据具体的数据，这里需要调整
    v = np.linspace(-1, 1.5, 50)

    z = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = np.dot(mapFeature(u[i].reshape(1, -1), v[j].reshape(1, -1)), theta)  # 计算对应的值，需要map

    z = np.transpose(z)
    plt.contour(u, v, z, [0, 0.01], linewidth=2.0)  # 画等高线，范围在[0,0.01]，即近似为决策边界
    # plt.legend()
    plt.show()


# 计算梯度
def gradient(initial_theta, X, y, inital_lambda):
    m = X.shape[0]
    h = sigmoid(np.dot(X, initial_theta.T))
    grad = (1 / m) * np.dot(X.T, h - y) + (inital_lambda / m) * initial_theta
    return grad


# 代价函数
def costFunction(initial_theta, X, y, inital_lambda):
    h = sigmoid(np.dot(X, initial_theta))
    m = X.shape[0]
    J = (-1 / m) * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h)))
    # 正则化
    J = J + inital_lambda / (2 * m) * np.sum(initial_theta*initial_theta)
    return J


# S型函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 映射为多项式
def mapFeature(X1, X2):
    '''
        这里以degree=2为例，映射为1,x1,x2,x1^2,x1,x2,x2^2
    '''
    out = np.ones((X1.shape[0], 1))
    degree = 2
    '''
    out = 1,x1,x2,x1**2,x1x2,x2**2
    '''
    for i in range(1, degree + 1):
        for j in range(i + 1):
            temp = X1 ** (i - j) * X2 ** j
            out = np.hstack((out, temp.reshape(X1.shape[0], 1)))

    return out


# 显示二维图形
def plot_data(X, y):
    ho = np.where(y == 1)
    ve = np.where(y == 0)
    plt.scatter(X[ho, 0], X[ho, 1], c='r')
    plt.scatter(X[ve, 0], X[ve, 1], c='b')
    plt.show()


# 测试逻辑回归函数
def testLogisticRegression():
    LogisticRegression()


if __name__ == "__main__":
    testLogisticRegression()
