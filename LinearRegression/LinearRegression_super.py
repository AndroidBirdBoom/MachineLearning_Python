import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

'''
线性回归步骤：
目标是求theta和X的关系式
y = t1 + t2*x1 + t3*x2
    -> 转化为求theta的值
    -> 转化为计算代价函数的最小值
    -> 需要先进行均一化（更快的迭代梯度下降到最小值）
    -> 进行梯度下降算法
    -> 求出代价函数最小
    -> 求出最合适的theta
    -> 带入线性回归方程
'''


def testLinearRegression():
    linearRegression()


def linearRegression(alpha=0.01, num_iters=400):
    data = loadtxtAndcsv_data('data.txt')
    X = data[:, 0:2]
    Y = data[:, -1].reshape(X.shape[0], 1)
    # 先均一化
    X = featureNormaliza(X)
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # 加一行，这行的目的就是作为常数项
    # 梯度下降
    thera = np.zeros((X.shape[1], 1))
    gradientDescent(X, Y, thera, alpha, num_iters)


# 梯度下降算法
def gradientDescent(X, y, theta, alpha, num_iters):
    m = X.shape[0]
    cost_h = np.zeros((num_iters, 1))
    theta_file = open('theta.txt', 'w')
    for i in range(num_iters):
        hx = np.dot(X, theta)          # 注意hx使用的是更新后的theta!!!!
        theta = theta - ((alpha / m) * np.dot(X.T, hx - y))
        theta_file.write(str(theta)+'\n')
        cost_h[i] = computerCost(X, y, theta)

    plt.plot(np.array(np.arange(1, num_iters + 1)), cost_h)
    plt.xlabel('times')
    plt.ylabel('cost')
    plt.show()


# 计算代价函数
def computerCost(X, y, theta):
    m = X.shape[0]
    hx = np.dot(X, theta)
    cost = np.sum((hx - y) * (hx - y), 0) / (2 * m)
    return cost


# 归一化feature( x = (x - mu)/theta)
def featureNormaliza(X):
    mean = np.mean(X, 0)
    theta = np.std(X, 0)
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - mean[i]) / theta[i]

    # plt.scatter(X[:, 0], X[:, 1])
    # plt.title("featureNormaliza")
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    return X


def loadtxtAndcsv_data(file_name):
    pattern = "^.+[.]txt$"
    if re.match(pattern, file_name):
        return np.loadtxt(file_name, delimiter=',')
    elif re.match('^.+[.]csv$', file_name):
        return pd.read_csv(file_name).to_numpy()
    else:
        print("输入文件不合法！")


if __name__ == "__main__":
    testLinearRegression()
