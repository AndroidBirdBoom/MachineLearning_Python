import numpy as np
from scipy import io
import matplotlib.pyplot as plt

from sklearn import svm


def SVM():
    data = np.loadtxt('data.txt', delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    plt1 = plot_data(X, y)
    plt1.show()
    model = svm.SVC(gamma=20).fit(X, y)
    plot_decisionBoundary(X, y, model, 'no')
    # 线性
    data1 = io.loadmat('data1.mat')
    X = data1['X']
    y = data1['y']
    plt1 = plot_data(X, y)
    plt1.show()
    model = svm.SVC(kernel='linear').fit(X, y)
    plot_decisionBoundary(X, y, model)
    data2 = io.loadmat('data2.mat')
    X = data2['X']
    y = data2['y']
    plt1 = plot_data(X, y)
    plt1.show()
    model = svm.SVC(gamma=100).fit(X, y)
    plot_decisionBoundary(X, y, model, 'no')
    data3 = io.loadmat('data3.mat')
    X = data3['X']
    y = data3['y']
    plt1 = plot_data(X, y)
    plt1.show()
    model = svm.SVC(gamma=100).fit(X, y)
    plot_decisionBoundary(X, y, model, 'no')


# 画决策边界
def plot_decisionBoundary(X, y, model, class_='linear'):
    plot = plot_data(X, y)
    if class_ == 'linear':
        w = model.coef_[0]
        b = model.intercept_
        xd = np.linspace(np.min(X[:, 0]), np.max(X[:, 1]), 100)
        # w0*x + w1*y + b = 0
        yd = -(w[0] * xd + b) / w[1]
        plot.plot(xd, yd, 'b-')
        plot.show()
    else:
        x1 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
        x2 = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
        X1, X2 = np.meshgrid(x1, x2)
        vel = np.zeros(X1.shape)
        for i in range(X1.shape[0]):
            X = np.hstack((X1[:, i].reshape(-1, 1), X2[:, i].reshape(-1, 1)))
            vel[:, i] = model.predict(X)

        plot.contour(X1, X2, vel, [0, 1], color='blue')
        plot.show()


# 作图
def plot_data(X, y):
    y1 = np.where(y == 1)
    y0 = np.where(y == 0)
    plt.plot(X[y0, 0], X[y0, 1], 'ro', ms=4)
    plt.plot(X[y1, 0], X[y1, 1], '^g', ms=4)
    return plt


if __name__ == "__main__":
    SVM()
