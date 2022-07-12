import numpy as np
from matplotlib import pyplot as plt
from scipy import io


def kmean(x0, x1, cp, cp_old):
    for index in range(10):
        c0 = getDistance(x0, x1, cp[0])
        c1 = getDistance(x0, x1, cp[1])
        c2 = getDistance(x0, x1, cp[2])
        index0 = np.where((c0 < c1) & (c0 < c2))
        index1 = np.where((c1 < c0) & (c1 < c2))
        index2 = np.where((c2 < c0) & (c2 < c1))
        plt.plot(x0[index0], x1[index0], 'go', ms=3)
        plt.plot(x0[index1], x1[index1], 'bo', ms=3)
        plt.plot(x0[index2], x1[index2], 'yo', ms=3)
        plt.plot(cp[:, 0], cp[:, 1], 'rx', ms=8)
        plt.show()

        cp_old = cp
        cp[0, :] = np.mean(np.array([x0[index0], x1[index0]]), axis=1)
        cp[1, :] = np.mean(np.array([x0[index1], x1[index1]]), axis=1)
        cp[2, :] = np.mean(np.array([x0[index2], x1[index2]]), axis=1)
        print(cp)


def getDistance(x0, x1, cp):
    return (x0 - cp[0]) ** 2 + (x1 - cp[1]) ** 2


if __name__ == "__main__":
    data = io.loadmat('data.mat')
    X = data['X']
    x0 = X[:, 0]
    x1 = X[:, 1]
    plt.plot(x0, x1, 'go', ms=3)

    # 随机选点
    centerpoint = np.array([[3, 3], [6, 2], [8, 5]],dtype=float)
    plt.plot(centerpoint[:, 0], centerpoint[:, 1], 'rx', ms=8)
    plt.show()

    kmean(x0, x1, centerpoint, centerpoint)
