import numpy as np


def hinge_loss(z):
    if z < 1:
        return 1 - z
    else:
        return 0


def squared_error(z):
    return z**2/2


def empirical_risk(theta, X, y, loss_func):
    n = X.shape[0]
    sum = 0
    for t in range(n):
        z = y[t] - np.dot(theta, X[t])
        sum += loss_func(z)
    return sum/n


if __name__ == '__main__':
    X = np.array([[1, 0, 1], [1, 1, 1], [1, 1, -1], [-1, 1, 1]])
    y = [2, 2.7, -0.7, 2]

    theta = [0, 1, 2]

    sum = empirical_risk(theta, X, y, squared_error)
    print(sum)