import numpy as np


def empirical_risk(theta, X, y):
    n = X.shape[0]
    sum = 0
    for t in range(n):
        z = y[t] - np.dot(theta, X[t])
        if z < 1:
            sum += 1-z
    return sum/n


if __name__ == '__main__':
    X = np.array([[1, 0, 1], [1, 1, 1], [1, 1, -1], [-1, 1, 1]])
    y = [2, 2.7, -0.7, 2]

    theta = [0, 1, 2]

    sum = empirical_risk(theta, X, y)
    print(sum)