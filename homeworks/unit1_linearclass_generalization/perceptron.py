import numpy as np


# Algorithm always starting to loop from x1
def perceptron(X, y, theta, t_times):
    n_mistakes = 0
    progress_theta = []
    explicit_mistakes = np.zeros(shape=y.shape[0])
    for t in t_times:
        for index, x in enumerate(X):
            # print('yx*theta: ', y[index] * np.dot(x, theta))
            if y[index] * np.dot(theta, x) <= 0:
                theta = theta + y[index] * x
                progress_theta.append(theta)
                print('theta: ', theta)
                n_mistakes += 1
                explicit_mistakes[index] += 1
    print('The perceptron did {} mistakes until convergence'.format(n_mistakes))
    return progress_theta, n_mistakes, explicit_mistakes


if __name__ == '__main__':
    X = np.array([[-1, -1], [1, 0], [-1, 1.5]])
    # X = np.array([[-1, -1], [1, 0], [-1, 10]])

    y = np.array([1, -1, 1])
    # x2 = np.array([1, 0])
    # x3 = np.array()
    t_times = range(0, 100)

    # theta = np.array([-1, -1])
    theta = np.array([1, 0])

    a, b, c = perceptron(X, y, theta, t_times)
    print(a, b, c)
