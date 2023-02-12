import numpy as np

X = np.array([[-1, -1], [1, 0], [-1, 1.5]])
# X = np.array([[-1, -1], [1, 0], [-1, 10]])

y = [1, -1, 1]

t_times = range(0, 40)

# theta = np.array([-1, -1])
start = 1

theta = X[start]
n_mistakes = 0


# Algorithm always starting to loop from x1
for t in t_times:
    for i in range(start, X.shape[0]):
        # print('yx*theta: ', y[index] * np.dot(x, theta))
        if y[i] * np.dot(X[i], theta) <= 0:
            theta = theta + y[i] * X[i]
            print('theta: ', theta)
            n_mistakes += 1
        start = 0

print('The perceptron did {} mistakes until convergence'.format(n_mistakes))