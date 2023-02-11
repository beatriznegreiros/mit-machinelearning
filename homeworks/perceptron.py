import numpy as np

# X = np.array([[-1, -1], [1, 0], [-1, 1.5]])
X = np.array([[-1, -1], [1, 0], [-1, 10]])

y = [1, -1, 1]
# x2 = np.array([1, 0])
# x3 = np.array()
t_times = range(0, 40)

# theta = np.array([-1, -1])
theta = np.array([1, 0])
n_mistakes = 0

# Algorithm always starting to loop from x1
for t in t_times:
    for index, x in enumerate(X):
        # print('yx*theta: ', y[index] * np.dot(x, theta))
        if y[index] * np.dot(x, theta) <= 0:
            theta = theta + y[index] * x
            print('theta: ', theta)
            n_mistakes += 1

print('The perceptron did {} mistakes until convergence'.format(n_mistakes))