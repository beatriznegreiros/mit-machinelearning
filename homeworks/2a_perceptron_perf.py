import numpy as np

X = np.array([[-4, 2], [-2, 1], [-1, -1], [2, 2], [1, -2]])
y = np.array([1, 1, -1, -1, -1])
theta = [0, 0]
theta_0 = 0

time_misclassified = np.array([1, 0, 2, 1, 0])
until_now_misclassified = np.array([0, 0, 0, 0, 0])
t_times = 40

for t in range(0, t_times):
    for index, x in enumerate(X):
        # print('yx*theta: ', y[index] * np.dot(x, theta))
        if y[index] * np.dot(theta, x) + theta_0 <= 0:
            theta = theta + y[index] * x
            theta_0 = theta_0 + y[index]
            print('theta: ', theta)
            print('theta_0: ', theta_0)
            until_now_misclassified[index] += 1
            print(until_now_misclassified)
            if np.array_equal(time_misclassified, until_now_misclassified):
                print('breaking the loop...')
                break

