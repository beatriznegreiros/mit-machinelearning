import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

X = np.array([[-4, 2], [-2, 1], [-1, -1], [2, 2], [1, -2]])

y = np.array([1, 1, -1, -1, -1])

# Plotting the points
df = pd.DataFrame(X, columns=['dim1', 'dim2'])
df['label'] = pd.DataFrame(y)

fig = plt.figure()
sns.scatterplot(data=df, x='dim1', y='dim2', hue='label')
plt.tight_layout()
plt.grid()
# Plotting possible theta and theta 0
x_t = np.array(range(-4, 4, 1))
reta = 0.5*x_t+1.5

plt.plot(x_t, reta)
fig.savefig('save.png')
t_times = range(0, 3)


# Example of init which which the perceptron doesnt make any mistakes
theta = np.array([-0.5, 1])
theta_0 = 1.5
n_mistakes = 0
progress_theta = []
explicit_mistakes = np.zeros(shape=y.shape[0])
for t in t_times:
    for index, x in enumerate(X):
        # print('yx*theta: ', y[index] * np.dot(x, theta))
        if y[index] * np.dot(theta, x) + theta_0 <= 0:
            theta = theta + y[index] * x
            theta_0 = theta_0 + y[index]
            progress_theta.append(theta)
            print('theta: ', theta)
            n_mistakes += 1
            explicit_mistakes[index] += 1
print('The perceptron did {} mistakes until convergence'.format(n_mistakes))