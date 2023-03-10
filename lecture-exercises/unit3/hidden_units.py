import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def relu(z):
    return max(0, z)


def fivez_minus2(z):
    return 5*z -2


if __name__ == '__main__':

    X = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
    y = np.array([1, -1, -1, 1])

    W = np.array([[1, -1], [-1, 1]])
    W_0 = np.array([1, 1])

    transform_coords = np.empty(shape=X.shape)

    for index, example in enumerate(X):
        z_1 = np.dot(example.T, W[0]) + W_0[0]
        z_2 = np.dot(example.T, W[1]) + W_0[1]
        transform_coords[index, 0] = z_1
        transform_coords[index, 1] = z_2

    relu_vec = np.vectorize(relu)
    alt_vec = np.vectorize(fivez_minus2)
    df = pd.DataFrame(transform_coords, columns=['dim1', 'dim2'])
    df['label'] = pd.DataFrame(y)

    fig = plt.figure()
    sns.scatterplot(data=df, x='dim1', y='dim2', hue='label')
    plt.tight_layout()
    plt.grid()
    plt.show()

    print(transform_coords)