import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# def opt_func(theta, x, y, lamb):
#     opt_func = max(0, 1-y*np.dot(theta, x))+lamb/2*np.linalg.norm(theta)**2
#     return opt_func


def opt_func(x1, x2, y, lamb, theta1, theta2):
    return 1-y*(theta1*x1+theta2*x2)+lamb/2*(theta1**2+theta2**2)


if __name__ == '__main__':
    lamb = 0.5
    y = 1
    x = [1, 0]
    x1, x2 = x[0], x[1]
    fig, ax = plt.subplots()
    theta_1 = np.arange(-5, 5, 0.1)
    theta_2 = np.arange(-5, 5, 0.1)

    theta_1grid, theta_2grid = np.meshgrid(theta_1, theta_2)

    grad_func_eval = opt_func(x1, x2, y, lamb, theta_1grid, theta_2grid)
    plt.contourf(theta_1grid, theta_2grid, grad_func_eval, levels=50, cmap='jet')
    plt.plot(2, 0, '*', color='white')
    print(opt_func(x1, x2, y, lamb, 1, 0))
    plt.tight_layout()
    plt.ylabel('theta_2')
    plt.xlabel('theta_1')
    plt.grid()
    cbar = plt.colorbar()
    cbar.solids.set_edgecolor('face')
    plt.draw()
    plt.axvline(x=1, color='fuchsia', label='theta_0<=1 so that Loss_h>=0')
    plt.legend()
    plt.tight_layout()
    fig.savefig('figs/opt_func.png')

