import numpy as np

Wmatrix = np.array([[1, 0, -1], [0, 1, -1], [-1, 0, -1], [0, -1, -1]])

Vmatrix = np.array([[1, 1, 1, 1, 0], [-1, -1, -1, -1, 2]])

inputunits = np.array([3, 14])

Z = np.dot(inputunits, Wmatrix[::, 0:-1].T) + Wmatrix[:, -1]
act_func = lambda x: max(x, 0)
act_func_vec = np.vectorize(act_func)

f_z = act_func_vec(Z)

U = np.dot(f_z, Vmatrix[::, 0:-1].T) + Vmatrix[:, -1]

f_u = act_func_vec(U)

softmax = lambda x: np.exp(x)
softmax_vec = np.vectorize(softmax)

exp_f_u = softmax_vec(f_u)

output = exp_f_u/np.sum(exp_f_u)

print(output)