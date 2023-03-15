import numpy as np


W = np.array([[0, 0], [0, 100], [0, 100]])
W_ct = np.array([-100, 50])
bc = 0

# initial (previous) states of the memory cell ct-1 and hidden state ht-1
ht = 0
ct = 0


bias = np.array([-100, 100, 0])


X = np.array([0, 0, 1, 1, 1, 0])

ht_history = []

for i, x in enumerate(X):
    ft_x = W[0, 0]*ht + W[0, 1] * x + bias[0]
    ft = 1/(1 + np.exp(-ft_x))

    it_x = W[1, 0]*ht + W[1, 1] * x + bias[1]
    it = 1/(1 + np.exp(-it_x))

    ot_x = W[2, 0]*ht + W[2, 1] * x + bias[2]
    ot = 1/(1 + np.exp(-ot_x))

    ct = ft * ct + it * np.tanh(W_ct[0] * ht + W_ct[1] * x + bc)
    ht = ot * np.tanh(ct)
    ht = round(ht)
    ht_history.append(ht)


print(ht_history)