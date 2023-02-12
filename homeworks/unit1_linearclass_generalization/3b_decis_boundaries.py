import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


X = [[-1, 1], [1, -1], [1, 1], [2, 2]]

y = [1, 1, -1, -1]

# Plotting the points
df = pd.DataFrame(X, columns=['dim1', 'dim2'])
df['label'] = pd.DataFrame(y)

fig = plt.figure()
sns.scatterplot(data=df, x='dim1', y='dim2', hue='label')
plt.tight_layout()
plt.grid()

# Circle with center at the origin
x_c = -1
y_c = -1

x_circle = np.arange(-1, 1.2, step=0.1)
radius = 3*0.5*np.sqrt(2)
y_circle = np.sqrt(radius**2-(x_circle+1)**2)-1

plt.plot(x_circle, y_circle)

fig.savefig('3b_decis_boundaries.png')