import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


X = np.array([[2, 2], [2, 2.5], [2.5, 2], [-2, 2], [-2, 2.5], [-2.5, 2], [-2, -2],
              [-2.5, -2], [-2.5, -2], [2, -2], [2, -2.5], [2.4, -2.5]])
y = np.ones(X.shape[0])

X = np.append(X, [[-1, -1], [1, 1], [-1, 1], [1, -1]], axis=0)
y = np.append(y, np.ones(X.shape[0])*-1, axis=0)

df = pd.DataFrame(X, columns=['dim1', 'dim2'])
df['label'] = pd.DataFrame(y)

fig = plt.figure()
sns.scatterplot(data=df, x='dim1', y='dim2', hue='label')
fig.savefig('point_cloud.png')

df['dim3'] = df['dim1']**2+df['dim2']**2
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(df[df['label']==-1]['dim1'], df[df['label']==-1]['dim2'], df[df['label']==-1]['dim3'], marker='s')
ax.scatter(df[df['label']==1]['dim1'], df[df['label']==1]['dim2'], df[df['label']==1]['dim3'], marker='o')
fig.savefig('3d-point-cloud.png')

surface_x1, surface_x2 = np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1)
x1_grid, x2_grid = np.meshgrid(surface_x1, surface_x2)
x3_grid = x1_grid*x2_grid

fig3, ax3 = plt.subplots()
# plt.contourf(x1_grid, x2_grid, x3_grid, levels=50, cmap='jet')

fig3.savefig('contour.png')

