"""
notes:
Extra dimensions might throw this off, 2 works for now.

target = our labelled data. so the machine knows how many of each class
it should roughly have.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors

df = pd.read_csv("wine.csv")

d1 = np.asarray(df['Alcohol'])
d2 = np.asarray(df['Color intensity'])
h = .01

d1d2 = []
for i in range(178):
  trainv = np.array([d1[i],d2[i]])
  d1d2.append(trainv)

d1d2 = np.asarray(d1d2)

#target tells what the training data has for labels
#in this case 71 wines are class 2, 59 wines are class 1, 48 wines are class 3
target = [2 for _ in range(71)]
target.extend([1 for _ in range(59)])
target.extend([3 for _ in range(48)])

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

clf = neighbors.KNeighborsClassifier(7)
clf.fit(d1d2,target)

#get points for plotting descision boundary
x_min, x_max = d1.min() - 1, d1.max() + 1
y_min, y_max = d2.min() - 1, d2.max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(d1, d2, cmap=cmap_bold,c=target ,edgecolor='k', s=20)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel("Alcohol %")
plt.ylabel("Wine Colour Intensity")
plt.title("3 Class Wine classification")

plt.show()