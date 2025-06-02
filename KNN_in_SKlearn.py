from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()


X = iris.data
y = iris.target

print(y.shape)
print(X.shape)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X, y)

import numpy as np