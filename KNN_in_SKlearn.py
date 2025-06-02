from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris = load_iris()


X = iris.data
y = iris.target

print(y.shape)
print(X.shape)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X, y)

new_set = np.array([[5,5,5,5]])
print(new_set.shape)

prediction = knn.predict(new_set)
print("Prediction: {}".format(prediction))
print(iris['target_names'][prediction])
