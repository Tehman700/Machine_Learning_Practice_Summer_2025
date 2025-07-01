from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)

print(X_train.shape)
print(y_train.shape)
X = iris.data
y = iris.target

print(y.shape)
print(X.shape)

knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train, y_train)

new_set = np.array([[1.2,1.8,2.4,1.0]])
print(new_set.shape)

prediction = knn.predict(new_set)
print("Prediction: {}".format(prediction))
print(iris['target_names'][prediction])



y_pred = knn.predict(X_test)
print(y_pred)

print(np.mean(y_pred == y_test))