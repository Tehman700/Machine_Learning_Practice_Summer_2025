import math
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np


df = pd.read_csv('D:/Machine Learning Pycharm/iris_dataset/Iris.csv')
df = df.drop('Id', axis =1)

print(df.head())

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
Y = df['Species']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, Y_train)


y_pred = knn.predict(X_test)
print(y_pred.shape)

print("Accuracy: ",  accuracy_score(Y_test, y_pred))

sample = np.array([[5, 1, 4, 0]])  # example input
prediction = knn.predict(sample)
print("Predicted species for sample:", prediction[0])

