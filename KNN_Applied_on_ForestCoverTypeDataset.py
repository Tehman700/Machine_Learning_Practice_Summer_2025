import math
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np



df = pd.read_csv('D:/Machine Learning Pycharm/covtype_labeled.csv')
print(df.head())

print(df.describe())

Y = df['Cover_Type']
t = df.columns.tolist()
X = df[[i for i in t if i != 'Cover_Type']]
#
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
#
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(X_train, Y_train)
#
# y_pred = knn.predict(X_test)
# print("Accuracy: ",  accuracy_score(Y_test, y_pred))
