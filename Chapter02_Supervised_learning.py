import mglearn
from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# # Load the Forge dataset
# X, y = mglearn.datasets.make_wave(n_samples=40)
# plt.plot(X, y, 'o')
# plt.ylim(-3, 3)
# plt.xlabel("Feature")
# plt.ylabel("Target")
#
#
# # Show the plot
# plt.show()



cancer = load_breast_cancer()

df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
print(cancer.keys())

df['target'] = cancer.target
print(df.describe())

benign = df[df['target'] == 0]
malignant = df[df['target'] == 1]


print(benign.shape)
print(malignant.shape)

print(cancer.feature_names)

X_train, X_test, y_train, y_test = train_test_split(cancer['data'], cancer['target'], random_state=0)

print(X_train.shape)
print(y_train.shape)

X = cancer.data
Y = cancer.target

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)


print("Test set predictions: {}".format(knn.predict(X_test)))

print("Test set accuracy: {:.2f}".format(knn.score(X_test, y_test)))
