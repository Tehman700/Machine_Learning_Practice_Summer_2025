import mglearn
import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
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

X_train, X_test, y_train, y_test = train_test_split(cancer['data'], cancer['target'], random_state=1)

print(X_train.shape)
print(y_train.shape)

X = cancer.data
Y = cancer.target

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)


print("Test set predictions: {}".format(knn.predict(X_test)))

print("Test set accuracy: {:.2f}".format(knn.score(X_test, y_test)))


sample = [
    14.5,   # mean radius
    20.3,   # mean texture
    95.0,   # mean perimeter
    660.0,  # mean area
    0.100,  # mean smoothness
    0.120,  # mean compactness
    0.080,  # mean concavity
    0.050,  # mean concave points
    0.180,  # mean symmetry
    0.060,  # mean fractal dimension
    0.40,   # radius error
    1.10,   # texture error
    2.50,   # perimeter error
    30.0,   # area error
    0.007,  # smoothness error
    0.025,  # compactness error
    0.031,  # concavity error
    0.011,  # concave points error
    0.020,  # symmetry error
    0.003,  # fractal dimension error
    16.5,   # worst radius
    28.0,   # worst texture
    110.0,  # worst perimeter
    850.0,  # worst area
    0.140,  # worst smoothness
    0.25,   # worst compactness
    0.20,   # worst concavity
    0.10,   # worst concave points
    0.30,   # worst symmetry
    0.080   # worst fractal dimension
]


predictions = knn.predict([sample])
print(predictions)


# Select 50 random test samples
np.random.seed(0)
indices = np.random.choice(len(X_test), size=50, replace=False)
X_sample = X_test[indices]
y_sample = y_test[indices]

# Predict
y_pred = knn.predict(X_sample)

# Accuracy
accuracy = accuracy_score(y_sample, y_pred)
print(f"Accuracy on 50 real test samples: {accuracy:.2f}")