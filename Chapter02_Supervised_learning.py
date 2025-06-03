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



# for Decision Boundaries
# NOTE: Decision boundary visualization only works on 2D data.
# So we reduce to 2 features for visualization purposes.

X_vis = cancer.data[:, :2]  # Use first two features
y_vis = cancer.target

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for n_neighbors, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_vis, y_vis)
    mglearn.plots.plot_2d_separator(clf, X_vis, fill=True, eps=0.5, ax=ax, alpha=0.4)
    mglearn.discrete_scatter(X_vis[:, 0], X_vis[:, 1], y_vis, ax=ax)
    ax.set_title(f"{n_neighbors} neighbor(s)")
    ax.set_xlabel("Feature 0 (mean radius)")
    ax.set_ylabel("Feature 1 (mean texture)")

axes[0].legend(loc=3)
plt.tight_layout()
plt.show()