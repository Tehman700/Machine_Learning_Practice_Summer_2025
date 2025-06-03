from sklearn.datasets import load_breast_cancer
import pandas as pd


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