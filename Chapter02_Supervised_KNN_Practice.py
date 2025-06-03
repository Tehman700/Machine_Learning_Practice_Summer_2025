import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

cancer = load_breast_cancer()

df  = pd.DataFrame(cancer.data, columns= cancer.feature_names)
print(df.head())

print(df.describe())


# Put another column in last that is basically the Y-State
df['target'] = cancer.target

print(df.describe())

# Counting how many sets are in benign and malignant points

benign = 0
malignant = 0

for i in cancer.target:
    if i == 0:
        benign += 1
    elif i ==1:
        malignant += 1

print(benign," " ,malignant)

# Now the Training and Testing Part of the Model

X_train, X_test,  Y_train, Y_test = train_test_split(cancer.data, cancer.target, random_state=0)


# What if we changed the neighbors each time and view our results
neighbors_settings = range(1, 11)

scoring_accuracy = []
y = []

regression_test_accuracy = []
regression_train_accuracy = []
for i in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)

    scoring_accuracy.append(knn.score(X_test, Y_test))
    y.append(knn.score(X_train, Y_train))




print(scoring_accuracy)
print(y)

print(regression_test_accuracy)
print(regression_train_accuracy)



# Can change this below according to the plot which we want
# plt.plot(neighbors_settings, regression_test_accuracy, label = "Testing Values")
# plt.plot(neighbors_settings, regression_train_accuracy, label = "Training Values")
# plt.ylabel("Accuracy")
# plt.xlabel("Number of Neighbors")
# plt.legend()
# plt.show()







for i in range(1,20):
    knn_regression = KNeighborsRegressor(n_neighbors=i)
    knn_regression.fit(X_train, Y_train)


    #print("Test set predictions in RegressionModel:\n{}".format(knn_regression.predict(X_test)))
    # Since it's a regression model we should be using R^2 Score Strategy


    print("TEST SET R^2: {:.2f}".format(knn_regression.score(X_test, Y_test)))