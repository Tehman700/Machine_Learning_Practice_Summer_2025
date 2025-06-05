from sklearn.datasets import  fetch_california_housing
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

diabetes = fetch_california_housing()

df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
print(df.head())
print(df.describe())

X_train, X_test,Y_train, Y_test = train_test_split(df.drop('Med'), diabetes.target,random_state=42)


ridge = Ridge()
ridge.fit(X_train, Y_train)

print("Training set score: {:.2f}".format(ridge.score(X_train, Y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, Y_test)))

lasso = Lasso(alpha=0.01)
lasso.fit(X_train, Y_train)

print("Training set score: {:.2f}".format(lasso.score(X_train, Y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, Y_test)))


regressor = LinearRegression()
regressor.fit(X_train, Y_train)
print("Training set score: {:.2f}".format(regressor.score(X_train, Y_train)))
print("Test set score: {:.2f}".format(regressor.score(X_test, Y_test)))



