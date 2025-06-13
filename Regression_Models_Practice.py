from sklearn.datasets import  load_diabetes
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

diabetes = load_diabetes()

df  = pd.DataFrame(diabetes.data, columns= diabetes.feature_names)
print(df.head())

print(df.describe())
print(diabetes.feature_names)
print("\n\n\n\n\n")


# df['target'] = diabetes.target
print(df.head())

X_train, X_test,Y_train, Y_test = train_test_split(diabetes.data, diabetes.target,random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# STEP 4: Make predictions
y_pred = regressor.predict(X_test)
#
# # STEP 5: Evaluate the model
print("R² score:", r2_score(Y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(Y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(Y_test, y_pred))

print("Training set score: {:.2f}".format(regressor.score(X_train, Y_train)))
print("Test set score: {:.2f}".format(regressor.score(X_test, Y_test)))

# plt.figure(figsize=(8, 5))
# plt.scatter(Y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
# plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
# plt.xlabel("Actual Target Values")
# plt.ylabel("Predicted Target Values")
# plt.title("Actual vs Predicted Diabetes Progression")
# plt.grid(True)
# plt.show()



