from sklearn.datasets import  load_diabetes
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

diabetes = load_diabetes()

df  = pd.DataFrame(diabetes.data, columns= diabetes.feature_names)
print(df.head())

print(df.describe())
print(diabetes.feature_names)
print("\n\n\n\n\n")


df['target'] = diabetes.target
print(df.head())

X_train, X_test,Y_train, Y_test = train_test_split(diabetes.data, diabetes.target,random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# STEP 4: Make predictions
y_pred = regressor.predict(X_test)

# STEP 5: Evaluate the model
print("R² score:", r2_score(Y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(Y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(Y_test, y_pred))



