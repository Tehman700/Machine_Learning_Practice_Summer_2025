from sklearn.datasets import  load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

diabetes = load_diabetes()

df  = pd.DataFrame(diabetes.data, columns= diabetes.feature_names)
print(df.head())

print(df.describe())