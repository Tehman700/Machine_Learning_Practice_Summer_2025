import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions

df = pd.read_csv("D:/New Shop Prediction Model/placement.csv")

#print(df.head())
#print(df.shape)
#print(df.info())

df = df.iloc[:,1:]
#print(df.head())
#print(df.shape)



plt.scatter(df['cgpa'], df['iq'], c = df['placement'])

# Now that i have an idea that i can use logistic regression
#print(df.head())
X = df.iloc[:,0:2]
Y = df.iloc[:,-1]
#print(X.head())
#print(Y.head())

X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Just the training
clf = LogisticRegression()
clf.fit(X_train, Y_train)

print(clf.predict(X_test))
print(Y_test)

y_pred = clf.predict(X_test)
print(accuracy_score(Y_test, y_pred))

plot_decision_regions(X_train, Y_train.values,clf=clf,legend = 2)
plt.show()

# WE CAN USE PICKLE TO DEPLOY THE FILE OF ML MODEL