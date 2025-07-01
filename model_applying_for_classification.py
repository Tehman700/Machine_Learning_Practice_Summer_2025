from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from full_scale_classification_problem import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

start = time.time()
# Below was basically a very basic splitting of features for training and testing split
X = user_stats[[
    'total_views', 'total_adds', 'total_purchases',
    'unique_items', 'unique_categories',
    'avg_seconds_per_event', 'view_to_cart', 'cart_to_purchase'
]]

Y = user_stats['abnormal']

# Replace infinities and Nans withs zeros

print("Now we are replacing for infinite's and Nans with Zeros")
X = X.replace([float('inf'), -float('inf')], 0)
X = X.fillna(0)

# The main part of training and testing the splits

print("Now we are training and testing the splits")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# First one is KNN
print("We are doing the model for KNN and being trained on KNN")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
print("Accuracy for KNN ", accuracy_score(Y_test, y_pred))
print("Precision Score for KNN ", precision_score(Y_test, y_pred))
print("Recall Score for KNN ", recall_score(Y_test, y_pred))
print("F1_score for KNN ", f1_score(Y_test, y_pred))



# Second one is for Logistic Regression

print("We are doing the model for SVM and being trained on SVM")
logistic = LogisticRegression()
logistic.fit(X_train, Y_train)
y_pred_lr = logistic.predict(X_test)
print("Accuracy for Logistic Regression ", accuracy_score(Y_test, y_pred_lr))
print("Precision Score for Logistic Regression ", precision_score(Y_test, y_pred_lr))
print("Recall Score for Logistic Regression ", recall_score(Y_test, y_pred_lr))
print("F1_score for Logistic Regression ", f1_score(Y_test, y_pred_lr))



# Third one is for Support Vector Machine Algorithm

print("We are doing the model for SVM and being trained on SVM")
svm = SVC()
svm.fit(X_train, Y_train)
svm_lr = svm.predict(X_test)
print("Accuracy for SVM ", accuracy_score(Y_test, svm_lr))
print("Precision Score for SVM ", precision_score(Y_test, svm_lr))
print("Recall Score for SVM ", recall_score(Y_test, svm_lr))
print("F1_score for SVM ", f1_score(Y_test, svm_lr))



# Fourth one is for Naive Bayes Algorithm

print("We are doing the model for Naive bayes and being trained on Naive Bayes")
nb = GaussianNB()
nb.fit(X_train, Y_train)
y_pred_nb_lr = nb.predict(X_test)
print("Accuracy for Naive Bayes ", accuracy_score(Y_test, y_pred_nb_lr))
print("Precision Score for Naive Bayes ", precision_score(Y_test, y_pred_nb_lr))
print("Recall Score for Naive Bayes ", recall_score(Y_test, y_pred_nb_lr))
print("F1_Score for Naive Bayes ", f1_score(Y_test, y_pred_nb_lr))

end_time = time.time()

print(f"\n Time taken: {end_time - start:.2f} seconds")


