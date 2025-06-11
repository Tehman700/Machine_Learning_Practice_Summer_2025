import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("D:/Machine Learning Pycharm/tmdb_5000_movies.csv/tmdb_5000_movies.csv")


# Extract primary genre
def extract_primary_genre(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        if genres and isinstance(genres, list) and len(genres) > 0:
            return genres[0]['name']
    except:
        return None
    return None


# Apply extraction
df['primary_genre'] = df['genres'].apply(extract_primary_genre)
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year

# Select relevant columns
columns_needed = ['vote_average', 'runtime', 'budget', 'popularity', 'release_year', 'primary_genre']
df_model = df[columns_needed].dropna()

# Ensure numeric types
numeric_columns = ['vote_average', 'runtime', 'budget', 'popularity', 'release_year']
df_model[numeric_columns] = df_model[numeric_columns].apply(pd.to_numeric, errors='coerce')
df_model = df_model.dropna()

# Features and labels
X = df_model[numeric_columns]
Y = df_model['primary_genre']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)


# ACKER logic function
def expected_accuracy(test_point, k_values, X_train, y_train, l=3):
    distances = np.linalg.norm(X_train - test_point, axis=1)
    nearest_indices = np.argsort(distances)[:l]

    best_k = None
    best_acc = -1

    for k in k_values:
        correct = 0
        for idx in nearest_indices:
            x_holdout = X_train[idx]
            y_holdout = y_train.iloc[idx]

            # Exclude this one from training temporarily
            mask = np.arange(len(X_train)) != idx
            X_temp = X_train[mask]
            y_temp = y_train.iloc[mask]

            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_temp, y_temp)
            prediction = model.predict([x_holdout])[0]
            if prediction == y_holdout:
                correct += 1

        acc = correct / l
        if acc > best_acc:
            best_acc = acc
            best_k = k

    return best_k


# Adaptive prediction
k_range = list(range(1, 100))  # Try k from 1 to 9
predictions = []

for i in range(len(X_test)):
    test_point = X_test[i]
    best_k = expected_accuracy(test_point, k_range, X_train, y_train, l=3)
    print("This is the best K for NOW: ", best_k)
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    prediction = knn.predict([test_point])[0]
    predictions.append(prediction)

# Accuracy
acker_accuracy = accuracy_score(y_test, predictions)
print("ACKER Logic Accuracy:", acker_accuracy)
