import pandas as pd
import ast  # For parsing genres list
import warnings
warnings.filterwarnings("ignore")

# Load CSV
df = pd.read_csv("D:/Machine Learning Pycharm/tmdb_5000_movies.csv/tmdb_5000_movies.csv")

# Extract primary genre
def extract_primary_genre(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        if genres and isinstance(genres, list):
            return genres[0]['name']
    except Exception as e:
        return None
    return None

# Apply genre extraction
df['primary_genre'] = df['genres'].apply(extract_primary_genre)

# Convert release_date to datetime
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year

# Select relevant columns
columns_needed = ['vote_average', 'runtime', 'budget', 'popularity', 'release_year', 'primary_genre']
df_model = df[columns_needed]

# Convert to numeric if necessary
numeric_columns = ['vote_average', 'runtime', 'budget', 'popularity', 'release_year']

for col in numeric_columns:
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

# Drop rows with missing values
df_model = df_model.dropna()

# Show final cleaned dataset
print(df_model.head())
#print(df_model.describe())




from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler


X= df_model[numeric_columns]
Y = df_model['primary_genre']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

f = []
for i in range(1,100):
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance')

    knn.fit(X_train, y_train)

    f.append(knn.score(X_test, y_test))
    print(knn.score(X_test, y_test))

fi = sum(f) / len(f)
print("This is the Average of the Accuracy ", fi)








