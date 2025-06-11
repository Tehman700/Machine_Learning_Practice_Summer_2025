import math
from collections import Counter

# Movie dataset: (name, imdb_rating, length, genre)
movies = [
    ("M1", 7.5, 130, "Action"),
    ("M2", 6.0, 90,  "Comedy"),
    ("M3", 8.2, 120, "Drama"),
    ("M4", 7.0, 95,  "Comedy"),
    ("M5", 6.5, 140, "Action"),
    ("M6", 8.0, 110, "Drama"),
    ("M7", 5.5, 85,  "Comedy"),
    ("M8", 6.2, 100, "Comedy")
]

# New point to classify
P = (7.3, 115)

# Euclidean distance between two movies
def distance(m1, m2):
    return math.sqrt((m1[1] - m2[1])**2 + (m1[2] - m2[2])**2)

# Step 1: Find 3 most similar movies to P
movies_with_dist = [(name, rating, length, genre, distance((name, rating, length), ("P", *P))) for (name, rating, length, genre) in movies]
movies_sorted = sorted(movies_with_dist, key=lambda x: x[4])
reference_movies = movies_sorted[:3]

print("📌 Reference Movies (Most similar to P):")
for m in reference_movies:
    print(f"{m[0]} - Genre: {m[3]}, Distance: {m[4]:.2f}")

# Step 2: Simulate accuracy for different k values
def simulate_expected_accuracy(k, ref_movies, full_dataset):
    correct = 0
    for m in ref_movies:
        test_name = m[0]
        test_point = (m[1], m[2])
        test_true_genre = m[3]

        # Leave-one-out: train on rest
        train = [x for x in full_dataset if x[0] != test_name]
        # Compute distances
        dists = [(x[3], distance((x[0], x[1], x[2]), ("", *test_point))) for x in train]
        dists.sort(key=lambda x: x[1])
        top_k = [genre for genre, dist in dists[:k]]

        # Majority vote
        counts = Counter(top_k)
        predicted_genre = counts.most_common(1)[0][0]

        if predicted_genre == test_true_genre:
            correct += 1

    return correct / len(ref_movies)

# Evaluate for k in {1, 3, 5}
print("\n🔁 Evaluating Expected Accuracy:")
for k in [1, 3, 5]:
    acc = simulate_expected_accuracy(k, reference_movies, movies)
    print(f"k = {k}: Expected Accuracy = {acc:.2f}")








import pandas as pd
