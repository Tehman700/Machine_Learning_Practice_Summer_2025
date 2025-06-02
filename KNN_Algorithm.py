import math

from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)


species = []

for i in iris.target:
    if i == 0:
        species.append('setosa')
    elif i == 1:
        species.append('versicolor')
    elif i == 2:
        species.append('virginica')

# Add to the DataFrame
df['species'] = species

print(df.head())


ee = [56,56]

for i in range(0,150):
    first = df.iloc[i]['sepal length (cm)']
    sec = df.iloc[i]['sepal width (cm)']
    third = df.iloc[i]['petal length (cm)']
    fourth = df.iloc[i]['petal width (cm)']

    finals = math.sqrt(pow(5.1-first,2)+pow(3.5-sec,2)+pow(1.4-third,2)+pow(0.2-fourth,2))
    ee.append(finals)



# I'm applying k =1
print(df.iloc[ee.index(min(ee))])


