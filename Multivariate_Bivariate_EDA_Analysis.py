import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


tips = sns.load_dataset("tips")
titanic = pd.read_csv('Titanic-Dataset.csv')
flights = sns.load_dataset('flights')
iris = sns.load_dataset('iris')

# WE IMPORTED ALL THE DATASETS NOW GOING TO APPLY MULTIVARIATE AND BIVARIATE EDA ON THIS
# TO PROOF THE POINTS ON DIFFERENT DATASET

print(tips.head())
print(flights.head())
print(iris.head())
print(titanic.head())


# Scatterplot (Numerical-Numerical Relation)
sns.scatterplot(x = tips['total_bill'], y = tips['tip'], hue = tips['sex'], style = tips['smoker'])
plt.show()



# BarPlot (when one is categorical and other is numerical)

sns.barplot(x = titanic['Pclass'], y = titanic['Survived'],hue = titanic['Sex'])
plt.show()


# BoxPlot (When one is numerical and other is categorical)
sns.boxplot(x = titanic['Sex'], y = titanic['Age'], hue=titanic['Survived'])
plt.show()