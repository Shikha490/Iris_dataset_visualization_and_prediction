from sklearn.datasets import load_iris
import pandas as pd
from scipy.stats import zscore

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
#identifying patterns and trends 
print(df.groupby('species').mean())
import seaborn as sns
sns.pairplot(df, hue='species', diag_kind='kde')
import matplotlib.pyplot as plt
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
for col in iris.feature_names:
    sns.boxplot(x='species', y=col, data=df)
    plt.title(col)
    plt.show()

#--------Identifying Anomolies (Outliers)--------
z_scores = df[iris.feature_names].apply(zscore)
outliers = (abs(z_scores) > 3).any(axis=1)
df[outliers]

#Visualize Anomolies
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='anomaly', palette={1:'blue', -1:'red'})
plt.title('Anomalies in Iris Dataset')
plt.show()
