from sklearn.datasets import load_iris
import pandas as pd

# Load dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.isnull().sum())

# Optionally, handle missing values if any
df = df.dropna()  # or use imputation if needed
duplicates = df.duplicated().sum()
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicates} duplicate rows.")
print(df.dtypes)

# Convert object columns if needed
# e.g., df['sepal length (cm)'] = df['sepal length (cm)'].astype(float)
# Basic sanity checks for value ranges
assert df['sepal length (cm)'].between(0, 10).all(), "Out-of-range sepal length"
assert df['sepal width (cm)'].between(0, 10).all(), "Out-of-range sepal width"
assert df['petal length (cm)'].between(0, 10).all(), "Out-of-range petal length"
assert df['petal width (cm)'].between(0, 10).all(), "Out-of-range petal width"
print(df['target'].unique())  # Should be [0, 1, 2]

assert set(df['target'].unique()) == {0, 1, 2}, "Unexpected class labels"
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
def validate_iris_data(df):
    assert df.isnull().sum().sum() == 0, "Missing values found"
    assert df.duplicated().sum() == 0, "Duplicates found"
    assert df['sepal length (cm)'].between(0, 10).all()
    assert df['sepal width (cm)'].between(0, 10).all()
    assert df['petal length (cm)'].between(0, 10).all()
    assert df['petal width (cm)'].between(0, 10).all()
    assert set(df['target'].unique()) == {0, 1, 2}
    print("✅ Iris dataset passed integrity and consistency checks.")
