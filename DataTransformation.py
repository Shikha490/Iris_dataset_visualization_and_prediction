from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#Standardization (Z-score Scaling)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_no_outliers[iris.feature_names])

#Min-Max Normalization
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df_no_outliers[iris.feature_names])

#Log or Square Root Transformation
df_log = df_no_outliers.copy()
for col in iris.feature_names:
    df_log[col] = np.log1p(df_log[col])  # log(1 + x) to handle 0s
