# Import Libraries

# import pandas as pd
import numpy as np
# import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
# from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler

# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
# from utils.utils_baseline import *
# from utils.utils_preprocess import *

# A function to extract data labels and detect data types, then return X and y
def preprocess_data(df, label):
    X, y = df.drop(label, axis=1), df[[label]]

    class_labels =  list(np.unique(y))
    class_mappings = list(range(len(class_labels)))
    custom_mapping = dict(zip(class_labels, class_mappings))
    print(custom_mapping)
    # Encode y to numeric
    encoder = OrdinalEncoder(categories=[list(custom_mapping.keys())], dtype=int)
    y_encoded = encoder.fit_transform(y)
    # Extract text features
    # We are detecting types that are categorical to provide correct perturbation also in the feature
    cats = X.select_dtypes(exclude=np.number).columns.tolist()

    # Convert to pd.Categorical
    for col in cats:
        X[col] = X[col].astype('category')

    return X, y, y_encoded, class_labels

# A function to return correlations in the dataset
def get_correlations_in_data(df):
    # make sure the types are all of numeric and not string, encode them
    # Detect and encode categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    correlation_matrix = df.corr()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix Heatmap')
    plt.show()

    return correlation_matrix
