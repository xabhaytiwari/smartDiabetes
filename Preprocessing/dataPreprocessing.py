import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

"""Load the CSV file into a pandas DataFrame."""
def load_data(file_path):
    return pd.read_csv(file_path)

"""Handle Missing or Inconsistent Values."""
def remove_incosistencies(df):
    cols_with_invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']

    print(f"Missing data before cleaning: \n{df[cols_with_invalid_zeros].isnull().sum()}")
    df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].replace(0, np.nan)

    print("\nMissing Values after replacing 0s with Nan:")
    print(df[cols_with_invalid_zeros].isnull().sum())

    return df

"Visualise Missing Data."
def plot_missing_data(df, title="Missing Data Visualisation"):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', cbar_kws={'label': 'Missing Data'})
    plt.title(title)
    plt.show

"""Impute Missing Values using Linear Regression."""
def impute_missing_values(df):
    columns_with_missing = df.columns[df.isnull().any()].tolist()

    for col in columns_with_missing:
        print(f"Imputing missing values for column: {col}")

        non_missing_cols = df.columns[df.columns != col].tolist()

        train_df = df[non_missing_cols +[col]].dropna()

        if train_df.empty:
            print(f"Skipping {col} — no complete rows available for training.")
            continue

        """Split the known(non_missing_cols)"""
        X_train = train_df[non_missing_cols]
        y_train = train_df[col]
        
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Get rows where col is null and predictor columns are not null
        predict_df = df[df[col].isnull()]
        predict_df = predict_df.dropna(subset=non_missing_cols)

        if predict_df.empty:
            print(f"Skipping prediction for {col} — no usable rows with complete predictors.")
            continue
        
        X_pred = predict_df[non_missing_cols]
        df.loc[X_pred.index, col] = model.predict(X_pred)

    return df

"""Visualise the Comparison before and after Imputation"""
def plot_data_comparison(before_df, after_df):
    missing_before = before_df.isnull().sum()
    missing_after = after_df.isnull().sum()

    comparison = pd.DataFrame({'Before Imputation' : missing_before, 'After Imputation' : missing_after})

    comparison = comparison[comparison['Before Imputation'] > 0]

    if comparison.empty:
        print("No columns had missing values before imputation. Skipping comparison plot.")
        return
    
    comparison.plot(kind='bar', figsize=(12, 6))
    plt.title("Comparison of Missing Values Before and After Imputation")
    plt.ylabel("Number of Missing Values")
    plt.show()

"""Visualise correlations between features"""
def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap")
    plt.show()

def main(file_path):
    #Load Data
    df = load_data(file_path)

    #Plot missing data before cleaning
    plot_missing_data(df, "Missing data before cleaning")

    #Store the original dataframe to compare it later
    original_df = df.copy()

    df = remove_incosistencies(df)

    #Plot Missing Data after cleaning
    plot_missing_data(df, "Missing data after Cleaning")

    df = impute_missing_values(df)

    #Plot Missing Data after imputation
    plot_missing_data(df, "Missing data after Imputation")

    #Compare missing data before and after Imputation
    plot_data_comparison(original_df, df)

    plot_correlation_heatmap(df)

    print(f"Data after processing: \n{df.head()}")
    return df




