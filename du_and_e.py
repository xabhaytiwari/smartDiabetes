import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression

def regression_impute(df, targetCol, featureCols):
    train_data = df[df[targetCol] != 0]
    test_data = df[df[targetCol] == 0]

    if train_data.empty or test_data.empty:
        return df
    
    model = LinearRegression()
    model.fit(train_data[featureCols], train_data[targetCol])
    predicted = model.predict(test_data[featureCols])
    df.loc[df[targetCol] == 0, targetCol] = predicted
    return df

def explore_dataset(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    df = pd.read_csv(path)
    print("\n--- Dataset Shape ---")
    print(df.shape)

    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- Target Variable Distribution ---")
    if 'Outcome' in df.columns:
        sns.countplot(x='Outcome', data=df)
        plt.title("Target Variable Distribution")
        plt.show()
    else:
        print("No 'Outcome' column found.")

    print("\n--- Descriptive Statistics ---")
    print(df.describe())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Non-Possible Values Check ---")
    invalid_check_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in invalid_check_columns:
        if col in df.columns:
            invalid_count = (df[col] == 0).sum()
            print(f"{col}: {invalid_count} non-possible (zero) values")

    print("\n--- Flagging and Imputing Non-Possible Values with Regression ---")
    for col in invalid_check_columns:
        if col in df.columns:
            df[col + '_was_zero'] = (df[col] == 0).astype(int)
            feature_columns = [c for c in df.columns if c != col and c != 'Outcome' and c not in invalid_check_columns and df[c].dtype != 'object']
            df = regression_impute(df, col, feature_columns)

    print("\n--- Correlation Matrix ---")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlations")
    plt.show()

    print("\n--- Feature Distributions ---")
    df.hist(bins=30, figsize=(15, 10), color='steelblue')
    plt.tight_layout()
    plt.show()

    print("\nData Exploration Complete.")
