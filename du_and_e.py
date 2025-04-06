import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

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
