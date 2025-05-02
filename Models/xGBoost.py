 # xgboost_baseline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns

def implement_xGBoost(df):
   
    # 1. Prepare X and y
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Train XGBoost Classifier
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', enable_categorical=True)
    model.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    print("\n--- Confusion Matrix ---")
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print("\n--- ROC AUC ---")
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.show()

    # 5. Feature Importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model.feature_importances_, y=X.columns)
    plt.title("XGBoost Feature Importances")
    plt.show()
