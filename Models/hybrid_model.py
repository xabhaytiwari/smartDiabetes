import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns


def feature_engineering(df):
    """Enhanced feature engineering with NaN handling"""
    # Interaction terms (safe for NaN values)
    df['Glucose_BMI'] = df['Glucose'] * df['BMI']
    df['Age_Insulin_Ratio'] = df['Age'] / (df['Insulin'].replace(0, np.nan))  # Handle zero insulin
    
    # Risk categories with proper NaN handling
    df['BMI_Class'] = pd.cut(df['BMI'], 
                           bins=[0, 18.5, 25, 30, 100], 
                           labels=[0, 1, 2, 3])
    df['BMI_Class'] = df['BMI_Class'].cat.add_categories(-1).fillna(-1).astype(int)
    
    df['Glucose_Class'] = pd.cut(df['Glucose'],
                               bins=[0, 90, 140, 200, 300],
                               labels=[0, 1, 2, 3])
    df['Glucose_Class'] = df['Glucose_Class'].cat.add_categories(-1).fillna(-1).astype(int)
    
    return df
def run_hybrid_model(df):
    """Optimized hybrid pipeline with feature engineering and class balancing"""
    # Create Models directory
    os.makedirs("Models", exist_ok=True)
    
    # Feature Engineering
    df = feature_engineering(df)
    
    # Split data
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Calculate class weights
    class_weights = {0: 1, 1: len(y_train[y_train==0]) / len(y_train[y_train==1])}
    print(f"\nClass weights: {class_weights}")

    # --- Optimized XGBoost ---
    print("\nTraining XGBoost with class weights...")
    xgb = XGBClassifier(
        scale_pos_weight=class_weights[1],
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        enable_categorical=True
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_pred):.2f}")

    # --- Enhanced DNN ---
    print("\nTraining Enhanced DNN...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    dnn = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    dnn.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    history = dnn.fit(
        X_train_scaled, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
        verbose=1
    )
    
    dnn_pred = (dnn.predict(X_test_scaled) > 0.5).astype(int)
    print(f"DNN Accuracy: {accuracy_score(y_test, dnn_pred):.2f}")

    # --- Advanced Stacking ---
    print("\nTraining Stacked Model with Disagreement Features...")
    xgb_probs = xgb.predict_proba(X_train)[:, 1]
    dnn_probs = dnn.predict(X_train_scaled).flatten()
    
    # Enhanced meta-features
    stacked_features = np.column_stack((
        xgb_probs,
        dnn_probs,
        xgb_probs - dnn_probs,  # Disagreement signal
        np.abs(xgb_probs - dnn_probs)  # Magnitude of disagreement
    ))
    
    # XGBoost as meta-learner
    meta_model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=42
    )
    meta_model.fit(stacked_features, y_train)

    # Evaluate
    test_xgb_probs = xgb.predict_proba(X_test)[:, 1]
    test_dnn_probs = dnn.predict(X_test_scaled).flatten()
    test_stacked = np.column_stack((
        test_xgb_probs,
        test_dnn_probs,
        test_xgb_probs - test_dnn_probs,
        np.abs(test_xgb_probs - test_dnn_probs)
    ))
    hybrid_pred = meta_model.predict(test_stacked)

    print("\n=== Optimized Hybrid Model Performance ===")
    print(f"Accuracy: {accuracy_score(y_test, hybrid_pred):.2f}")
    print(classification_report(y_test, hybrid_pred))

    y_proba = meta_model.predict_proba(test_stacked)[:, 1]
    
    # Idhar se

    print("\n--- Confusion Matrix ---")
    sns.heatmap(confusion_matrix(y_test, hybrid_pred), annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print("\n--- ROC AUC ---")
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.show()

    # Save models
    # try:
    #     joblib.dump(xgb, os.path.join("Models", "xgboost_model.pkl"))
    #     dnn.save(os.path.join("Models", "dnn_model.h5"))
    #     joblib.dump(meta_model, os.path.join("Models", "stacking_model.pkl"))
    #     print("\nAll models saved successfully in Models/ directory")
    # except Exception as e:
    #     print(f"\nError saving models: {str(e)}")

    return hybrid_pred