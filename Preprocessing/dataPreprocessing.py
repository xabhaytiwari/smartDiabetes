import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class DiabetesDataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.scalers = {}
        self.feature_importance = None
        self.pca = None
        
    def load_data(self):
        """Load data with type validation"""
        df = pd.read_csv(self.file_path)
        
        # Validate data ranges
        valid_ranges = {
            'Pregnancies': (0, 20),
            'Glucose': (0, 300),
            'BloodPressure': (0, 150),
            'SkinThickness': (0, 100),
            'Insulin': (0, 1000),
            'BMI': (0, 70),
            'DiabetesPedigreeFunction': (0, 3),
            'Age': (20, 100)
        }
        
        for col, (min_val, max_val) in valid_ranges.items():
            if col in df.columns:
                df[col] = df[col].clip(min_val, max_val)
                
        return df

    def handle_missing_values(self, df):
        """Advanced missing value treatment"""
        # Smart zero handling - only for biologically impossible zeros
        zero_to_nan_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        df[zero_to_nan_cols] = df[zero_to_nan_cols].replace(0, np.nan)
        
        # MICE imputation using XGBoost
        imputer = IterativeImputer(
            estimator=XGBRegressor(random_state=42, n_estimators=50),
            max_iter=20,
            random_state=42,
            skip_complete=True
        )
        
        imputed_data = imputer.fit_transform(df)
        df_imputed = pd.DataFrame(imputed_data, columns=df.columns)
        
        return df_imputed

    def detect_outliers(self, df):
        """Isolation Forest for outlier detection"""
        iso = IsolationForest(contamination=0.05, random_state=42)
        outliers = iso.fit_predict(df.select_dtypes(include=[np.number]))
        df['IsOutlier'] = (outliers == -1).astype(int)
        return df

    def create_features(self, df):
        """Advanced feature engineering"""
        # Metabolic syndrome features
        df['Metabolic_Syndrome'] = (
            (df['Glucose'] > 100) & 
            (df['BMI'] > 30) & 
            (df['BloodPressure'] > 130)
        ).astype(int)
        
        # Interaction terms
        df['Glucose_Age_Risk'] = df['Glucose'] * (df['Age'] / 100)
        df['BP_BMI_Ratio'] = df['BloodPressure'] / df['BMI']
        
        # Physiological ratios
        df['Insulin_Glucose_Ratio'] = df['Insulin'] / (df['Glucose'] + 1)
        df['Skin_BMI_Ratio'] = df['SkinThickness'] / df['BMI']
        
        # Binning continuous variables
        df['Age_Group'] = pd.cut(df['Age'], 
                                bins=[20, 30, 40, 50, 60, 100],
                                labels=[0, 1, 2, 3, 4])
        
        df['BMI_Category'] = pd.cut(df['BMI'],
                                  bins=[0, 18.5, 25, 30, 100],
                                  labels=[0, 1, 2, 3])
        
        return df

    def analyze_features(self, df, target_col='Outcome'):
        """Feature importance analysis"""
        X = df.drop(columns=[target_col], errors='ignore')
        y = df[target_col]
        
        # Mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        self.feature_importance = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        self.feature_importance.plot(kind='bar')
        plt.title("Feature Importance (Mutual Information)")
        plt.ylabel("Importance Score")
        plt.tight_layout()
        plt.show()
        
        return df
    
    def scale_features(self, df):
        """Multiple scaling strategies"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Robust scaling for neural networks
        robust_scaler = RobustScaler()
        df_robust = robust_scaler.fit_transform(df[numerical_cols])
        self.scalers['robust'] = robust_scaler

        # Power transform for skewed features
        power = PowerTransformer(method='yeo-johnson')
        df_power = power.fit_transform(df[numerical_cols])
        self.scalers['power'] = power

        # MinMax scaling for tree models
        minmax = MinMaxScaler(feature_range=(0, 1))
        df_minmax = minmax.fit_transform(df[numerical_cols])
        self.scalers['minmax'] = minmax

        return df   


    def handle_imbalance(self, df, target_col='Outcome'):
        """SMOTENC for categorical features"""

        X = df.drop(columns=[target_col])
        y = df[target_col]

    # Get column indices of categorical features in X
        categorical_features = [
        X.columns.get_loc(col)
        for col in ['Age_Group', 'BMI_Category', 'Metabolic_Syndrome']
        if col in X.columns
            ]
    
        if len(categorical_features) > 0:
            smote = SMOTENC(
            categorical_features=categorical_features,
            random_state=42,
            k_neighbors=5
        )
        else:
            smote = SMOTE(random_state=42)
    
        X_res, y_res = smote.fit_resample(X, y)
        return pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=target_col)], axis=1)


    def reduce_dimensionality(self, df, target_col='Outcome'):
        """PCA for visualization and feature reduction"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
            
        self.pca = PCA(n_components=2, random_state=42)
        pca_results = self.pca.fit_transform(df[numerical_cols])
        
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_results[:, 0], pca_results[:, 1], c=df[target_col], alpha=0.6)
        plt.title("PCA Visualization of Diabetes Data")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(label='Diabetes Outcome')
        plt.show()
        
        return df

    def preprocess(self):
        """Complete preprocessing pipeline"""
        # 1. Load and validate data
        df = self.load_data()
        
        # 2. Handle missing values
        df = self.handle_missing_values(df)
        
        # 3. Outlier detection
        df = self.detect_outliers(df)
        
        # 4. Feature engineering
        df = self.create_features(df)
        
        # 5. Feature analysis
        df = self.analyze_features(df)
        
        # 6. Handle class imbalance
        df = self.handle_imbalance(df)
        
        # 7. Feature scaling
        df = self.scale_features(df)
        
        # 8. Dimensionality reduction (optional)
        df = self.reduce_dimensionality(df)
        
        return df, self.scalers, self.feature_importance

# Usage example:
def main(file_path):
    preprocessor = DiabetesDataPreprocessor(file_path)
    processed_data, scalers, feature_importance = preprocessor.preprocess()
    return processed_data

if __name__ == "__main__":
    df = main('../Data/diabetesData.csv')
