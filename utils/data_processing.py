import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess_data(df, target_column=None):
    """
    Preprocess agricultural data for machine learning
    """
    # Make a copy of the dataframe
    df_processed = df.copy()
    
    # Handle missing values
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    
    # Impute numeric columns with mean
    if len(numeric_columns) > 0:
        numeric_imputer = SimpleImputer(strategy='mean')
        df_processed[numeric_columns] = numeric_imputer.fit_transform(df_processed[numeric_columns])
    
    # Impute categorical columns with mode
    if len(categorical_columns) > 0:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_columns] = categorical_imputer.fit_transform(df_processed[categorical_columns])
    
    # Encode categorical variables if needed
    for col in categorical_columns:
        if col != target_column:
            df_processed = pd.get_dummies(df_processed, columns=[col], prefix=col)
    
    # Scale numeric features (excluding target if specified)
    if target_column and target_column in numeric_columns:
        feature_columns = [col for col in numeric_columns if col != target_column]
    else:
        feature_columns = numeric_columns
    
    if len(feature_columns) > 0:
        scaler = StandardScaler()
        df_processed[feature_columns] = scaler.fit_transform(df_processed[feature_columns])
    
    return df_processed, scaler

def feature_importance_analysis(model, feature_names):
    """
    Analyze feature importance for tree-based models
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        # Sort feature importance in descending order
        indices = np.argsort(importance)[::-1]
        
        # Return sorted feature names and their importance
        return [(feature_names[i], importance[i]) for i in indices]
    else:
        return None

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats