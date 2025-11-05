import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_sample_data():
    """Load sample data for demonstration"""
    # This would typically load from a CSV or database
    # For now, we'll create some sample data
    data = {
        'temperature_avg': [18.5, 22.3, 20.1, 17.8, 19.2, 21.5, 16.8, 23.1],
        'rainfall_total': [450, 380, 420, 480, 390, 410, 500, 370],
        'humidity_avg': [65, 70, 68, 62, 67, 69, 63, 71],
        'soil_ph': [6.8, 7.2, 6.5, 7.0, 6.9, 7.1, 6.7, 7.3],
        'fertilizer_used': [120, 150, 90, 130, 110, 140, 100, 160],
        'irrigation_mm': [300, 450, 380, 320, 400, 430, 350, 470],
        'area_hectares': [50, 75, 60, 55, 65, 70, 45, 80],
        'yield_amount': [3.2, 5.8, 2.9, 3.5, 4.1, 5.2, 3.0, 6.1]
    }
    
    return pd.DataFrame(data)

def preprocess_data(df, target_column='yield_amount'):
    """Preprocess the data for training"""
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, X.columns.tolist()

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return rmse, mae, r2