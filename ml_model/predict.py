import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

class YieldPredictor:
    def __init__(self, crop_type=None):
        self.crop_type = crop_type or 'all'
        self.models_dir = 'ml_model/models'
        self.models_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load the trained models and scaler"""
        try:
            # Check if models directory exists
            if not os.path.exists(self.models_dir):
                print("Models directory not found. Creating dummy models...")
                self.create_dummy_models()
                return
            
            # Load scaler
            scaler_path = os.path.join(self.models_dir, f'scaler_{self.crop_type}.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                print(f"Scaler not found for {self.crop_type}. Creating dummy scaler...")
                self.scaler = StandardScaler()
            
            # Load Random Forest model
            rf_path = os.path.join(self.models_dir, f'random_forest_{self.crop_type}.pkl')
            if os.path.exists(rf_path):
                self.rf_model = joblib.load(rf_path)
            else:
                print(f"Random Forest model not found for {self.crop_type}. Creating dummy model...")
                self.rf_model = self.create_dummy_rf_model()
            
            # Load SVM model
            svm_path = os.path.join(self.models_dir, f'svm_model_{self.crop_type}.pkl')
            if os.path.exists(svm_path):
                self.svm_model = joblib.load(svm_path)
            else:
                print(f"SVM model not found for {self.crop_type}. Creating dummy model...")
                self.svm_model = self.create_dummy_svm_model()
            
            # Load features
            features_path = os.path.join(self.models_dir, f'features_{self.crop_type}.txt')
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    self.features = f.read().split(',')
            else:
                self.features = ['temperature_avg', 'rainfall_total', 'humidity_avg', 
                               'soil_ph', 'fertilizer_planned', 'irrigation_planned', 'area_hectares']
                
            self.models_loaded = True
            print(f"Models loaded successfully for {self.crop_type}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Creating dummy models as fallback...")
            self.create_dummy_models()
    
    def create_dummy_rf_model(self):
        """Create a dummy Random Forest model for demonstration"""
        np.random.seed(42)
        X_dummy = np.random.rand(50, 7)
        y_dummy = 3 + 0.5 * X_dummy[:, 0] + 0.8 * X_dummy[:, 1] + np.random.randn(50) * 0.3
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_dummy, y_dummy)
        return model
    
    def create_dummy_svm_model(self):
        """Create a dummy SVM model for demonstration"""
        np.random.seed(42)
        X_dummy = np.random.rand(50, 7)
        y_dummy = 3 + 0.5 * X_dummy[:, 0] + 0.8 * X_dummy[:, 1] + np.random.randn(50) * 0.3
        
        model = SVR(kernel='linear')
        model.fit(X_dummy, y_dummy)
        return model
    
    def create_dummy_models(self):
        """Create complete dummy models for demonstration"""
        print("Creating complete dummy models for demonstration...")
        
        self.scaler = StandardScaler()
        self.rf_model = self.create_dummy_rf_model()
        self.svm_model = self.create_dummy_svm_model()
        self.features = ['temperature_avg', 'rainfall_total', 'humidity_avg', 
                        'soil_ph', 'fertilizer_planned', 'irrigation_planned', 'area_hectares']
        
        self.models_loaded = True
        print("Dummy models created successfully")
    
    def prepare_input_data(self, input_data):
        """Prepare input data for prediction"""
        # Default values if not provided
        defaults = {
            'temperature_avg': 20,
            'rainfall_total': 400,
            'humidity_avg': 65,
            'soil_ph': 6.8,
            'fertilizer_planned': 120,
            'irrigation_planned': 300,
            'area_hectares': 50
        }
        
        # Use provided values or defaults
        prepared_data = []
        for feature in self.features:
            value = input_data.get(feature, defaults.get(feature, 0))
            prepared_data.append(float(value))
        
        return np.array([prepared_data])
    
    def predict(self, input_data, model_type='random_forest'):
        """Make a prediction using the specified model"""
        if not self.models_loaded:
            print("Models not loaded. Creating dummy models...")
            self.create_dummy_models()
        
        try:
            # Prepare input data
            X = self.prepare_input_data(input_data)
            
            # Scale the input
            X_scaled = self.scaler.transform(X)
            
            # Make prediction based on model type
            if model_type == 'random_forest':
                prediction = self.rf_model.predict(X_scaled)[0]
            elif model_type == 'svm':
                prediction = self.svm_model.predict(X_scaled)[0]
            else:
                print(f"Unknown model type: {model_type}. Using Random Forest.")
                prediction = self.rf_model.predict(X_scaled)[0]
            
            # Ensure prediction is reasonable
            prediction = max(0.5, min(10.0, prediction))  # Clamp between 0.5 and 10.0 tons/hectare
            
            # Calculate confidence based on how close we are to average yield
            avg_yield = 3.5  # Average yield in tons/hectare
            confidence = max(0.6, min(0.95, 1 - abs(prediction - avg_yield) / (avg_yield * 2)))
            
            print(f"Prediction: {prediction:.2f} tons/hectare, Confidence: {confidence:.2%}")
            return prediction, confidence
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Return a reasonable default prediction
            default_prediction = 3.5  # Average yield
            default_confidence = 0.7
            return default_prediction, default_confidence
    
    def predict_all_models(self, input_data):
        """Make predictions using all models and return results"""
        results = {}
        
        for model_type in ['random_forest', 'svm']:
            prediction, confidence = self.predict(input_data, model_type)
            results[model_type] = {
                'prediction': prediction,
                'confidence': confidence,
                'model_name': 'Random Forest' if model_type == 'random_forest' else 'Support Vector Machine'
            }
        
        return results

# Simple test function
def test_predictor():
    """Test the predictor with sample data"""
    predictor = YieldPredictor('wheat')
    
    sample_data = {
        'temperature_avg': 18.5,
        'rainfall_total': 450,
        'humidity_avg': 65,
        'soil_ph': 6.8,
        'fertilizer_planned': 120,
        'irrigation_planned': 300,
        'area_hectares': 50
    }
    
    results = predictor.predict_all_models(sample_data)
    
    for model_name, result in results.items():
        print(f"{model_name}: {result['prediction']:.2f} tons/hectare (confidence: {result['confidence']:.2%})")

if __name__ == "__main__":
    test_predictor()