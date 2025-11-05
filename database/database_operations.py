import sqlite3
from sqlite3 import Error
from config import Config
import pandas as pd
import os

class DatabaseOperations:
    def __init__(self):
        self.config = Config()
        
    def create_connection(self):
        """Create a database connection to SQLite database"""
        try:
            # Get database path from SQLALCHEMY_DATABASE_URI
            if hasattr(self.config, 'SQLALCHEMY_DATABASE_URI'):
                db_url = self.config.SQLALCHEMY_DATABASE_URI
                # Extract the database file path from SQLAlchemy URI
                # Format: sqlite:///path/to/database.db
                if db_url.startswith('sqlite:///'):
                    db_path = db_url.replace('sqlite:///', '')
                else:
                    db_path = 'app.db'
            else:
                db_path = 'app.db'
                
            # Ensure the directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
                
            connection = sqlite3.connect(db_path)
            # Enable foreign keys
            connection.execute("PRAGMA foreign_keys = ON")
            return connection
        except Error as e:
            print(f"Error connecting to SQLite: {e}")
            return None
    
    def execute_query(self, query, params=None):
        """Execute a SQL query"""
        connection = self.create_connection()
        if connection is None:
            return None
            
        try:
            cursor = connection.cursor()
            if params:
                # Convert params to tuple if it's not already
                if not isinstance(params, (tuple, list)):
                    params = (params,)
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            # For SELECT statements, fetch results
            if query.strip().lower().startswith('select'):
                result = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                return pd.DataFrame(result, columns=columns)
            else:
                connection.commit()
                return cursor.rowcount
        except Error as e:
            print(f"Error executing query: {e}")
            print(f"Query: {query}")
            print(f"Params: {params}")
            return None
        finally:
            if connection:
                connection.close()
    
    def get_historical_data(self, user_id=None, farm_id=None, crop_type=None):
        """Retrieve historical yield data with optional filters"""
        query = """
        SELECT hyd.*, f.name as farm_name, f.location, f.area_hectares, f.soil_type, f.soil_ph,
               u.first_name, u.last_name
        FROM historical_yield_data hyd
        JOIN farms f ON hyd.farm_id = f.id
        JOIN users u ON f.user_id = u.id
        WHERE 1=1
        """
        params = []
        
        if user_id:
            query += " AND u.id = ?"
            params.append(user_id)
            
        if farm_id:
            query += " AND f.id = ?"
            params.append(farm_id)
            
        if crop_type:
            query += " AND hyd.crop_type = ?"
            params.append(crop_type)
            
        query += " ORDER BY hyd.harvest_date DESC"
        
        return self.execute_query(query, params)
    
    def save_prediction(self, user_id, farm_id, crop_type, features, predicted_yield, confidence, model_used):
        """Save a prediction to the database"""
        query = """
        INSERT INTO prediction_requests 
        (user_id, farm_id, crop_type, temperature_avg, rainfall_total, humidity_avg, 
         soil_ph, fertilizer_planned, irrigation_planned, predicted_yield, confidence, model_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            user_id, farm_id, crop_type, 
            features.get('temperature_avg'), 
            features.get('rainfall_total'),
            features.get('humidity_avg'),
            features.get('soil_ph'),
            features.get('fertilizer_planned'),
            features.get('irrigation_planned'),
            predicted_yield,
            confidence,
            model_used
        )
        
        return self.execute_query(query, params)
    
    def get_user_predictions(self, user_id):
        """Get all predictions for a user"""
        query = """
        SELECT pr.*, f.name as farm_name, f.location
        FROM prediction_requests pr
        JOIN farms f ON pr.farm_id = f.id
        WHERE pr.user_id = ?
        ORDER BY pr.created_at DESC
        """
        
        return self.execute_query(query, (user_id,))