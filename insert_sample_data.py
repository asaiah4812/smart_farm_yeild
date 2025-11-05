# insert_sample_data.py
from app import app, db
import sqlite3
import os
from werkzeug.security import generate_password_hash

def insert_sample_data():
    # Connect to SQLite database
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.db')
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    
    # First, check if tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    if not cursor.fetchone():
        print("Error: Database tables don't exist. Please run create_database.py first!")
        return
    
    # Clear existing data (optional - remove if you want to keep existing data)
    cursor.execute("DELETE FROM prediction_requests")
    cursor.execute("DELETE FROM historical_yield_data")
    cursor.execute("DELETE FROM farms")
    cursor.execute("DELETE FROM system_logs")
    cursor.execute("DELETE FROM users")
    
    # Insert sample users (with hashed passwords)
    users_data = [
        ('admin', 'admin@agriculture.com', generate_password_hash('admin123'), 'admin', 'System', 'Administrator', '+1234567890', 'Admin Address', 1),
        ('farmer_john', 'john@example.com', generate_password_hash('password123'), 'farmer', 'John', 'Doe', '+1234567890', '123 Farm Road, Countryside', 1),
        ('farmer_mary', 'mary@example.com', generate_password_hash('password123'), 'farmer', 'Mary', 'Smith', '+0987654321', '456 Ranch Street, Farmville', 1),
        ('agro_expert', 'expert@example.com', generate_password_hash('password123'), 'agronomist', 'Robert', 'Green', '+1122334455', '789 Agriculture Avenue, Crop City', 1)
    ]
    
    for user in users_data:
        cursor.execute('''
            INSERT INTO users 
            (username, email, password_hash, role, first_name, last_name, phone, address, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', user)
    
    # Insert sample farms
    farms_data = [
        (2, 'Sunshine Farm', 'Countryside', 50.5, 'loam', 6.8),
        (3, 'Green Valley Ranch', 'Farmville', 75.2, 'clay', 7.2)
    ]
    
    for farm in farms_data:
        cursor.execute('''
            INSERT INTO farms 
            (user_id, name, location, area_hectares, soil_type, soil_ph)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', farm)
    
    # Insert sample historical yield data
    historical_data = [
        (1, 'wheat', 3.2, '2023-06-15', 22.5, 450.0, 65.0, 6.8, 120.0, 300.0),
        (1, 'corn', 4.1, '2023-07-20', 25.0, 500.0, 70.0, 6.8, 150.0, 350.0),
        (2, 'rice', 5.3, '2023-08-10', 28.0, 600.0, 75.0, 7.2, 180.0, 400.0),
        (2, 'soybean', 2.8, '2023-09-05', 24.0, 480.0, 68.0, 7.2, 130.0, 320.0)
    ]
    
    for data in historical_data:
        cursor.execute('''
            INSERT INTO historical_yield_data 
            (farm_id, crop_type, yield_amount, harvest_date, temperature_avg, rainfall_total, humidity_avg, soil_ph, fertilizer_used, irrigation_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', data)
    
    # Insert sample predictions
    predictions_data = [
        (2, 1, 'wheat', 22.0, 420.0, 62.0, 6.8, 115.0, 290.0, 3.1, 0.85, 'random_forest'),
        (3, 2, 'rice', 27.5, 580.0, 73.0, 7.2, 175.0, 380.0, 5.2, 0.78, 'random_forest')
    ]
    
    for prediction in predictions_data:
        cursor.execute('''
            INSERT INTO prediction_requests 
            (user_id, farm_id, crop_type, temperature_avg, rainfall_total, humidity_avg, soil_ph, fertilizer_planned, irrigation_planned, predicted_yield, confidence, model_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', prediction)
    
    connection.commit()
    connection.close()
    print("Sample data inserted successfully!")
    print("You can now login with:")
    print("Admin: username='admin', password='admin123'")
    print("Farmer John: username='farmer_john', password='password123'")
    print("Farmer Mary: username='farmer_mary', password='password123'")
    print("Agronomist: username='agro_expert', password='password123'")

if __name__ == "__main__":
    with app.app_context():
        insert_sample_data()