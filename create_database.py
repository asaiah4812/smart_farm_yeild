# create_database.py
from app import app, db
import sqlite3
import os

def create_database_tables():
    # Get the database path from the app configuration
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.db')
    print(f"Creating database at: {db_path}")
    
    # Create a connection to SQLite database
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(80) UNIQUE NOT NULL,
            email VARCHAR(120) UNIQUE NOT NULL,
            password_hash VARCHAR(128) NOT NULL,
            role VARCHAR(20) DEFAULT 'farmer',
            first_name VARCHAR(100),
            last_name VARCHAR(100),
            phone VARCHAR(20),
            address TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # Create farms table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS farms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name VARCHAR(100) NOT NULL,
            location VARCHAR(200),
            area_hectares FLOAT,
            soil_type VARCHAR(50),
            soil_ph FLOAT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create historical_yield_data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS historical_yield_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farm_id INTEGER NOT NULL,
            crop_type VARCHAR(50) NOT NULL,
            yield_amount FLOAT NOT NULL,
            harvest_date DATE NOT NULL,
            temperature_avg FLOAT,
            rainfall_total FLOAT,
            humidity_avg FLOAT,
            soil_ph FLOAT,
            fertilizer_used FLOAT,
            irrigation_used FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (farm_id) REFERENCES farms (id)
        )
    ''')
    
    # Create prediction_requests table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            farm_id INTEGER NOT NULL,
            crop_type VARCHAR(50) NOT NULL,
            temperature_avg FLOAT,
            rainfall_total FLOAT,
            humidity_avg FLOAT,
            soil_ph FLOAT,
            fertilizer_planned FLOAT,
            irrigation_planned FLOAT,
            predicted_yield FLOAT,
            confidence FLOAT,
            model_used VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (farm_id) REFERENCES farms (id)
        )
    ''')
    
    # Create system_logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action VARCHAR(100) NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    connection.commit()
    connection.close()
    print("All tables created successfully!")

if __name__ == "__main__":
    with app.app_context():
        create_database_tables()