-- Create database
CREATE DATABASE IF NOT EXISTS smart_agriculture_db;
USE smart_agriculture_db;

-- Users table
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(128) NOT NULL,
    role ENUM('admin', 'agronomist', 'farmer') DEFAULT 'farmer',
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    phone VARCHAR(20),
    address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Farms table
CREATE TABLE farms (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    name VARCHAR(200) NOT NULL,
    location VARCHAR(200),
    area_hectares FLOAT,
    soil_type ENUM('sandy', 'clay', 'silt', 'loam', 'peat', 'chalky'),
    soil_ph FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Historical yield data table
CREATE TABLE historical_yield_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    farm_id INT NOT NULL,
    crop_type VARCHAR(50) NOT NULL,
    planting_date DATE,
    harvest_date DATE,
    yield_amount FLOAT NOT NULL, -- in tons per hectare
    temperature_avg FLOAT,
    rainfall_total FLOAT,
    humidity_avg FLOAT,
    fertilizer_used FLOAT, -- kg per hectare
    irrigation_mm FLOAT,
    notes TEXT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (farm_id) REFERENCES farms(id) ON DELETE CASCADE
);

-- Prediction requests table
CREATE TABLE prediction_requests (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    farm_id INT NOT NULL,
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
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (farm_id) REFERENCES farms(id) ON DELETE CASCADE
);

-- System logs table
CREATE TABLE system_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    action VARCHAR(200) NOT NULL,
    details TEXT,
    ip_address VARCHAR(45),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Model performance table
CREATE TABLE model_performance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    crop_type VARCHAR(50) NOT NULL,
    rmse FLOAT,
    mae FLOAT,
    r2_score FLOAT,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default admin user
INSERT INTO users (username, email, password_hash, role, first_name, last_name, is_active)
VALUES ('admin', 'admin@agriculture.com', 'pbkdf2:sha256:260000$N4ap6x2Q$3d8e58d6c7c32d3c6b7c8e7c5e8c7a5d3e4f6a7b8c9d0e1f2a3b4c5d6e7f8a9b', 'admin', 'System', 'Administrator', TRUE);

-- Add soil composition fields to farms table
ALTER TABLE farms ADD COLUMN soil_nitrogen FLOAT DEFAULT 0.0;
ALTER TABLE farms ADD COLUMN soil_phosphorus FLOAT DEFAULT 0.0;
ALTER TABLE farms ADD COLUMN soil_potassium FLOAT DEFAULT 0.0;
ALTER TABLE farms ADD COLUMN soil_organic_matter FLOAT DEFAULT 0.0;

-- Add model performance tracking
CREATE TABLE IF NOT EXISTS model_performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name VARCHAR(100) NOT NULL,
    crop_type VARCHAR(50) NOT NULL,
    rmse FLOAT,
    mae FLOAT,
    r2_score FLOAT,
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    confusion_matrix TEXT,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);