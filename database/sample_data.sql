-- Sample data for testing
USE smart_agriculture_db;

-- Sample farmers
INSERT INTO users (username, email, password_hash, role, first_name, last_name, phone, address)
VALUES 
('farmer_john', 'john@example.com', 'pbkdf2:sha256:260000$N4ap6x2Q$3d8e58d6c7c32d3c6b7c8e7c5e8c7a5d3e4f6a7b8c9d0e1f2a3b4c5d6e7f8a9b', 'farmer', 'John', 'Doe', '+1234567890', '123 Farm Road, Countryside'),
('farmer_mary', 'mary@example.com', 'pbkdf2:sha256:260000$N4ap6x2Q$3d8e58d6c7c32d3c6b7c8e7c5e8c7a5d3e4f6a7b8c9d0e1f2a3b4c5d6e7f8a9b', 'farmer', 'Mary', 'Smith', '+0987654321', '456 Ranch Street, Farmville');

-- Sample agronomist
INSERT INTO users (username, email, password_hash, role, first_name, last_name, phone, address)
VALUES 
('agro_expert', 'expert@example.com', 'pbkdf2:sha256:260000$N4ap6x2Q$3d8e58d6c7c32d3c6b7c8e7c5e8c7a5d3e4f6a7b8c9d0e1f2a3b4c5d6e7f8a9b', 'agronomist', 'Robert', 'Green', '+1122334455', '789 Agriculture Avenue, Crop City');

-- Sample farms
INSERT INTO farms (user_id, name, location, area_hectares, soil_type, soil_ph)
VALUES 
(2, 'Sunshine Farm', 'Countryside', 50.5, 'loam', 6.8),
(3, 'Green Valley Ranch', 'Farmville', 75.2, 'clay', 7.2);

-- Sample historical yield data
INSERT INTO historical_yield_data (farm_id, crop_type, planting_date, harvest_date, yield_amount, temperature_avg, rainfall_total, humidity_avg, fertilizer_used, irrigation_mm)
VALUES 
(1, 'wheat', '2022-10-15', '2023-06-20', 3.2, 18.5, 450.0, 65.0, 120.0, 300.0),
(1, 'corn', '2022-04-10', '2022-09-15', 5.8, 22.3, 380.0, 70.0, 150.0, 450.0),
(2, 'soybean', '2022-05-05', '2022-10-10', 2.9, 20.1, 420.0, 68.0, 90.0, 380.0),
(2, 'wheat', '2022-10-20', '2023-06-25', 3.5, 17.8, 480.0, 62.0, 130.0, 320.0);