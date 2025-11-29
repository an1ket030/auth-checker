-- Users table to track who is scanning
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- The "Golden Record" database of legitimate medicines provided by manufacturers
CREATE TABLE valid_medicines (
    id SERIAL PRIMARY KEY,
    brand_name VARCHAR(100) NOT NULL,
    manufacturer VARCHAR(100) NOT NULL,
    batch_number VARCHAR(50) UNIQUE NOT NULL,
    expiry_date DATE NOT NULL,
    mrp DECIMAL(10, 2),
    packaging_hash VARCHAR(255) -- Represents a hash of the valid packaging image features
);

-- History of all scans performed
CREATE TABLE scan_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    scanned_batch_number VARCHAR(50),
    authenticity_score FLOAT,
    status VARCHAR(20) CHECK (status IN ('AUTHENTIC', 'SUSPICIOUS', 'FAKE')),
    scan_image_url TEXT,
    geo_lat FLOAT,
    geo_long FLOAT,
    scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Reports of counterfeits
CREATE TABLE counterfeit_reports (
    id SERIAL PRIMARY KEY,
    scan_id INTEGER REFERENCES scan_history(id),
    description TEXT,
    reported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- SEED DATA FOR DEMO PURPOSES
INSERT INTO valid_medicines (brand_name, manufacturer, batch_number, expiry_date, mrp, packaging_hash)
VALUES 
('Paracip-500', 'Cipla', 'B12345', '2026-12-31', 20.00, 'hash_xyz_123'),
('Dolo-650', 'Micro Labs', 'DL9999', '2025-10-15', 30.00, 'hash_abc_789');