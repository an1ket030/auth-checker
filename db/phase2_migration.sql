-- Phase 2 Database Migration
-- Run against Supabase PostgreSQL
-- Date: 2026-02-26

-- ==============================
-- 2.1: Add fields to users table
-- ==============================
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_verified BOOLEAN DEFAULT FALSE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_url TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMP;

-- ==============================
-- 2.1: Email Verifications table
-- ==============================
CREATE TABLE IF NOT EXISTS email_verifications (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    otp_code VARCHAR(6) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_email_verifications_user_id ON email_verifications(user_id);

-- ==============================
-- 2.1: Password Resets table
-- ==============================
CREATE TABLE IF NOT EXISTS password_resets (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    reset_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    used BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_password_resets_token ON password_resets(reset_token);

-- ==============================
-- 2.2: User Sessions table  
-- ==============================
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    refresh_token VARCHAR(255) UNIQUE NOT NULL,
    device_info TEXT,
    ip_address VARCHAR(45),
    expires_at TIMESTAMP NOT NULL,
    revoked BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_user_sessions_refresh_token ON user_sessions(refresh_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);

-- ==============================
-- 2.3: Add fields to scan_history
-- ==============================
ALTER TABLE scan_history ADD COLUMN IF NOT EXISTS image_thumbnail_url TEXT;
ALTER TABLE scan_history ADD COLUMN IF NOT EXISTS ml_confidence FLOAT;
ALTER TABLE scan_history ADD COLUMN IF NOT EXISTS ml_model_version VARCHAR(20) DEFAULT 'v1.0';
ALTER TABLE scan_history ADD COLUMN IF NOT EXISTS result_breakdown_json TEXT;
ALTER TABLE scan_history ADD COLUMN IF NOT EXISTS blockchain_tx_hash VARCHAR(100);
ALTER TABLE scan_history ADD COLUMN IF NOT EXISTS nfc_verified BOOLEAN DEFAULT FALSE;
ALTER TABLE scan_history ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP;

-- ==============================
-- 2.5: Drug Information table
-- ==============================
CREATE TABLE IF NOT EXISTS drug_information (
    id SERIAL PRIMARY KEY,
    brand_name VARCHAR(200) NOT NULL,
    generic_name VARCHAR(200),
    composition TEXT,
    usage TEXT,
    side_effects TEXT,
    manufacturer VARCHAR(200),
    category VARCHAR(100)
);
CREATE INDEX IF NOT EXISTS idx_drug_information_brand ON drug_information(brand_name);

-- ==============================
-- 2.6: Counterfeit Reports table (expanded)
-- Drop and recreate if the old one exists with fewer columns
-- ==============================
-- First check: the old table only had (id, scan_id, description, reported_at)
-- We need to add user_id, pharmacy_name, pharmacy_location, geo_lat, geo_long, status
ALTER TABLE counterfeit_reports ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id);
ALTER TABLE counterfeit_reports ADD COLUMN IF NOT EXISTS pharmacy_name VARCHAR(200);
ALTER TABLE counterfeit_reports ADD COLUMN IF NOT EXISTS pharmacy_location TEXT;
ALTER TABLE counterfeit_reports ADD COLUMN IF NOT EXISTS geo_lat FLOAT;
ALTER TABLE counterfeit_reports ADD COLUMN IF NOT EXISTS geo_long FLOAT;
ALTER TABLE counterfeit_reports ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'pending';
