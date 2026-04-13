CREATE TABLE customer_profiles_00002 (
  profile_id BIGINT PRIMARY KEY,
  full_name VARCHAR(255),
  email VARCHAR(255),
  phone VARCHAR(30),
  home_address TEXT,
  city VARCHAR(100),
  country VARCHAR(80),
  dob DATE,
  ssn VARCHAR(20),
  passport_number VARCHAR(30),
  credit_card_number VARCHAR(40),
  account_number VARCHAR(40),
  salary_amount DECIMAL(12,2),
  health_diagnosis TEXT,
  biometric_hash TEXT,
  ip_address VARCHAR(64),
  device_id VARCHAR(80)
);
