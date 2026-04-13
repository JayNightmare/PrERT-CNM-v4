CREATE TABLE user_records_00005 (
  user_id BIGINT PRIMARY KEY,
  first_name VARCHAR(120),
  last_name VARCHAR(120),
  email VARCHAR(255),
  phone VARCHAR(32),
  postal_address TEXT,
  birth_date DATE,
  national_id VARCHAR(40),
  bank_account_number VARCHAR(40),
  income_band VARCHAR(40),
  medical_notes TEXT,
  biometric_template TEXT,
  ip_address VARCHAR(64),
  device_identifier VARCHAR(64)
);
