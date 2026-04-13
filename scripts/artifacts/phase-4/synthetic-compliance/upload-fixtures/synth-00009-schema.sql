CREATE TABLE accounts_00009 (
  account_id BIGINT PRIMARY KEY,
  email VARCHAR(255),
  phone VARCHAR(30),
  address_line1 TEXT,
  country VARCHAR(80),
  loyalty_tier VARCHAR(40),
  last_login_ip VARCHAR(64),
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);
