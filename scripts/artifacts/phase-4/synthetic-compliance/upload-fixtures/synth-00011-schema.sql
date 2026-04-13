CREATE TABLE billing_profiles_00011 (
  profile_id BIGINT PRIMARY KEY,
  email VARCHAR(255),
  customer_name VARCHAR(200),
  billing_city VARCHAR(100),
  card_last4 VARCHAR(4),
  account_reference VARCHAR(40),
  last_login_ip VARCHAR(64),
  created_at TIMESTAMP
);
