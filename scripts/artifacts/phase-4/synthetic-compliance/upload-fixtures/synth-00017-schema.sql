CREATE TABLE privacy_controlled_accounts_00017 (
  account_id UUID PRIMARY KEY,
  pseudonymous_user_id VARCHAR(80),
  email VARCHAR(255),
  consent_state VARCHAR(32),
  retention_expiry_at TIMESTAMP,
  processor_region VARCHAR(32),
  encryption_key_ref VARCHAR(64),
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);
