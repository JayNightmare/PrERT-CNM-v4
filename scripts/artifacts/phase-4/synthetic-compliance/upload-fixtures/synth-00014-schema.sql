CREATE TABLE governed_customer_registry_00014 (
  customer_uuid UUID PRIMARY KEY,
  privacy_subject_id VARCHAR(80),
  email VARCHAR(255),
  data_portability_token VARCHAR(128),
  retention_window_days INTEGER,
  processor_vendor_code VARCHAR(32),
  security_tier VARCHAR(32),
  created_at TIMESTAMP
);
