-- Airspace Command: PostgreSQL schema with row-level security (RLS).
--
-- Tenant isolation is enforced twice: by the application (every query filters on tenant_id taken from the verified
-- identity) and by the database (policies below). If application code ever forgets a tenant filter, the database
-- still returns, updates and deletes only the current tenant's rows, and refuses to insert rows for another tenant.
--
-- Per-transaction contract for the application role:
--     BEGIN;
--     SELECT set_config('app.tenant_id', <tenant id from the verified token>, true);   -- true = transaction-local
--     ... queries ...
--     COMMIT;
-- With app.tenant_id unset, every tenant table reads as empty.
--
-- Roles (created idempotently; run this file as a superuser or a role with CREATEROLE):
--   airspace_owner        owns the schema and tables. FORCE ROW LEVEL SECURITY applies to it too.
--   airspace_app          what the API connects as (via a LOGIN user granted this role). No BYPASSRLS.
--   airspace_definer      owns the narrow SECURITY DEFINER lookups used before the tenant is known (login by email,
--                         refresh token, device and API-token lookup). BYPASSRLS, NOLOGIN, cannot be connected as.
--   airspace_maintenance  CLI-only jobs that span tenants (bootstrap, retention purge, key rewrap). BYPASSRLS,
--                         separate credentials, never used by the API.

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'airspace_owner') THEN
    CREATE ROLE airspace_owner NOLOGIN NOBYPASSRLS;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'airspace_app') THEN
    CREATE ROLE airspace_app NOLOGIN NOBYPASSRLS;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'airspace_definer') THEN
    CREATE ROLE airspace_definer NOLOGIN BYPASSRLS;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'airspace_maintenance') THEN
    CREATE ROLE airspace_maintenance NOLOGIN BYPASSRLS;
  END IF;
END
$$;

CREATE SCHEMA IF NOT EXISTS airspace AUTHORIZATION airspace_owner;
REVOKE ALL ON SCHEMA airspace FROM PUBLIC;
GRANT USAGE ON SCHEMA airspace TO airspace_app, airspace_definer, airspace_maintenance;

SET search_path = airspace, pg_catalog;
SET ROLE airspace_owner;

-- The tenant of the current transaction (NULL when unset, which matches no rows).
CREATE OR REPLACE FUNCTION airspace.app_tenant() RETURNS text
  LANGUAGE sql STABLE
  AS $$ SELECT NULLIF(current_setting('app.tenant_id', true), '') $$;

CREATE TABLE tenants (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL CHECK (length(name) BETWEEN 1 AND 120),
  retention_days INTEGER NOT NULL DEFAULT 180 CHECK (retention_days BETWEEN 30 AND 3650),
  created_at BIGINT NOT NULL
);

CREATE TABLE users (
  id TEXT PRIMARY KEY,
  tenant_id TEXT REFERENCES tenants(id),
  email TEXT NOT NULL UNIQUE CHECK (length(email) BETWEEN 3 AND 254),
  pw_hash TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('viewer','operator','supervisor','tenant_admin','platform_auditor')),
  totp_secret TEXT,                      -- AES-256-GCM ciphertext (enc:v1:...)
  mfa_enabled BOOLEAN NOT NULL DEFAULT false,
  totp_last_step BIGINT NOT NULL DEFAULT 0,
  failed_count INTEGER NOT NULL DEFAULT 0,
  locked_until BIGINT NOT NULL DEFAULT 0,
  active BOOLEAN NOT NULL DEFAULT true,
  created_at BIGINT NOT NULL,
  CHECK (role = 'platform_auditor' OR tenant_id IS NOT NULL),
  UNIQUE (tenant_id, id)
);

CREATE TABLE refresh_tokens (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  family_id TEXT NOT NULL,
  token_hash TEXT NOT NULL UNIQUE,
  expires_at BIGINT NOT NULL,
  used_at BIGINT,
  revoked_at BIGINT,
  created_at BIGINT NOT NULL
);
CREATE INDEX ix_refresh_family ON refresh_tokens(family_id);

CREATE TABLE sites (
  id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL REFERENCES tenants(id),
  name TEXT NOT NULL CHECK (length(name) BETWEEN 1 AND 120),
  alert_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.85 CHECK (alert_threshold BETWEEN 0.5 AND 0.99),
  created_at BIGINT NOT NULL,
  UNIQUE (tenant_id, id)
);

CREATE TABLE devices (
  id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  site_id TEXT NOT NULL,
  public_key TEXT NOT NULL,
  active BOOLEAN NOT NULL DEFAULT true,
  created_at BIGINT NOT NULL,
  FOREIGN KEY (tenant_id, site_id) REFERENCES sites(tenant_id, id),
  UNIQUE (tenant_id, id)
);

CREATE TABLE sensors (
  id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  site_id TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('radar','rf','acoustic','camera_vis','camera_ir','remote_id')),
  enabled BOOLEAN NOT NULL DEFAULT true,
  status TEXT NOT NULL DEFAULT 'unknown' CHECK (status IN ('ok','degraded','offline','unknown')),
  last_seen BIGINT,
  FOREIGN KEY (tenant_id, site_id) REFERENCES sites(tenant_id, id),
  UNIQUE (tenant_id, id)
);

CREATE TABLE nonces (
  device_id TEXT NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
  nonce TEXT NOT NULL,
  expires_at BIGINT NOT NULL,
  PRIMARY KEY (device_id, nonce)
);

CREATE TABLE idempotency (
  tenant_id TEXT NOT NULL REFERENCES tenants(id),
  device_id TEXT NOT NULL,
  key TEXT NOT NULL,
  response TEXT NOT NULL,
  created_at BIGINT NOT NULL,
  PRIMARY KEY (tenant_id, device_id, key)
);

CREATE TABLE tracks (
  seq BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  id TEXT NOT NULL UNIQUE,
  tenant_id TEXT NOT NULL,
  site_id TEXT NOT NULL,
  ext_track_id TEXT NOT NULL,
  cls TEXT NOT NULL,
  confidence DOUBLE PRECISION NOT NULL,
  x DOUBLE PRECISION NOT NULL, y DOUBLE PRECISION NOT NULL, z DOUBLE PRECISION NOT NULL,
  observed_at BIGINT NOT NULL,
  created_at BIGINT NOT NULL,
  FOREIGN KEY (tenant_id, site_id) REFERENCES sites(tenant_id, id)
);
CREATE INDEX ix_tracks_scope ON tracks(tenant_id, site_id, observed_at);

CREATE TABLE alerts (
  seq BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  id TEXT NOT NULL UNIQUE,
  tenant_id TEXT NOT NULL,
  site_id TEXT NOT NULL,
  ext_track_id TEXT NOT NULL,
  severity TEXT NOT NULL CHECK (severity IN ('suspect','alert','spoof')),
  title TEXT NOT NULL,
  remote_id_text TEXT,                   -- AES-256-GCM ciphertext
  status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','acknowledged','resolved')),
  created_at BIGINT NOT NULL,
  ack_by TEXT, ack_at BIGINT,
  resolved_by TEXT, resolved_at BIGINT, resolution_note TEXT,   -- resolution_note: AES-256-GCM ciphertext
  legal_hold BOOLEAN NOT NULL DEFAULT false,
  FOREIGN KEY (tenant_id, site_id) REFERENCES sites(tenant_id, id),
  UNIQUE (tenant_id, id)
);
CREATE INDEX ix_alerts_scope ON alerts(tenant_id, status, seq);

CREATE TABLE evidence (
  id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  site_id TEXT NOT NULL,
  alert_id TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  size INTEGER NOT NULL,
  content_type TEXT NOT NULL CHECK (content_type IN ('image/jpeg','image/png')),
  storage_key TEXT NOT NULL UNIQUE,
  server_sig TEXT NOT NULL,
  created_at BIGINT NOT NULL,
  legal_hold BOOLEAN NOT NULL DEFAULT false,
  FOREIGN KEY (tenant_id, alert_id) REFERENCES alerts(tenant_id, id),
  FOREIGN KEY (tenant_id, site_id) REFERENCES sites(tenant_id, id),
  UNIQUE (tenant_id, id)
);

CREATE TABLE approvals (
  id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL REFERENCES tenants(id),
  action TEXT NOT NULL CHECK (action IN ('disable_sensor','enable_sensor','set_alert_threshold','set_retention','export_evidence')),
  params TEXT NOT NULL,
  requested_by TEXT NOT NULL,
  requested_at BIGINT NOT NULL,
  expires_at BIGINT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','executed','rejected','expired')),
  decided_by TEXT,
  decided_at BIGINT,
  FOREIGN KEY (tenant_id, requested_by) REFERENCES users(tenant_id, id)
);

CREATE TABLE export_grants (
  id_hash TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  evidence_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  expires_at BIGINT NOT NULL,
  used_at BIGINT,
  FOREIGN KEY (tenant_id, evidence_id) REFERENCES evidence(tenant_id, id) ON DELETE CASCADE
);

CREATE TABLE api_tokens (
  id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL REFERENCES tenants(id),
  name TEXT NOT NULL,
  token_hash TEXT NOT NULL UNIQUE,
  role TEXT NOT NULL CHECK (role IN ('viewer','operator')),
  allowed_tools TEXT NOT NULL,
  expires_at BIGINT NOT NULL,
  revoked_at BIGINT,
  created_by TEXT NOT NULL,
  created_at BIGINT NOT NULL
);

CREATE TABLE confirmations (
  code_hash TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  alert_id TEXT NOT NULL,
  created_by TEXT NOT NULL,
  expires_at BIGINT NOT NULL,
  used_at BIGINT,
  FOREIGN KEY (tenant_id, alert_id) REFERENCES alerts(tenant_id, id)
);

CREATE TABLE audit_log (
  seq BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  ts BIGINT NOT NULL,
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  target TEXT NOT NULL,
  detail TEXT NOT NULL,
  prev_hash TEXT NOT NULL,
  hash TEXT NOT NULL
);
CREATE INDEX ix_audit_tenant ON audit_log(tenant_id, seq);

CREATE TABLE pow_used (                  -- single-use proof-of-work challenges (hashes only; not tenant data)
  token_hash TEXT PRIMARY KEY,
  expires_at BIGINT NOT NULL
);

CREATE TABLE rate_buckets (              -- shared auth rate-limit buckets (hashed keys; not tenant data)
  key_hash TEXT PRIMARY KEY CHECK (length(key_hash) = 64),
  tokens DOUBLE PRECISION NOT NULL,
  updated_at DOUBLE PRECISION NOT NULL
);
CREATE INDEX ix_rate_buckets_updated ON rate_buckets(updated_at);

-- Append-only audit log (also enforced by grants: the app role has no UPDATE/DELETE on it).
CREATE OR REPLACE FUNCTION airspace.audit_append_only() RETURNS trigger
  LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'audit log is append-only'; END $$;
CREATE TRIGGER audit_no_update BEFORE UPDATE ON audit_log FOR EACH ROW EXECUTE FUNCTION airspace.audit_append_only();
CREATE TRIGGER audit_no_delete BEFORE DELETE ON audit_log FOR EACH ROW EXECUTE FUNCTION airspace.audit_append_only();

-- ---------------------------------------------------------------------------------------------------------------
-- Row-level security. ENABLE turns policies on; FORCE applies them to the table owner as well.
-- ---------------------------------------------------------------------------------------------------------------
DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['sites','devices','sensors','idempotency','tracks','alerts','evidence','approvals',
                           'export_grants','api_tokens','confirmations','audit_log']
  LOOP
    EXECUTE format('ALTER TABLE airspace.%I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('ALTER TABLE airspace.%I FORCE ROW LEVEL SECURITY', t);
    EXECUTE format('CREATE POLICY tenant_isolation ON airspace.%I USING (tenant_id = airspace.app_tenant()) '
                   'WITH CHECK (tenant_id = airspace.app_tenant())', t);
  END LOOP;
END
$$;

ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenants FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON tenants USING (id = airspace.app_tenant()) WITH CHECK (id = airspace.app_tenant());

-- Platform auditors have no tenant; they are visible only in the platform scope.
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE users FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON users
  USING (tenant_id = airspace.app_tenant() OR (tenant_id IS NULL AND airspace.app_tenant() = '_platform'))
  WITH CHECK (tenant_id = airspace.app_tenant() OR (tenant_id IS NULL AND airspace.app_tenant() = '_platform'));

-- Tables without their own tenant_id inherit it through their parent row.
ALTER TABLE refresh_tokens ENABLE ROW LEVEL SECURITY;
ALTER TABLE refresh_tokens FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON refresh_tokens
  USING (EXISTS (SELECT 1 FROM airspace.users u WHERE u.id = refresh_tokens.user_id))
  WITH CHECK (EXISTS (SELECT 1 FROM airspace.users u WHERE u.id = refresh_tokens.user_id));

ALTER TABLE nonces ENABLE ROW LEVEL SECURITY;
ALTER TABLE nonces FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON nonces
  USING (EXISTS (SELECT 1 FROM airspace.devices d WHERE d.id = nonces.device_id))
  WITH CHECK (EXISTS (SELECT 1 FROM airspace.devices d WHERE d.id = nonces.device_id));

-- ---------------------------------------------------------------------------------------------------------------
-- Grants: least privilege. The app role can't alter schema, can't touch the audit log except to append.
-- ---------------------------------------------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON tenants, users, refresh_tokens, sites, devices, sensors, nonces, idempotency,
  tracks, alerts, evidence, approvals, export_grants, api_tokens, confirmations, pow_used, rate_buckets TO airspace_app;
GRANT SELECT, INSERT ON audit_log TO airspace_app;
GRANT SELECT ON ALL TABLES IN SCHEMA airspace TO airspace_definer;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA airspace TO airspace_maintenance;
REVOKE UPDATE, DELETE ON audit_log FROM airspace_maintenance;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA airspace TO airspace_app, airspace_maintenance;

RESET ROLE;

-- ---------------------------------------------------------------------------------------------------------------
-- Narrow lookups that run before the tenant is known. SECURITY DEFINER as airspace_definer (BYPASSRLS), with a
-- pinned search_path, returning only the columns the auth path needs for exactly one row.
-- ---------------------------------------------------------------------------------------------------------------
GRANT CREATE ON SCHEMA airspace TO airspace_definer;
SET ROLE airspace_definer;

CREATE OR REPLACE FUNCTION airspace.auth_user_by_email(p_email text)
  RETURNS TABLE (id text, tenant_id text, pw_hash text, role text, totp_secret text, mfa_enabled boolean,
                 totp_last_step bigint, failed_count integer, locked_until bigint, active boolean)
  LANGUAGE sql STABLE SECURITY DEFINER SET search_path = airspace, pg_temp
  AS $$ SELECT id, tenant_id, pw_hash, role, totp_secret, mfa_enabled, totp_last_step, failed_count, locked_until, active
        FROM airspace.users WHERE email = lower(p_email) LIMIT 1 $$;

CREATE OR REPLACE FUNCTION airspace.auth_user_by_id(p_id text)
  RETURNS TABLE (id text, tenant_id text, role text, active boolean)
  LANGUAGE sql STABLE SECURITY DEFINER SET search_path = airspace, pg_temp
  AS $$ SELECT id, tenant_id, role, active FROM airspace.users WHERE id = p_id LIMIT 1 $$;

CREATE OR REPLACE FUNCTION airspace.auth_refresh_by_hash(p_hash text)
  RETURNS TABLE (id text, user_id text, family_id text, expires_at bigint, used_at bigint, revoked_at bigint, tenant_id text)
  LANGUAGE sql STABLE SECURITY DEFINER SET search_path = airspace, pg_temp
  AS $$ SELECT r.id, r.user_id, r.family_id, r.expires_at, r.used_at, r.revoked_at, u.tenant_id
        FROM airspace.refresh_tokens r JOIN airspace.users u ON u.id = r.user_id WHERE r.token_hash = p_hash LIMIT 1 $$;

CREATE OR REPLACE FUNCTION airspace.auth_device_by_id(p_id text)
  RETURNS TABLE (id text, tenant_id text, site_id text, public_key text, active boolean)
  LANGUAGE sql STABLE SECURITY DEFINER SET search_path = airspace, pg_temp
  AS $$ SELECT id, tenant_id, site_id, public_key, active FROM airspace.devices WHERE id = p_id LIMIT 1 $$;

CREATE OR REPLACE FUNCTION airspace.auth_api_token_by_hash(p_hash text)
  RETURNS TABLE (id text, tenant_id text, role text, allowed_tools text, expires_at bigint, revoked_at bigint)
  LANGUAGE sql STABLE SECURITY DEFINER SET search_path = airspace, pg_temp
  AS $$ SELECT id, tenant_id, role, allowed_tools, expires_at, revoked_at FROM airspace.api_tokens
        WHERE token_hash = p_hash LIMIT 1 $$;

RESET ROLE;
REVOKE CREATE ON SCHEMA airspace FROM airspace_definer;
REVOKE ALL ON FUNCTION airspace.auth_user_by_email(text), airspace.auth_user_by_id(text), airspace.auth_refresh_by_hash(text),
  airspace.auth_device_by_id(text), airspace.auth_api_token_by_hash(text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION airspace.auth_user_by_email(text), airspace.auth_user_by_id(text), airspace.auth_refresh_by_hash(text),
  airspace.auth_device_by_id(text), airspace.auth_api_token_by_hash(text) TO airspace_app;
GRANT EXECUTE ON FUNCTION airspace.app_tenant() TO airspace_app, airspace_definer, airspace_maintenance;
