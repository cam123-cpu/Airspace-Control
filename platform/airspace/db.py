"""Database access. Every statement in this codebase is a string LITERAL with `?` placeholders.

Rules (enforced by tests/test_sql_safety.py, which AST-scans the package):
  * the SQL argument of q()/one()/x()/execute() is always a constant string, never an f-string, concat or %-format;
  * values travel only as bound parameters;
  * every tenant-owned table carries tenant_id and every query on it filters by tenant_id;
  * composite foreign keys (tenant_id, id) make cross-tenant references impossible at the schema level.

Postgres: the same literals work after a mechanical `?` -> `%s` (psycopg) swap. ops/postgres/schema.sql adds
row-level security as a second, independent isolation layer: every transaction must first run
`SET LOCAL app.tenant_id = <tenant>` (see ops/postgres/README.md), and the database itself hides other
tenants' rows even if a query forgets its tenant filter.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterator

MIGRATIONS: list[tuple[int, str]] = [
    (1, """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);
CREATE TABLE tenants (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL CHECK (length(name) BETWEEN 1 AND 120),
  retention_days INTEGER NOT NULL DEFAULT 180 CHECK (retention_days BETWEEN 30 AND 3650),
  created_at INTEGER NOT NULL
);
CREATE TABLE users (
  id TEXT PRIMARY KEY,
  tenant_id TEXT REFERENCES tenants(id),
  email TEXT NOT NULL UNIQUE CHECK (length(email) BETWEEN 3 AND 254),
  pw_hash TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('viewer','operator','supervisor','tenant_admin','platform_auditor')),
  totp_secret TEXT,
  mfa_enabled INTEGER NOT NULL DEFAULT 0,
  totp_last_step INTEGER NOT NULL DEFAULT 0,
  failed_count INTEGER NOT NULL DEFAULT 0,
  locked_until INTEGER NOT NULL DEFAULT 0,
  active INTEGER NOT NULL DEFAULT 1,
  created_at INTEGER NOT NULL,
  CHECK (role = 'platform_auditor' OR tenant_id IS NOT NULL),
  UNIQUE (tenant_id, id)
);
CREATE TABLE refresh_tokens (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  family_id TEXT NOT NULL,
  token_hash TEXT NOT NULL UNIQUE,
  expires_at INTEGER NOT NULL,
  used_at INTEGER,
  revoked_at INTEGER,
  created_at INTEGER NOT NULL
);
CREATE INDEX ix_refresh_family ON refresh_tokens(family_id);
CREATE TABLE sites (
  id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL REFERENCES tenants(id),
  name TEXT NOT NULL CHECK (length(name) BETWEEN 1 AND 120),
  alert_threshold REAL NOT NULL DEFAULT 0.85 CHECK (alert_threshold BETWEEN 0.5 AND 0.99),
  created_at INTEGER NOT NULL,
  UNIQUE (tenant_id, id)
);
CREATE TABLE devices (
  id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  site_id TEXT NOT NULL,
  public_key TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 1,
  created_at INTEGER NOT NULL,
  FOREIGN KEY (tenant_id, site_id) REFERENCES sites(tenant_id, id),
  UNIQUE (tenant_id, id)
);
CREATE TABLE sensors (
  id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  site_id TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('radar','rf','acoustic','camera_vis','camera_ir','remote_id')),
  enabled INTEGER NOT NULL DEFAULT 1,
  status TEXT NOT NULL DEFAULT 'unknown' CHECK (status IN ('ok','degraded','offline','unknown')),
  last_seen INTEGER,
  FOREIGN KEY (tenant_id, site_id) REFERENCES sites(tenant_id, id),
  UNIQUE (tenant_id, id)
);
CREATE TABLE nonces (
  device_id TEXT NOT NULL,
  nonce TEXT NOT NULL,
  expires_at INTEGER NOT NULL,
  PRIMARY KEY (device_id, nonce)
);
CREATE TABLE idempotency (
  tenant_id TEXT NOT NULL,
  device_id TEXT NOT NULL,
  key TEXT NOT NULL,
  response TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  PRIMARY KEY (tenant_id, device_id, key)
);
CREATE TABLE tracks (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  id TEXT NOT NULL UNIQUE,
  tenant_id TEXT NOT NULL,
  site_id TEXT NOT NULL,
  ext_track_id TEXT NOT NULL,
  cls TEXT NOT NULL,
  confidence REAL NOT NULL,
  x REAL NOT NULL, y REAL NOT NULL, z REAL NOT NULL,
  observed_at INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  FOREIGN KEY (tenant_id, site_id) REFERENCES sites(tenant_id, id)
);
CREATE INDEX ix_tracks_scope ON tracks(tenant_id, site_id, observed_at);
CREATE TABLE alerts (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  id TEXT NOT NULL UNIQUE,
  tenant_id TEXT NOT NULL,
  site_id TEXT NOT NULL,
  ext_track_id TEXT NOT NULL,
  severity TEXT NOT NULL CHECK (severity IN ('suspect','alert','spoof')),
  title TEXT NOT NULL,
  remote_id_text TEXT,
  status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','acknowledged','resolved')),
  created_at INTEGER NOT NULL,
  ack_by TEXT, ack_at INTEGER,
  resolved_by TEXT, resolved_at INTEGER, resolution_note TEXT,
  legal_hold INTEGER NOT NULL DEFAULT 0,
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
  created_at INTEGER NOT NULL,
  legal_hold INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY (tenant_id, alert_id) REFERENCES alerts(tenant_id, id),
  FOREIGN KEY (tenant_id, site_id) REFERENCES sites(tenant_id, id)
);
CREATE TABLE approvals (
  id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL REFERENCES tenants(id),
  action TEXT NOT NULL CHECK (action IN ('disable_sensor','enable_sensor','set_alert_threshold','set_retention','export_evidence')),
  params TEXT NOT NULL,
  requested_by TEXT NOT NULL,
  requested_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','executed','rejected','expired')),
  decided_by TEXT,
  decided_at INTEGER,
  FOREIGN KEY (tenant_id, requested_by) REFERENCES users(tenant_id, id)
);
CREATE TABLE export_grants (
  id_hash TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  evidence_id TEXT NOT NULL REFERENCES evidence(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL,
  expires_at INTEGER NOT NULL,
  used_at INTEGER
);
CREATE TABLE api_tokens (
  id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL REFERENCES tenants(id),
  name TEXT NOT NULL,
  token_hash TEXT NOT NULL UNIQUE,
  role TEXT NOT NULL CHECK (role IN ('viewer','operator')),
  allowed_tools TEXT NOT NULL,
  expires_at INTEGER NOT NULL,
  revoked_at INTEGER,
  created_by TEXT NOT NULL,
  created_at INTEGER NOT NULL
);
CREATE TABLE confirmations (
  code_hash TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  alert_id TEXT NOT NULL,
  created_by TEXT NOT NULL,
  expires_at INTEGER NOT NULL,
  used_at INTEGER,
  FOREIGN KEY (tenant_id, alert_id) REFERENCES alerts(tenant_id, id)
);
CREATE TABLE audit_log (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  tenant_id TEXT NOT NULL,
  ts INTEGER NOT NULL,
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  target TEXT NOT NULL,
  detail TEXT NOT NULL,
  prev_hash TEXT NOT NULL,
  hash TEXT NOT NULL
);
CREATE INDEX ix_audit_tenant ON audit_log(tenant_id, seq);
CREATE TRIGGER audit_no_update BEFORE UPDATE ON audit_log BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER audit_no_delete BEFORE DELETE ON audit_log BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
"""),
    (2, """
CREATE TABLE pow_used (
  token_hash TEXT PRIMARY KEY,
  expires_at INTEGER NOT NULL
);
"""),
]


class DB:
    """Thin wrapper: one connection per request/unit of work. Only literal SQL + bound params."""

    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, timeout=10, isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA busy_timeout = 10000")
        self.conn.execute("PRAGMA secure_delete = ON")

    def q(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        r = self.conn.execute(sql, params).fetchone()
        return dict(r) if r is not None else None

    def x(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        return self.conn.execute(sql, params).rowcount

    @contextmanager
    def tx(self) -> Iterator["DB"]:
        """Serializable write transaction (BEGIN IMMEDIATE takes the write lock up front)."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    def close(self) -> None:
        self.conn.close()


_migrate_lock = threading.Lock()


def migrate(path: str) -> None:
    with _migrate_lock:
        db = DB(path)
        try:
            db.conn.execute("PRAGMA journal_mode = WAL")
            db.conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)")
            row = db.one("SELECT MAX(version) AS v FROM schema_version")
            current = (row or {}).get("v") or 0
            for version, script in MIGRATIONS:
                if version > current:
                    _apply(db, version, script)
        finally:
            db.close()


def _apply(db: DB, version: int, script: str) -> None:
    # executescript manages its own transaction; record the version in the same script run.
    db.conn.executescript(script)
    db.x("INSERT INTO schema_version (version) VALUES (?)", (version,))
