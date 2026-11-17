"""Tenant administration: users (MFA provisioned at creation), edge devices, scoped API tokens, and bootstrap."""
from __future__ import annotations

import json
import secrets
import time
import uuid

from . import audit
from .auth import MFA_ROLES, Principal, hash_password, new_totp_secret, sha256_hex, totp_uri
from .config import Settings
from .crypto import aad
from .db import DB
from .errors import Conflict, NotFound
from .ingest import b64u_decode
from .rbac import require
from .schemas import ApiTokenIn, DeviceIn, UserIn


def create_user(s: Settings, db: DB, p: Principal, body: dict, now: int | None = None) -> dict:
    require(p, "users.manage")
    u = UserIn.model_validate(body)
    now = int(now or time.time())
    uid = uuid.uuid4().hex
    secret = new_totp_secret()
    with db.tx():
        if db.one("SELECT id FROM users WHERE email = ?", (u.email.lower(),)):
            raise Conflict("email already registered")
        db.x("INSERT INTO users (id, tenant_id, email, pw_hash, role, totp_secret, mfa_enabled, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
             (uid, p.tenant_id, u.email.lower(), hash_password(u.password, s.pepper), u.role,
              s.keyring().encrypt_field(secret, aad("users.totp_secret", p.tenant_id, uid)), 1, now))
        audit.append(db, p.tenant_id, p.actor, "user.created", uid, {"role": u.role}, now)
    # the TOTP secret is shown exactly once, to the admin enrolling the user in person
    return {"id": uid, "role": u.role, "mfa_required": u.role in MFA_ROLES, "totp_provisioning_uri": totp_uri(secret, u.email.lower())}


def register_device(db: DB, p: Principal, body: dict, now: int | None = None) -> dict:
    require(p, "devices.manage")
    d = DeviceIn.model_validate(body)
    if len(b64u_decode(d.public_key)) != 32:
        raise NotFound()
    now = int(now or time.time())
    did = "dev_" + secrets.token_hex(8)
    with db.tx():
        if not db.one("SELECT id FROM sites WHERE tenant_id = ? AND id = ?", (p.tenant_id, d.site_id)):
            raise NotFound()
        db.x("INSERT INTO devices (id, tenant_id, site_id, public_key, created_at) VALUES (?, ?, ?, ?, ?)", (did, p.tenant_id, d.site_id, d.public_key, now))
        audit.append(db, p.tenant_id, p.actor, "device.registered", did, {"site": d.site_id}, now)
    return {"id": did, "site_id": d.site_id}


def create_api_token(db: DB, p: Principal, body: dict, now: int | None = None) -> dict:
    require(p, "api_tokens.manage")
    t = ApiTokenIn.model_validate(body)
    if "acknowledge_alert" in t.allowed_tools and t.role != "operator":
        raise Conflict("acknowledge_alert needs an operator-role token")
    now = int(now or time.time())
    raw = "ast_" + secrets.token_urlsafe(32)
    tid = uuid.uuid4().hex
    with db.tx():
        db.x("INSERT INTO api_tokens (id, tenant_id, name, token_hash, role, allowed_tools, expires_at, created_by, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
             (tid, p.tenant_id, t.name, sha256_hex(raw), t.role, json.dumps(sorted(set(t.allowed_tools))), now + t.ttl_hours * 3600, p.actor, now))
        audit.append(db, p.tenant_id, p.actor, "api_token.created", tid, {"role": t.role, "tools": sorted(set(t.allowed_tools))}, now)
    return {"id": tid, "token": raw, "expires_at": now + t.ttl_hours * 3600}


def revoke_api_token(db: DB, p: Principal, token_id: str, now: int | None = None) -> dict:
    require(p, "api_tokens.manage")
    now = int(now or time.time())
    with db.tx():
        if db.x("UPDATE api_tokens SET revoked_at = ? WHERE tenant_id = ? AND id = ? AND revoked_at IS NULL", (now, p.tenant_id, token_id)) == 0:
            raise NotFound()
        audit.append(db, p.tenant_id, p.actor, "api_token.revoked", token_id, {}, now)
    return {"id": token_id, "revoked": True}


# ---------------- bootstrap (CLI only, never exposed over HTTP) ----------------
def bootstrap_tenant(s: Settings, db: DB, tenant_name: str, admin_email: str, admin_password: str, now: int | None = None) -> dict:
    now = int(now or time.time())
    tid, uid, secret = "tn_" + secrets.token_hex(8), uuid.uuid4().hex, new_totp_secret()
    with db.tx():
        db.x("INSERT INTO tenants (id, name, created_at) VALUES (?, ?, ?)", (tid, tenant_name, now))
        db.x("INSERT INTO users (id, tenant_id, email, pw_hash, role, totp_secret, mfa_enabled, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
             (uid, tid, admin_email.lower(), hash_password(admin_password, s.pepper), "tenant_admin",
              s.keyring().encrypt_field(secret, aad("users.totp_secret", tid, uid)), 1, now))
        audit.append(db, tid, "system:bootstrap", "tenant.created", tid, {"admin": uid}, now)
    return {"tenant_id": tid, "admin_user_id": uid, "totp_secret": secret}


def create_site(db: DB, tenant_id: str, name: str, sensor_kinds: list[str], now: int | None = None) -> dict:
    now = int(now or time.time())
    sid = "site_" + secrets.token_hex(6)
    sensors = []
    with db.tx():
        db.x("INSERT INTO sites (id, tenant_id, name, created_at) VALUES (?, ?, ?, ?)", (sid, tenant_id, name, now))
        for k in sensor_kinds:
            sen = "sen_" + secrets.token_hex(6)
            db.x("INSERT INTO sensors (id, tenant_id, site_id, kind) VALUES (?, ?, ?, ?)", (sen, tenant_id, sid, k))
            sensors.append({"id": sen, "kind": k})
        audit.append(db, tenant_id, "system:cli", "site.created", sid, {"sensors": len(sensors)}, now)
    return {"site_id": sid, "sensors": sensors}


def create_platform_auditor(s: Settings, db: DB, email: str, password: str, now: int | None = None) -> dict:
    now = int(now or time.time())
    uid, secret = uuid.uuid4().hex, new_totp_secret()
    with db.tx():
        db.x("INSERT INTO users (id, tenant_id, email, pw_hash, role, totp_secret, mfa_enabled, created_at) VALUES (?, NULL, ?, ?, 'platform_auditor', ?, 1, ?)",
             (uid, email.lower(), hash_password(password, s.pepper), s.keyring().encrypt_field(secret, aad("users.totp_secret", None, uid)), now))
        audit.append(db, None, "system:cli", "auditor.created", uid, {}, now)
    return {"user_id": uid, "totp_secret": secret}
