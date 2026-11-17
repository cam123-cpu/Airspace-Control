"""Login / refresh / logout flows with MFA, lockout with exponential backoff, and generic failure responses."""
from __future__ import annotations

import hashlib
import hmac
import time

from . import audit
from .challenge import FAILURES, challenge_needed, issue_pow, record_failure, verifier_for
from .auth import (AuthError, MFA_ROLES, dummy_verify, issue_access_token, issue_refresh_token, revoke_family_of,
                   rotate_refresh_token, verify_password, verify_totp)
from .config import Settings
from .db import DB
from .crypto import DecryptError, aad
from .errors import ChallengeRequired, TooMany, Unauthorized
from .ratelimit import check
from .schemas import LoginIn, RefreshIn


def login(s: Settings, db: DB, body: dict, client_ip: str, now: int | None = None) -> dict:
    now = int(now or time.time())
    if not check("login_ip", client_ip):
        raise TooMany()
    req = LoginIn.model_validate(body)
    email = req.email.lower()
    if not check("login_account", email):
        raise TooMany()
    if challenge_needed(s, email, client_ip):
        v = verifier_for(s)
        ch = req.challenge
        if ch is None or ch.kind != v.kind or not v.verify(s, db, ch.model_dump(), client_ip):
            raise ChallengeRequired("solve the challenge and retry",
                                    {"challenge": issue_pow(s, client_ip, now) if v.kind == "pow" else {"kind": v.kind}})
    u = db.one("SELECT id, tenant_id, pw_hash, role, totp_secret, mfa_enabled, totp_last_step, failed_count, locked_until, active FROM users WHERE email = ?", (email,))
    if not u:
        record_failure(email, client_ip)
        dummy_verify(req.password, s.pepper)
        with db.tx():
            audit.append(db, None, "anonymous", "login.failed", "unknown-account", {"ip": client_ip}, now)
        raise Unauthorized()
    if u["locked_until"] > now or not u["active"]:
        record_failure(email, client_ip)
        dummy_verify(req.password, s.pepper)
        raise Unauthorized()          # same response as a wrong password: lockout state isn't revealed
    ok = verify_password(req.password, u["pw_hash"], s.pepper)
    step = None
    if ok and (u["mfa_enabled"] or u["role"] in MFA_ROLES):
        try:
            secret = s.keyring().decrypt_field(u["totp_secret"], aad("users.totp_secret", u["tenant_id"], u["id"])) if u["totp_secret"] else None
        except DecryptError:
            secret = None                      # tampered or foreign ciphertext: fail closed
        step = verify_totp(secret, req.totp or "", u["totp_last_step"], now) if secret else None
        ok = step is not None
    with db.tx():
        if not ok:
            fails = u["failed_count"] + 1
            lock = now + min(3600, 60 * 2 ** (fails - s.lockout_threshold)) if fails >= s.lockout_threshold else 0
            db.x("UPDATE users SET failed_count = ?, locked_until = ? WHERE id = ?", (fails, lock, u["id"]))
            audit.append(db, u["tenant_id"], f"user:{u['id']}", "login.failed", u["id"], {"ip": client_ip, "fails": fails, "locked": bool(lock)}, now)
        else:
            db.x("UPDATE users SET failed_count = 0, locked_until = 0, totp_last_step = ? WHERE id = ?", (step or u["totp_last_step"], u["id"]))
            refresh = issue_refresh_token(s, db, u["id"], None, now)
            audit.append(db, u["tenant_id"], f"user:{u['id']}", "login.ok", u["id"], {"ip": client_ip}, now)
    if not ok:
        record_failure(email, client_ip)
        raise Unauthorized()
    FAILURES.clear("acct:" + email)
    return {"access_token": issue_access_token(s, u["id"], u["tenant_id"], u["role"], now), "token_type": "Bearer",
            "expires_in": s.access_ttl_s, "refresh_token": refresh}


def refresh(s: Settings, db: DB, body: dict, client_ip: str, now: int | None = None) -> dict:
    now = int(now or time.time())
    if not check("refresh_ip", client_ip):
        raise TooMany()
    req = RefreshIn.model_validate(body)
    try:
        with db.tx():
            new_rt, user = rotate_refresh_token(s, db, req.refresh_token, now)
            audit.append(db, user["tenant_id"], f"user:{user['id']}", "token.refreshed", user["id"], {}, now)
    except AuthError as e:
        if "reuse" in str(e):
            with db.tx():
                # rotate_refresh_token's revocation ran inside the rolled-back transaction: re-apply it durably
                revoke_family_of(db, req.refresh_token, now)
                audit.append(db, None, "anonymous", "token.reuse_detected", "refresh-family", {"ip": client_ip}, now)
        raise Unauthorized()
    return {"access_token": issue_access_token(s, user["id"], user["tenant_id"], user["role"], now), "token_type": "Bearer",
            "expires_in": s.access_ttl_s, "refresh_token": new_rt}


def csrf_token_for(s: Settings, refresh_token: str) -> str:
    """Double-submit CSRF token for cookie sessions, bound to the current refresh token (stateless, HMAC)."""
    key = hmac.new(s.pepper, b"airspace-csrf-v1", hashlib.sha256).digest()
    return hmac.new(key, hashlib.sha256(refresh_token.encode()).digest(), hashlib.sha256).hexdigest()


def logout(db: DB, body: dict, now: int | None = None) -> dict:
    req = RefreshIn.model_validate(body)
    with db.tx():
        revoke_family_of(db, req.refresh_token, now)
    return {"ok": True}
