"""Authentication primitives: scrypt+pepper passwords, RFC 6238 TOTP, Ed25519 JWTs, rotating refresh tokens."""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import struct
import time
import uuid
from dataclasses import dataclass

import jwt

from .config import Settings
from .db import DB

SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 15, 8, 1
MFA_ROLES = {"operator", "supervisor", "tenant_admin", "platform_auditor"}


class AuthError(Exception):
    """Generic authentication failure. The message is never shown to clients."""


# ---------------- passwords ----------------
def _peppered(password: str, pepper: bytes) -> bytes:
    return hmac.new(pepper, password.encode("utf-8"), hashlib.sha256).digest()


def hash_password(password: str, pepper: bytes) -> str:
    if not (12 <= len(password) <= 256):
        raise ValueError("password must be 12-256 characters")
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(_peppered(password, pepper), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, maxmem=64 * 1024 * 1024, dklen=32)
    return "scrypt${}${}${}${}${}".format(SCRYPT_N, SCRYPT_R, SCRYPT_P, base64.b64encode(salt).decode(), base64.b64encode(dk).decode())


def verify_password(password: str, stored: str, pepper: bytes) -> bool:
    try:
        algo, n, r, p, salt_b64, dk_b64 = stored.split("$")
        if algo != "scrypt":
            return False
        dk = hashlib.scrypt(_peppered(password[:256], pepper), salt=base64.b64decode(salt_b64), n=int(n), r=int(r), p=int(p),
                            maxmem=64 * 1024 * 1024, dklen=32)
        return hmac.compare_digest(dk, base64.b64decode(dk_b64))
    except (ValueError, TypeError):
        return False


_DUMMY_HASH: str | None = None


def dummy_verify(password: str, pepper: bytes) -> None:
    """Spend the same time as a real check when the user doesn't exist (no account enumeration by timing)."""
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password("x" * 16, pepper)
    verify_password(password, _DUMMY_HASH, pepper)


# ---------------- TOTP (RFC 6238, SHA-1, 30 s, 6 digits) ----------------
def new_totp_secret() -> str:
    return base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")


def _hotp(secret_b32: str, counter: int) -> str:
    key = base64.b32decode(secret_b32 + "=" * (-len(secret_b32) % 8))
    digest = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    off = digest[-1] & 0x0F
    code = (struct.unpack(">I", digest[off:off + 4])[0] & 0x7FFFFFFF) % 1_000_000
    return f"{code:06d}"


def totp_now(secret_b32: str, t: float | None = None) -> str:
    return _hotp(secret_b32, int((t if t is not None else time.time()) // 30))


def verify_totp(secret_b32: str, code: str, last_step: int, t: float | None = None) -> int | None:
    """Returns the matched time step (to store, preventing replay) or None. Window ±1 step."""
    if not (isinstance(code, str) and len(code) == 6 and code.isdigit()):
        return None
    step = int((t if t is not None else time.time()) // 30)
    matched = None
    for s in (step - 1, step, step + 1):
        if hmac.compare_digest(_hotp(secret_b32, s), code) and s > last_step:
            matched = s
    return matched


def totp_uri(secret_b32: str, email: str) -> str:
    from urllib.parse import quote
    return f"otpauth://totp/AirspaceCommand:{quote(email)}?secret={secret_b32}&issuer=AirspaceCommand"


# ---------------- access tokens (EdDSA JWT) ----------------
@dataclass(frozen=True)
class Principal:
    kind: str            # "user" | "api_token" | "device"
    id: str
    tenant_id: str | None
    role: str
    allowed_tools: frozenset[str] = frozenset()

    @property
    def actor(self) -> str:
        return f"{self.kind}:{self.id}"


def issue_access_token(s: Settings, user_id: str, tenant_id: str | None, role: str, now: int | None = None) -> str:
    now = int(now or time.time())
    claims = {"iss": s.issuer, "aud": s.audience, "sub": user_id, "tid": tenant_id, "role": role,
              "iat": now, "nbf": now - 5, "exp": now + s.access_ttl_s, "jti": uuid.uuid4().hex}
    return jwt.encode(claims, s.jwt_keys[s.current_kid], algorithm="EdDSA", headers={"kid": s.current_kid, "typ": "at+jwt"})


def verify_access_token(s: Settings, token: str) -> dict:
    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        if header.get("alg") != "EdDSA" or kid not in s.jwt_keys:
            raise AuthError("bad header")
        return jwt.decode(token, s.jwt_keys[kid].public_key(), algorithms=["EdDSA"], audience=s.audience, issuer=s.issuer,
                          options={"require": ["exp", "nbf", "iat", "iss", "aud", "sub", "jti"]}, leeway=5)
    except jwt.PyJWTError as e:
        raise AuthError("invalid token") from e


def principal_from_access_token(s: Settings, db: DB, token: str) -> Principal:
    claims = verify_access_token(s, token)
    # re-check the account on every request: disabling a user or changing a role takes effect immediately
    u = db.one("SELECT id, tenant_id, role, active FROM users WHERE id = ?", (claims["sub"],))
    if not u or not u["active"] or u["role"] != claims.get("role") or u["tenant_id"] != claims.get("tid"):
        raise AuthError("stale token")
    return Principal("user", u["id"], u["tenant_id"], u["role"])


# ---------------- refresh tokens (opaque, rotating, reuse detection) ----------------
def sha256_hex(v: str | bytes) -> str:
    return hashlib.sha256(v.encode() if isinstance(v, str) else v).hexdigest()


def issue_refresh_token(s: Settings, db: DB, user_id: str, family_id: str | None = None, now: int | None = None) -> str:
    now = int(now or time.time())
    raw = "rt_" + secrets.token_urlsafe(32)
    db.x("INSERT INTO refresh_tokens (id, user_id, family_id, token_hash, expires_at, created_at) VALUES (?, ?, ?, ?, ?, ?)",
         (uuid.uuid4().hex, user_id, family_id or uuid.uuid4().hex, sha256_hex(raw), now + s.refresh_ttl_s, now))
    return raw


def rotate_refresh_token(s: Settings, db: DB, raw: str, now: int | None = None) -> tuple[str, dict]:
    """Consume a refresh token and return (new_token, user). A second use of the same token revokes the whole family."""
    now = int(now or time.time())
    row = db.one("SELECT id, user_id, family_id, expires_at, used_at, revoked_at FROM refresh_tokens WHERE token_hash = ?", (sha256_hex(raw),))
    if not row:
        raise AuthError("unknown refresh token")
    if row["used_at"] is not None or row["revoked_at"] is not None:
        db.x("UPDATE refresh_tokens SET revoked_at = ? WHERE family_id = ? AND revoked_at IS NULL", (now, row["family_id"]))
        raise AuthError("refresh token reuse")
    if row["expires_at"] <= now:
        raise AuthError("expired refresh token")
    # compare-and-set: if a parallel request consumed this token between the read and here, this is reuse
    if db.x("UPDATE refresh_tokens SET used_at = ? WHERE id = ? AND used_at IS NULL AND revoked_at IS NULL", (now, row["id"])) != 1:
        db.x("UPDATE refresh_tokens SET revoked_at = ? WHERE family_id = ? AND revoked_at IS NULL", (now, row["family_id"]))
        raise AuthError("refresh token reuse")
    user = db.one("SELECT id, tenant_id, role, active FROM users WHERE id = ?", (row["user_id"],))
    if not user or not user["active"]:
        raise AuthError("inactive user")
    return issue_refresh_token(s, db, user["id"], row["family_id"], now), user


def revoke_family_of(db: DB, raw: str, now: int | None = None) -> None:
    row = db.one("SELECT family_id FROM refresh_tokens WHERE token_hash = ?", (sha256_hex(raw),))
    if row:
        db.x("UPDATE refresh_tokens SET revoked_at = ? WHERE family_id = ? AND revoked_at IS NULL", (int(now or time.time()), row["family_id"]))


# ---------------- API tokens (for MCP / automation) ----------------
def principal_from_api_token(db: DB, raw: str, now: int | None = None) -> Principal:
    import json
    now = int(now or time.time())
    if not isinstance(raw, str) or not raw.startswith("ast_") or len(raw) > 200:
        raise AuthError("bad api token")
    row = db.one("SELECT id, tenant_id, role, allowed_tools, expires_at, revoked_at FROM api_tokens WHERE token_hash = ?", (sha256_hex(raw),))
    if not row or row["revoked_at"] is not None or row["expires_at"] <= now:
        raise AuthError("bad api token")
    return Principal("api_token", row["id"], row["tenant_id"], row["role"], frozenset(json.loads(row["allowed_tools"])))
