import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from airspace import admin
from airspace.auth import (Principal, hash_password, issue_access_token, totp_now, verify_password, verify_totp)
from conftest import PW


# ---------- primitives ----------
def test_password_hash_roundtrip_and_pepper():
    pep = b"p" * 32
    h = hash_password(PW, pep)
    assert h.startswith("scrypt$") and PW not in h
    assert verify_password(PW, h, pep)
    assert not verify_password(PW + "x", h, pep)
    assert not verify_password(PW, h, b"q" * 32)          # stolen DB without the pepper is useless
    assert hash_password(PW, pep) != h                     # per-user random salt


def test_password_policy():
    with pytest.raises(ValueError):
        hash_password("short", b"p" * 32)


def test_totp_rfc6238_vector():
    # RFC 6238 test secret "12345678901234567890" (base32 GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ), T=59 -> 94287082 (8 digits) -> 287082
    assert totp_now("GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ", 59) == "287082"


def test_totp_replay_blocked():
    sec = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"
    step = verify_totp(sec, totp_now(sec, 1000), 0, 1000)
    assert step is not None
    assert verify_totp(sec, totp_now(sec, 1000), step, 1000) is None


# ---------- login flow ----------
def _login(env, email, password=PW, totp=None, solve=True):
    """Log in; when the server answers 428 with a proof-of-work challenge, solve it and retry once
    (what a legitimate browser does). solve=False shows the raw 428."""
    body = {"email": email, "password": password}
    if totp:
        body["totp"] = totp
    r = env.post("/api/v1/auth/login", json=body)
    if r.status_code == 428 and solve:
        from airspace.challenge import solve_pow
        ch = r.json()["challenge"]
        body["challenge"] = {"kind": "pow", "token": ch["token"], "solution": solve_pow(ch["token"], ch["difficulty"])}
        r = env.post("/api/v1/auth/login", json=body)
    return r


@pytest.fixture
def fresh_user(env):
    """A brand-new operator per test so TOTP steps and lockout counters don't collide."""
    db = env.db()
    adm = Principal("user", env.A.users["tenant_admin"]["id"], env.A.id, "tenant_admin")
    email = f"op{time.time_ns()}@a.example"
    r = admin.create_user(env.s, db, adm, {"email": email, "role": "operator", "password": PW})
    from conftest import totp_secret_of
    sec = totp_secret_of(env.s, db, r["id"])
    db.close()
    return {"id": r["id"], "email": email, "totp": sec}


def test_login_requires_mfa_for_operator(env, fresh_user):
    r = _login(env, fresh_user["email"])
    assert r.status_code == 401
    r = _login(env, fresh_user["email"], totp=totp_now(fresh_user["totp"]))
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["token_type"] == "Bearer" and j["expires_in"] <= 600 and j["refresh_token"].startswith("rt_")
    me = env.get("/api/v1/me", headers={"Authorization": "Bearer " + j["access_token"]})
    assert me.status_code == 200 and me.json()["role"] == "operator"


def test_totp_code_cannot_be_reused(env, fresh_user):
    code = totp_now(fresh_user["totp"])
    assert _login(env, fresh_user["email"], totp=code).status_code == 200
    assert _login(env, fresh_user["email"], totp=code).status_code == 401


def test_errors_are_generic(env, fresh_user):
    a = _login(env, fresh_user["email"], password="wrong password!!", totp=totp_now(fresh_user["totp"]))
    b = _login(env, "nobody@nowhere.example", password="wrong password!!")
    assert a.status_code == b.status_code == 401
    assert a.json()["error"] == b.json()["error"]


def test_lockout_after_failures(env, fresh_user):
    # a bot that pays the proof-of-work on every attempt still hits the account lockout
    from airspace.ratelimit import LIMITER
    for _ in range(5):
        LIMITER.reset()                       # this test is about lockout; rate limits have their own tests
        assert _login(env, fresh_user["email"], password="wrong password!!").status_code == 401
    LIMITER.reset()
    # correct credentials are now refused (locked), with the same generic error
    r = _login(env, fresh_user["email"], totp=totp_now(fresh_user["totp"]))
    assert r.status_code == 401
    db = env.db()
    row = db.one("SELECT failed_count, locked_until FROM users WHERE id = ?", (fresh_user["id"],))
    db.close()
    assert row["failed_count"] >= 5 and row["locked_until"] > time.time()


def test_login_rate_limited_per_ip(env):
    codes = [_login(env, f"x{i}@nowhere.example", password="wrong password!!").status_code for i in range(15)]
    assert 429 in codes


# ---------- token tampering ----------
def _me(env, tok):
    return env.get("/api/v1/me", headers={"Authorization": "Bearer " + tok}).status_code


def test_valid_token_accepted(env):
    assert _me(env, env.token(env.A, "viewer")) == 200


def test_alg_none_rejected(env):
    u = env.A.users["viewer"]
    claims = {"iss": env.s.issuer, "aud": env.s.audience, "sub": u["id"], "tid": env.A.id, "role": "tenant_admin",
              "iat": int(time.time()), "nbf": int(time.time()), "exp": int(time.time()) + 300, "jti": "x"}
    tok = jwt.encode(claims, None, algorithm="none", headers={"kid": "k1"})
    assert _me(env, tok) == 401


def test_hs256_with_public_key_rejected(env):
    from cryptography.hazmat.primitives import serialization
    pub = env.s.jwt_keys["k1"].public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    u = env.A.users["viewer"]
    claims = {"iss": env.s.issuer, "aud": env.s.audience, "sub": u["id"], "tid": env.A.id, "role": "viewer",
              "iat": int(time.time()), "nbf": int(time.time()), "exp": int(time.time()) + 300, "jti": "x"}
    import hmac, hashlib, base64, json as js
    def b64(b): return base64.urlsafe_b64encode(b).rstrip(b"=")
    head = b64(js.dumps({"alg": "HS256", "kid": "k1", "typ": "JWT"}).encode())
    body = b64(js.dumps(claims).encode())
    sig = b64(hmac.new(pub, head + b"." + body, hashlib.sha256).digest())
    assert _me(env, (head + b"." + body + b"." + sig).decode()) == 401


def test_foreign_key_and_role_escalation_rejected(env):
    u = env.A.users["viewer"]
    claims = {"iss": env.s.issuer, "aud": env.s.audience, "sub": u["id"], "tid": env.A.id, "role": "tenant_admin",
              "iat": int(time.time()), "nbf": int(time.time()), "exp": int(time.time()) + 300, "jti": "x"}
    attacker = Ed25519PrivateKey.generate()
    assert _me(env, jwt.encode(claims, attacker, algorithm="EdDSA", headers={"kid": "k1"})) == 401
    # correctly signed but claiming a role the user doesn't have (e.g. issued before a demotion): DB re-check rejects
    assert _me(env, jwt.encode(claims, env.s.jwt_keys["k1"], algorithm="EdDSA", headers={"kid": "k1"})) == 401


def test_expired_wrong_aud_unknown_kid(env):
    u = env.A.users["viewer"]
    base = {"iss": env.s.issuer, "aud": env.s.audience, "sub": u["id"], "tid": env.A.id, "role": "viewer",
            "iat": int(time.time()) - 1000, "nbf": int(time.time()) - 1000, "jti": "x"}
    k = env.s.jwt_keys["k1"]
    assert _me(env, jwt.encode({**base, "exp": int(time.time()) - 100}, k, algorithm="EdDSA", headers={"kid": "k1"})) == 401
    assert _me(env, jwt.encode({**base, "exp": int(time.time()) + 300, "aud": "other"}, k, algorithm="EdDSA", headers={"kid": "k1"})) == 401
    assert _me(env, jwt.encode({**base, "exp": int(time.time()) + 300}, k, algorithm="EdDSA", headers={"kid": "nope"})) == 401
    missing = {k2: v for k2, v in base.items() if k2 != "jti"}
    assert _me(env, jwt.encode({**missing, "exp": int(time.time()) + 300}, k, algorithm="EdDSA", headers={"kid": "k1"})) == 401


def test_key_rotation_previous_kid_still_verifies(env):
    u = env.A.users["viewer"]
    now = int(time.time())
    claims = {"iss": env.s.issuer, "aud": env.s.audience, "sub": u["id"], "tid": env.A.id, "role": "viewer",
              "iat": now, "nbf": now, "exp": now + 300, "jti": "y"}
    assert _me(env, jwt.encode(claims, env.s.jwt_keys["k0"], algorithm="EdDSA", headers={"kid": "k0"})) == 200


def test_malformed_auth_headers(env):
    for h in ["", "Bearer", "Basic abc", "Bearer " + "a" * 5000, "Bearer a.b.c"]:
        assert env.get("/api/v1/me", headers={"Authorization": h}).status_code in (401, 413)


# ---------- refresh rotation + reuse detection ----------
def test_refresh_rotation_and_reuse_detection(env, fresh_user):
    r = _login(env, fresh_user["email"], totp=totp_now(fresh_user["totp"])).json()
    rt1 = r["refresh_token"]
    r2 = env.post("/api/v1/auth/refresh", json={"refresh_token": rt1})
    assert r2.status_code == 200
    rt2 = r2.json()["refresh_token"]
    assert rt2 != rt1
    # replaying the consumed token = theft signal: whole family revoked, including the newer token
    assert env.post("/api/v1/auth/refresh", json={"refresh_token": rt1}).status_code == 401
    assert env.post("/api/v1/auth/refresh", json={"refresh_token": rt2}).status_code == 401


def test_logout_revokes(env, fresh_user):
    rt = _login(env, fresh_user["email"], totp=totp_now(fresh_user["totp"])).json()["refresh_token"]
    assert env.post("/api/v1/auth/logout", json={"refresh_token": rt}).status_code == 200
    assert env.post("/api/v1/auth/refresh", json={"refresh_token": rt}).status_code == 401


def test_disabled_user_token_dies_immediately(env, fresh_user):
    tok = issue_access_token(env.s, fresh_user["id"], env.A.id, "operator")
    assert _me(env, tok) == 200
    db = env.db()
    db.x("UPDATE users SET active = 0 WHERE id = ?", (fresh_user["id"],))
    db.close()
    assert _me(env, tok) == 401
