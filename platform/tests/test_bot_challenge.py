"""Bot protection: after repeated failed logins a challenge is mandatory, proof-of-work challenges are signed,
IP-bound, expiring and single-use, and the hosted-CAPTCHA adapter fails closed."""
from __future__ import annotations

import io
import json
import time

import pytest

from airspace import admin, challenge
from airspace.auth import Principal, totp_now
from airspace.challenge import HostedCaptchaVerifier, issue_pow, solve_pow, verify_pow
from conftest import PW, totp_secret_of


@pytest.fixture
def user(env):
    db = env.db()
    adm = Principal("user", env.A.users["tenant_admin"]["id"], env.A.id, "tenant_admin")
    email = f"bot{time.time_ns()}@a.example"
    u = admin.create_user(env.s, db, adm, {"email": email, "role": "operator", "password": PW})
    sec = totp_secret_of(env.s, db, u["id"])
    db.close()
    return {"email": email, "totp": sec}


def _fail(env, email, n):
    for _ in range(n):
        assert env.post("/api/v1/auth/login", json={"email": email, "password": "wrong password!!"}).status_code == 401


def test_challenge_required_after_failures(env, user):
    _fail(env, user["email"], env.s.challenge_after)
    r = env.post("/api/v1/auth/login", json={"email": user["email"], "password": PW, "totp": totp_now(user["totp"])})
    assert r.status_code == 428
    ch = r.json()["challenge"]
    assert ch["kind"] == "pow" and ch["difficulty"] == env.s.pow_difficulty and ch["token"]


def test_correct_password_without_challenge_still_refused(env, user):
    """The gate runs before the password check: a stolen password alone doesn't get through a challenged account."""
    _fail(env, user["email"], env.s.challenge_after)
    r = env.post("/api/v1/auth/login", json={"email": user["email"], "password": PW, "totp": totp_now(user["totp"])})
    assert r.status_code == 428 and "access_token" not in r.text


def test_solved_challenge_admits_and_is_single_use(env, user):
    _fail(env, user["email"], env.s.challenge_after)
    ch = env.post("/api/v1/auth/login", json={"email": user["email"], "password": PW}).json()["challenge"]
    sol = {"kind": "pow", "token": ch["token"], "solution": solve_pow(ch["token"], ch["difficulty"])}
    r = env.post("/api/v1/auth/login", json={"email": user["email"], "password": PW, "totp": totp_now(user["totp"]), "challenge": sol})
    assert r.status_code == 200, r.text
    # replaying the same solved challenge once the account needs a challenge again
    from airspace.ratelimit import LIMITER
    LIMITER.reset()
    for _ in range(env.s.challenge_after):
        challenge.FAILURES.record("acct:" + user["email"])
    r2 = env.post("/api/v1/auth/login", json={"email": user["email"], "password": PW, "challenge": sol})
    assert r2.status_code == 428


def test_ip_threshold_triggers_challenge_for_any_account(env):
    for i in range(env.s.challenge_after):
        env.post("/api/v1/auth/login", json={"email": f"spray{i}@nowhere.example", "password": "wrong password!!"})
    r = env.post("/api/v1/auth/login", json={"email": "another@nowhere.example", "password": "wrong password!!"})
    assert r.status_code == 428            # password spraying across accounts from one source


def test_pow_rejects_tampering_expiry_and_foreign_ip(env):
    db = env.db()
    try:
        ch = issue_pow(env.s, "203.0.113.5")
        sol = solve_pow(ch["token"], ch["difficulty"])
        assert not verify_pow(env.s, db, ch["token"], sol, "198.51.100.9")                 # bound to the issuing IP
        assert not verify_pow(env.s, db, ch["token"], sol, "203.0.113.5", now=ch["expires_at"] + 1)   # expired
        raw, sig = ch["token"].split(".")
        assert not verify_pow(env.s, db, raw + "." + sig[:-2] + "AA", sol, "203.0.113.5")   # forged signature
        bad = format(int(sol, 16) + 1, "x")
        if challenge.leading_zero_bits(__import__("hashlib").sha256(f"{ch['token']}:{bad}".encode()).digest()) < ch["difficulty"]:
            assert not verify_pow(env.s, db, ch["token"], bad, "203.0.113.5")                 # wrong solution
        assert verify_pow(env.s, db, ch["token"], sol, "203.0.113.5")
        assert not verify_pow(env.s, db, ch["token"], sol, "203.0.113.5")                    # single use
    finally:
        db.close()


def test_pow_difficulty_cannot_be_lowered_by_client(env):
    """A token issued at a lower difficulty (e.g. by an older config) is refused once the server requires more."""
    db = env.db()
    try:
        old = env.s.pow_difficulty
        ch = issue_pow(env.s, "203.0.113.7")
        env.s.pow_difficulty = old + 4
        assert not verify_pow(env.s, db, ch["token"], solve_pow(ch["token"], old), "203.0.113.7")
    finally:
        env.s.pow_difficulty = old
        db.close()


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.mark.parametrize("payload,expected", [({"success": True}, True), ({"success": False}, False), ({}, False)])
def test_hosted_captcha_adapter(env, monkeypatch, payload, expected):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"], seen["timeout"] = req.full_url, timeout
        return _Resp(json.dumps(payload).encode())
    monkeypatch.setattr(challenge.urllib.request, "urlopen", fake_urlopen)
    v = HostedCaptchaVerifier("turnstile")
    assert v.verify(env.s, None, {"token": "tok"}, "203.0.113.1") is expected
    assert seen["url"].startswith("https://challenges.cloudflare.com/") and seen["timeout"] <= 5


def test_hosted_captcha_fails_closed(env, monkeypatch):
    def boom(req, timeout):
        raise OSError("provider down")
    monkeypatch.setattr(challenge.urllib.request, "urlopen", boom)
    assert HostedCaptchaVerifier("hcaptcha").verify(env.s, None, {"token": "tok"}, "203.0.113.1") is False
    assert HostedCaptchaVerifier("hcaptcha").verify(env.s, None, {"token": ""}, "203.0.113.1") is False
