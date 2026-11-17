"""Browser session mode: the refresh token lives only in a __Host- HttpOnly cookie, and every cookie-authenticated
request must pass an Origin allowlist check plus a double-submit CSRF token bound to the refresh token."""
from __future__ import annotations

import time

import pytest
import requests

from airspace import admin
from airspace.auth import Principal, totp_now
from conftest import ORIGIN, PW, totp_secret_of

RT, CSRF = "__Host-airspace_rt", "__Host-airspace_csrf"


def _set_cookies(r) -> dict[str, str]:
    out = {}
    for line in r.raw.headers.getlist("set-cookie"):
        name, rest = line.split("=", 1)
        out[name] = line
    return out


def _value(line: str) -> str:
    return line.split("=", 1)[1].split(";", 1)[0]


@pytest.fixture
def browser_session(env):
    """Log a fresh operator in through cookie mode; returns (rt, csrf, login_response)."""
    db = env.db()
    adm = Principal("user", env.A.users["tenant_admin"]["id"], env.A.id, "tenant_admin")
    email = f"web{time.time_ns()}@a.example"
    u = admin.create_user(env.s, db, adm, {"email": email, "role": "operator", "password": PW})
    sec = totp_secret_of(env.s, db, u["id"])
    db.close()
    r = env.post("/api/v1/auth/login", json={"email": email, "password": PW, "totp": totp_now(sec)},
                 headers={"X-Session-Mode": "cookie", "Origin": ORIGIN})
    assert r.status_code == 200, r.text
    c = _set_cookies(r)
    return _value(c[RT]), _value(c[CSRF]), r


def _cookie_post(env, path, rt, csrf_cookie, csrf_header=None, origin=ORIGIN, body=b"{}"):
    hdrs = {"X-Session-Mode": "cookie", "Content-Type": "application/json", "Cookie": f"{RT}={rt}; {CSRF}={csrf_cookie}"}
    if origin is not None:
        hdrs["Origin"] = origin
    if csrf_header is not None:
        hdrs["X-CSRF-Token"] = csrf_header
    return requests.post(env.base + path, data=body, headers=hdrs, timeout=10)


def test_login_sets_hardened_cookies_and_hides_refresh_token(browser_session):
    rt, csrf, r = browser_session
    body = r.json()
    assert "refresh_token" not in body and body["csrf_token"] == csrf and body["access_token"]
    c = _set_cookies(r)
    rt_attrs = c[RT].lower()
    for attr in ("secure", "httponly", "samesite=strict", "path=/"):
        assert attr in rt_attrs, attr
    assert "domain=" not in rt_attrs                     # __Host- prefix: host-only cookie
    assert "secure" in c[CSRF].lower() and "httponly" not in c[CSRF].lower()   # readable for double submit


def test_cookie_refresh_rotates(env, browser_session):
    rt, csrf, _ = browser_session
    r = _cookie_post(env, "/api/v1/auth/refresh", rt, csrf, csrf)
    assert r.status_code == 200, r.text
    new = _set_cookies(r)
    assert _value(new[RT]) != rt and "refresh_token" not in r.json()
    # the old refresh token is now spent: replaying it trips reuse detection
    assert _cookie_post(env, "/api/v1/auth/refresh", rt, csrf, csrf).status_code == 401


@pytest.mark.parametrize("origin", [None, "https://evil.example", "null", ORIGIN + ".evil.example"])
def test_cookie_refresh_requires_allowed_origin(env, browser_session, origin):
    rt, csrf, _ = browser_session
    assert _cookie_post(env, "/api/v1/auth/refresh", rt, csrf, csrf, origin=origin).status_code == 403


def test_cookie_refresh_requires_csrf_header(env, browser_session):
    rt, csrf, _ = browser_session
    assert _cookie_post(env, "/api/v1/auth/refresh", rt, csrf, None).status_code == 401


def test_csrf_must_match_cookie_and_session(env, browser_session):
    rt, csrf, _ = browser_session
    assert _cookie_post(env, "/api/v1/auth/refresh", rt, csrf, "0" * 64).status_code == 403
    # attacker who can plant a cookie but doesn't know the session-bound value
    assert _cookie_post(env, "/api/v1/auth/refresh", rt, "f" * 64, "f" * 64).status_code == 403


def test_cookie_login_from_foreign_origin_refused(env):
    r = env.post("/api/v1/auth/login", json={"email": "x@a.example", "password": PW},
                 headers={"X-Session-Mode": "cookie", "Origin": "https://evil.example"})
    assert r.status_code == 403 and not r.raw.headers.getlist("set-cookie")


def test_cookie_logout_revokes_and_clears(env, browser_session):
    rt, csrf, _ = browser_session
    r = _cookie_post(env, "/api/v1/auth/logout", rt, csrf, csrf)
    assert r.status_code == 200
    c = _set_cookies(r)
    assert "max-age=0" in c[RT].lower() and "max-age=0" in c[CSRF].lower()
    assert _cookie_post(env, "/api/v1/auth/refresh", rt, csrf, csrf).status_code == 401


def test_cookie_mode_rejects_body_tokens(env, browser_session):
    rt, csrf, _ = browser_session
    r = _cookie_post(env, "/api/v1/auth/refresh", rt, csrf, csrf, body=b'{"refresh_token":"rt_xxxxxxxxxxxxxxxxxxxxxxxx"}')
    assert r.status_code == 422
