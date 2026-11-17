"""HTTP hardening and injection probes (an automated slice of the pen-test plan)."""
import re

import pytest
import requests

from conftest import ORIGIN

INJECTIONS = ["' OR '1'='1", "1; DROP TABLE alerts;--", "../../etc/passwd", "%00", "<script>alert(1)</script>",
              "{{7*7}}", "${jndi:ldap://x}", "admin'--", "‮evil", "A" * 5000]


def test_security_headers_everywhere(env):
    for r in (env.get("/healthz"), env.get("/api/v1/me"), env.get("/nope")):
        h = r.headers
        assert h["x-content-type-options"] == "nosniff"
        assert "frame-ancestors 'none'" in h["content-security-policy"]
        assert h["strict-transport-security"].startswith("max-age=")
        assert h["referrer-policy"] == "no-referrer"
        assert "camera=()" in h["permissions-policy"]
        assert h["cache-control"] == "no-store"
        assert "x-request-id" in h
        assert "server" not in {k.lower() for k in h}


def test_errors_are_json_without_internals(env):
    r = env.get("/does/not/exist")
    assert r.status_code == 404 and r.json()["error"] == "not_found"
    r = env.post("/api/v1/auth/login", data=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422 and "Traceback" not in r.text


def test_body_size_cap(env):
    r = env.post("/api/v1/auth/login", data=b"{" + b" " * (300 * 1024) + b"}", headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_json_depth_limit(env):
    deep = "[" * 50 + "]" * 50
    r = env.post("/api/v1/auth/login", data=('{"email":' + deep + "}").encode(), headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_content_type_enforced(env):
    r = env.post("/api/v1/auth/login", data=b'{"email":"a@b.c","password":"x"}', headers={"Content-Type": "text/plain"})
    assert r.status_code == 422


def test_cors_allowlist(env):
    ok = requests.options(env.base + "/api/v1/alerts", headers={"Origin": ORIGIN, "Access-Control-Request-Method": "GET"})
    assert ok.status_code == 204 and ok.headers["access-control-allow-origin"] == ORIGIN
    assert "access-control-allow-credentials" not in ok.headers
    bad = requests.options(env.base + "/api/v1/alerts", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert bad.status_code == 403 and "access-control-allow-origin" not in bad.headers
    g = env.get("/healthz", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in g.headers


def test_request_id_sanitized(env):
    r = env.get("/healthz", headers={"X-Request-ID": "abc<script>;x=1"})
    assert r.headers["x-request-id"] != "abc<script>;x=1" and re.match(r"^[A-Za-z0-9-]+$", r.headers["x-request-id"])
    ok = env.get("/healthz", headers={"X-Request-ID": "trace-1234abcd"})
    assert ok.headers["x-request-id"] == "trace-1234abcd"


@pytest.mark.parametrize("payload", INJECTIONS)
def test_injection_in_path_and_query(env, payload):
    h = env.h(env.A, "tenant_admin")
    for path, params in [("/api/v1/alerts", {"status": payload}), ("/api/v1/alerts", {"site_id": payload}),
                         ("/api/v1/alerts", {"cursor": payload}), ("/api/v1/alerts", {"limit": payload}),
                         ("/api/v1/tracks", {"site_id": payload}), ("/api/v1/tracks", {"site_id": env.A.site, "since": payload})]:
        r = env.get(path, params=params, headers=h)
        assert r.status_code in (200, 404, 422), (path, params, r.status_code)
        assert "sqlite" not in r.text.lower() and "Traceback" not in r.text
    r = requests.get(env.base + "/api/v1/alerts/" + requests.utils.quote(payload, safe=""), headers=h)
    assert r.status_code in (404, 422)


@pytest.mark.parametrize("payload", INJECTIONS)
def test_injection_in_login(env, payload):
    r = env.post("/api/v1/auth/login", json={"email": payload, "password": payload})
    assert r.status_code in (401, 422, 429)


def test_database_intact_after_probes(env):
    db = env.db()
    assert db.one("SELECT COUNT(*) AS n FROM tenants")["n"] >= 2
    assert db.one("SELECT COUNT(*) AS n FROM users")["n"] >= 10
    db.close()


def test_pagination_bounded(env, new_alert):
    for _ in range(3):
        new_alert()
    r = env.get("/api/v1/alerts", params={"limit": 100000}, headers=env.h(env.A, "viewer")).json()
    assert len(r["items"]) <= 100
    r = env.get("/api/v1/alerts", params={"limit": 2}, headers=env.h(env.A, "viewer")).json()
    assert len(r["items"]) == 2 and r["next_cursor"]
    r2 = env.get("/api/v1/alerts", params={"limit": 2, "cursor": r["next_cursor"]}, headers=env.h(env.A, "viewer")).json()
    assert {a["id"] for a in r2["items"]}.isdisjoint({a["id"] for a in r["items"]})


def test_method_not_allowed(env):
    assert env.get("/api/v1/auth/login").status_code == 405
    assert requests.delete(env.base + "/api/v1/alerts", headers=env.h(env.A, "tenant_admin")).status_code == 405
