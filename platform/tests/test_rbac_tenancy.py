"""RBAC matrix and cross-tenant isolation (IDOR) tests."""
import sqlite3

import pytest

from airspace.rbac import POLICY, ROLES


def test_policy_is_deny_by_default():
    from airspace.auth import Principal
    from airspace.rbac import allowed
    p = Principal("user", "u", "t", "tenant_admin")
    assert not allowed(p, "made.up.permission")
    assert all(set(v) <= set(ROLES) for v in POLICY.values())


@pytest.mark.parametrize("role,expect", [("viewer", 403), ("operator", 200), ("supervisor", 200), ("tenant_admin", 200)])
def test_ack_matrix(env, new_alert, role, expect):
    aid = new_alert()
    r = env.post(f"/api/v1/alerts/{aid}/ack", headers=env.h(env.A, role))
    assert r.status_code == expect, r.text


@pytest.mark.parametrize("path,method,allowed", [
    ("/api/v1/users", "post", {"tenant_admin"}),
    ("/api/v1/devices", "post", {"tenant_admin"}),
    ("/api/v1/api-tokens", "post", {"tenant_admin"}),
    ("/api/v1/audit", "get", {"tenant_admin"}),
    ("/api/v1/approvals", "get", {"supervisor", "tenant_admin"}),
])
def test_admin_matrix(env, path, method, allowed):
    for role in ("viewer", "operator", "supervisor", "tenant_admin"):
        kw = {"headers": env.h(env.A, role)}
        if method == "post":
            kw["json"] = {}
        r = getattr(env, method)(path, **kw)
        if role in allowed:
            assert r.status_code != 403, (role, path)
        else:
            assert r.status_code == 403, (role, path, r.status_code)


def test_viewer_cannot_read_evidence(env, new_alert):
    assert env.get("/api/v1/evidence/abcdef123456", headers=env.h(env.A, "viewer")).status_code == 403


# ---------- tenant isolation ----------
def test_cross_tenant_alert_is_invisible(env, new_alert):
    aid = new_alert(env.A)
    hb = env.h(env.B, "tenant_admin")
    assert env.get(f"/api/v1/alerts/{aid}", headers=hb).status_code == 404
    assert env.post(f"/api/v1/alerts/{aid}/ack", headers=hb).status_code == 404
    assert env.post(f"/api/v1/alerts/{aid}/resolve", headers=hb, json={"note": "x"}).status_code == 404
    assert env.post(f"/api/v1/alerts/{aid}/legal-hold", headers=hb, json={"hold": True}).status_code == 404
    assert env.post(f"/api/v1/alerts/{aid}/mcp-confirmation", headers=hb).status_code == 404
    ids = [a["id"] for a in env.get("/api/v1/alerts", headers=hb).json()["items"]]
    assert aid not in ids


def test_cross_tenant_site_sensor_track(env):
    hb = env.h(env.B, "tenant_admin")
    assert env.get(f"/api/v1/sites/{env.A.site}/sensors", headers=hb).status_code == 404
    assert env.get("/api/v1/tracks", params={"site_id": env.A.site}, headers=hb).status_code == 404
    assert env.A.site not in [s["id"] for s in env.get("/api/v1/sites", headers=hb).json()["items"]]


def test_cross_tenant_device_registration_blocked(env):
    r = env.post("/api/v1/devices", headers=env.h(env.B, "tenant_admin"), json={"site_id": env.A.site, "public_key": "A" * 43})
    assert r.status_code == 404


def test_cross_tenant_approval_target_blocked(env):
    r = env.post("/api/v1/approvals", headers=env.h(env.B, "supervisor"),
                 json={"action": "disable_sensor", "sensor_id": env.A.sensors["radar"], "reason": "x"})
    assert r.status_code == 404


def test_tenant_id_in_body_is_ignored(env):
    # extra fields are rejected outright; tenant never comes from input
    r = env.post("/api/v1/users", headers=env.h(env.B, "tenant_admin"),
                 json={"email": "evil@b.example", "role": "viewer", "password": "x" * 12, "tenant_id": env.A.id})
    assert r.status_code == 422


def test_composite_fk_blocks_cross_tenant_rows(env):
    db = env.db()
    with pytest.raises(sqlite3.IntegrityError):
        db.x("INSERT INTO sensors (id, tenant_id, site_id, kind) VALUES (?, ?, ?, ?)", ("sen_evil01", env.B.id, env.A.site, "radar"))
    db.close()


def test_platform_auditor_read_only_cross_tenant(env):
    from airspace import admin
    from airspace.auth import issue_access_token
    db = env.db()
    a = admin.create_platform_auditor(env.s, db, "auditor@platform.example", "ledger quartz meadow 71")
    db.close()
    h = {"Authorization": "Bearer " + issue_access_token(env.s, a["user_id"], None, "platform_auditor")}
    assert env.get("/api/v1/audit/verify", params={"tenant_id": env.A.id}, headers=h).json()["ok"] is True
    assert env.get("/api/v1/alerts", headers=h).status_code == 403            # cannot read operational data
    assert env.post("/api/v1/users", headers=h, json={}).status_code == 403
    assert env.get("/api/v1/audit/verify", params={"tenant_id": "x' OR 1=1 --"}, headers=h).status_code == 422
