"""Trim API responses. Two layers:
  1. the whole test suite runs with strict_outputs=True (conftest), so ANY response with a field outside its
     allowlist fails the request that produced it;
  2. this module crawls every readable route as several roles and checks the JSON for secret-shaped keys and values.
"""
from __future__ import annotations

import re

import pytest

from airspace.outputs import FORBIDDEN, OutputViolation, trim

SECRET_VALUE = re.compile(r"^(rt_|eg_|scrypt\$|enc:v1:|-----BEGIN)|LS0tLS1CRUdJTi")
TOKEN_FIELDS = {"access_token", "refresh_token", "token", "csrf_token", "code", "grant_for_requester", "totp_provisioning_uri"}


def _scan(obj, path="$"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert k not in FORBIDDEN, f"{path}.{k}"
            if isinstance(v, str) and k not in TOKEN_FIELDS:
                assert not SECRET_VALUE.search(v), f"{path}.{k} looks like a secret"
            _scan(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _scan(v, f"{path}[{i}]")


def test_strict_mode_rejects_unexpected_and_forbidden_fields():
    with pytest.raises(OutputViolation):
        trim("me", {"id": "u", "tenant_id": "t", "role": "viewer", "email": "x@y"}, strict=True)
    with pytest.raises(OutputViolation):
        trim("sites", {"items": [{"id": "s", "name": "n", "pw_hash": "scrypt$..."}]}, strict=True)


def test_lenient_mode_drops_and_reports():
    body, dropped = trim("alert_list", {"items": [{"id": "a", "status": "open", "tenant_id": "tn_x"}], "next_cursor": None})
    assert body == {"items": [{"id": "a", "status": "open"}], "next_cursor": None}
    assert dropped == ["alert_list.items[].tenant_id"]


def test_crawl_readable_routes(env, stored_evidence):
    aid, eid = stored_evidence
    site = env.A.site
    gets = ["/api/v1/me", "/api/v1/sites", f"/api/v1/sites/{site}/sensors", "/api/v1/alerts", f"/api/v1/alerts/{aid}",
            f"/api/v1/tracks?site_id={site}", f"/api/v1/evidence/{eid}", "/api/v1/approvals", "/api/v1/audit",
            "/api/v1/audit/verify"]
    for role in ("viewer", "operator", "supervisor", "tenant_admin"):
        for path in gets:
            r = env.get(path, headers=env.h(env.A, role))
            assert r.status_code in (200, 403), (role, path, r.status_code)
            if r.status_code == 200:
                _scan(r.json())


def test_alert_rows_never_expose_tenant_or_internal_ids(env, new_alert):
    new_alert()
    items = env.get("/api/v1/alerts", headers=env.h(env.A, "viewer")).json()["items"]
    assert items and all("tenant_id" not in i and "seq" not in i for i in items)
