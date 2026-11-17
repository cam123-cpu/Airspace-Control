"""Inventory tests: they enumerate what exists, so a new route or input model can't ship without the
protection every other one has.

  * every route requires a credential unless it is on the explicit public list
  * the list in this file matches the app exactly (adding a route forces a decision here)
  * every pydantic input model forbids unknown fields (blocks mass assignment / field tampering)
  * every route has an output allowlist in airspace/outputs.py
"""
from __future__ import annotations

import inspect
import pkgutil
import re

import requests
from pydantic import BaseModel
from starlette.routing import Route

import airspace
from airspace.app import create_app
from airspace.outputs import FORBIDDEN, SPECS

PUBLIC = {
    ("GET", "/healthz"),
    ("POST", "/api/v1/auth/challenge"),
    ("POST", "/api/v1/auth/login"),
    ("POST", "/api/v1/auth/refresh"),
    ("POST", "/api/v1/auth/logout"),
}
DEVICE_SIGNED = {("POST", "/ingest/v1/telemetry"), ("POST", "/ingest/v1/evidence")}
USER_AUTH = {
    ("GET", "/api/v1/me"), ("GET", "/api/v1/sites"), ("GET", "/api/v1/sites/{site_id}/sensors"),
    ("GET", "/api/v1/alerts"), ("GET", "/api/v1/alerts/{alert_id}"), ("POST", "/api/v1/alerts/{alert_id}/ack"),
    ("POST", "/api/v1/alerts/{alert_id}/resolve"), ("POST", "/api/v1/alerts/{alert_id}/legal-hold"),
    ("POST", "/api/v1/alerts/{alert_id}/mcp-confirmation"), ("GET", "/api/v1/tracks"),
    ("GET", "/api/v1/evidence/{evidence_id}"), ("GET", "/api/v1/evidence/{evidence_id}/download"),
    ("POST", "/api/v1/approvals"), ("GET", "/api/v1/approvals"), ("POST", "/api/v1/approvals/{approval_id}/approve"),
    ("POST", "/api/v1/approvals/{approval_id}/reject"), ("POST", "/api/v1/users"), ("POST", "/api/v1/devices"),
    ("POST", "/api/v1/api-tokens"), ("POST", "/api/v1/api-tokens/{token_id}/revoke"), ("GET", "/api/v1/audit"),
    ("GET", "/api/v1/audit/verify"),
}


def _routes(env) -> set[tuple[str, str]]:
    out = set()
    for r in create_app(env.s).routes:
        if isinstance(r, Route):
            for m in r.methods or ():
                if m != "HEAD":
                    out.add((m, r.path))
    return out


def test_route_inventory_is_complete(env):
    assert _routes(env) == PUBLIC | DEVICE_SIGNED | USER_AUTH


def _url(env, path: str) -> str:
    return env.base + re.sub(r"\{[a-z_]+\}", "abcdef123456", path)


def test_every_protected_route_rejects_anonymous_requests(env):
    for method, path in sorted(USER_AUTH | DEVICE_SIGNED):
        r = requests.request(method, _url(env, path), json={}, timeout=10)
        assert r.status_code == 401, (method, path, r.status_code)


def test_every_protected_route_rejects_garbage_tokens(env):
    for method, path in sorted(USER_AUTH):
        for tok in ("Bearer x", "Bearer " + "a" * 5000, "Basic dXNlcjpwYXNz", "bearer " + env.token(env.A, "viewer")):
            r = requests.request(method, _url(env, path), json={}, headers={"Authorization": tok}, timeout=10)
            assert r.status_code == 401, (method, path, tok[:12], r.status_code)


def test_public_routes_return_nothing_sensitive_without_input(env):
    for method, path in sorted(PUBLIC):
        r = requests.request(method, _url(env, path), json={}, timeout=10)
        if path in ("/healthz", "/api/v1/auth/challenge"):
            assert r.status_code == 200
        else:
            assert r.status_code in (401, 422), (path, r.status_code)
            assert "token" not in r.text.lower()


def _models() -> list[type[BaseModel]]:
    found = []
    for mod in pkgutil.iter_modules(airspace.__path__):
        if mod.name == "asgi":                 # builds an app from the environment at import time
            continue
        m = __import__(f"airspace.{mod.name}", fromlist=["_"])
        for _, obj in inspect.getmembers(m, inspect.isclass):
            if issubclass(obj, BaseModel) and obj is not BaseModel and obj.__module__.startswith("airspace"):
                found.append(obj)
    return found


def test_every_input_model_forbids_unknown_fields():
    models = _models()
    assert len(models) >= 10
    for m in models:
        assert m.model_config.get("extra") == "forbid", f"{m.__name__} must set extra='forbid'"


def test_every_route_has_an_output_spec():
    src = inspect.getsource(airspace.app)
    used = set(re.findall(r'\bout\("([a-z_]+)"', src))
    assert used and used <= set(SPECS), used - set(SPECS)
    # specs never allow a forbidden field, at any depth
    def walk(spec):
        if isinstance(spec, dict):
            assert not (set(spec) & FORBIDDEN), set(spec) & FORBIDDEN
            for v in spec.values():
                walk(v)
        elif isinstance(spec, list):
            walk(spec[0])
    for spec in SPECS.values():
        walk(spec)
