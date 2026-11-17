"""Shared fixtures: a real uvicorn server on localhost, two tenants with users of every role, sites, sensors
and signed edge devices. HTTP tests talk to it with `requests` (the service is exercised end to end)."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import socket
import sys
import threading
import time
from dataclasses import dataclass, field

import pytest
import requests
import uvicorn
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airspace import admin  # noqa: E402
from airspace.app import create_app  # noqa: E402
from airspace.auth import issue_access_token, totp_now  # noqa: E402
from airspace.config import Settings  # noqa: E402
from airspace.db import DB, migrate  # noqa: E402
from airspace.ingest import canonical  # noqa: E402
from airspace.ratelimit import LIMITER  # noqa: E402

PW = "correct horse battery staple 9"
ORIGIN = "https://console.airspace.example"


def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


@dataclass
class Tenant:
    id: str
    site: str
    sensors: dict
    users: dict = field(default_factory=dict)       # role -> {"id","email","totp"}
    device_id: str = ""
    device_key: Ed25519PrivateKey | None = None


@dataclass
class Env:
    s: Settings
    base: str
    A: Tenant
    B: Tenant

    def db(self) -> DB:
        return DB(self.s.db_path)

    def token(self, t: Tenant, role: str) -> str:
        u = t.users[role]
        return issue_access_token(self.s, u["id"], t.id, role.rstrip("0123456789"))

    def h(self, t: Tenant, role: str) -> dict:
        return {"Authorization": "Bearer " + self.token(t, role)}

    def get(self, path, **kw):
        return requests.get(self.base + path, timeout=10, **kw)

    def post(self, path, **kw):
        return requests.post(self.base + path, timeout=10, **kw)

    def signed(self, t: Tenant, path: str, body: bytes, *, ts: int | None = None, nonce: str | None = None,
               key: Ed25519PrivateKey | None = None, device_id: str | None = None, extra: dict | None = None, method: str = "POST"):
        ts_s = str(ts if ts is not None else int(time.time()))
        nonce = nonce or b64u(secrets.token_bytes(18))
        sig = (key or t.device_key).sign(canonical(method, path, ts_s, nonce, body))
        hdrs = {"X-Device-Id": device_id or t.device_id, "X-Timestamp": ts_s, "X-Nonce": nonce, "X-Signature": b64u(sig),
                "Content-Type": "application/json"}
        hdrs.update(extra or {})
        return requests.post(self.base + path, data=body, headers=hdrs, timeout=10), nonce


def totp_secret_of(s: Settings, db: DB, user_id: str) -> str:
    """TOTP seeds are encrypted at rest; tests decrypt them the same way the login path does."""
    from airspace.crypto import aad
    u = db.one("SELECT tenant_id, totp_secret FROM users WHERE id = ?", (user_id,))
    return s.keyring().decrypt_field(u["totp_secret"], aad("users.totp_secret", u["tenant_id"], user_id))


def _free_port() -> int:
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        return sk.getsockname()[1]


def _make_tenant(s: Settings, db: DB, name: str, prefix: str) -> Tenant:
    boot = admin.bootstrap_tenant(s, db, name, f"admin@{prefix}.example", PW)
    tid = boot["tenant_id"]
    site = admin.create_site(db, tid, f"{name} Site 1", ["radar", "rf", "acoustic", "camera_ir"])
    t = Tenant(tid, site["site_id"], {x["kind"]: x["id"] for x in site["sensors"]})
    secret = totp_secret_of(s, db, boot["admin_user_id"])
    t.users["tenant_admin"] = {"id": boot["admin_user_id"], "email": f"admin@{prefix}.example", "totp": secret}
    from airspace.auth import Principal
    adm = Principal("user", boot["admin_user_id"], tid, "tenant_admin")
    for role in ("viewer", "operator", "supervisor"):
        for n in ("", "2"):
            if n and role != "supervisor":
                continue
            email = f"{role}{n}@{prefix}.example"
            r = admin.create_user(s, db, adm, {"email": email, "role": role, "password": PW})
            sec = totp_secret_of(s, db, r["id"])
            t.users[role + n] = {"id": r["id"], "email": email, "totp": sec}
    key = Ed25519PrivateKey.generate()
    pub = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    d = admin.register_device(db, adm, {"site_id": t.site, "public_key": b64u(pub)})
    t.device_id, t.device_key = d["id"], key
    return t


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("airspace")
    s = Settings(db_path=str(d / "test.db"), evidence_dir=str(d / "evidence"), cors_origins=(ORIGIN,), strict_outputs=True,
                 pow_difficulty=10)
    s.pepper = secrets.token_bytes(32)
    s.jwt_keys = {"k1": Ed25519PrivateKey.generate(), "k0": Ed25519PrivateKey.generate()}
    s.current_kid = "k1"
    s.evidence_key = Ed25519PrivateKey.generate()
    migrate(s.db_path)
    db = DB(s.db_path)
    A = _make_tenant(s, db, "Tenant A", "a")
    B = _make_tenant(s, db, "Tenant B", "b")
    db.close()
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(s), host="127.0.0.1", port=port, log_level="warning", access_log=False, server_header=False, date_header=False))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield Env(s, f"http://127.0.0.1:{port}", A, B)
    server.should_exit = True
    th.join(timeout=5)


@pytest.fixture(autouse=True)
def _reset_limits():
    from airspace.challenge import FAILURES
    LIMITER.reset()
    FAILURES.reset()
    yield


def telemetry(alerts=1, tracks=1, title="Drone over yard", rid=None, sensors=None):
    return json.dumps({
        "sensors": sensors or [],
        "tracks": [{"ext_track_id": f"T-{i}", "cls": "drone", "confidence": 0.93, "x": 10.0 + i, "y": -5.0, "z": 30.0,
                    "observed_at": int(time.time())} for i in range(tracks)],
        "alerts": [{"ext_track_id": f"T-{i}", "severity": "alert", "title": title, **({"remote_id_text": rid} if rid else {})}
                   for i in range(alerts)],
    }).encode()


@pytest.fixture
def new_alert(env):
    """Create a fresh open alert in tenant A through the signed ingest path; returns its id."""
    def make(t=None, **kw):
        t = t or env.A
        r, _ = env.signed(t, "/ingest/v1/telemetry", telemetry(**kw))
        assert r.status_code == 202, r.text
        return r.json()["alert_ids"][0]
    return make


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


JPEG_PLAIN_MARKER = b"evidence-bytes"
JPEG = b"\xff\xd8\xff\xe0" + JPEG_PLAIN_MARKER * 50


@pytest.fixture
def stored_evidence(env, new_alert):
    """An alert in tenant A with one stored JPEG; returns (alert_id, evidence_id)."""
    aid = new_alert()
    r, _ = env.signed(env.A, f"/ingest/v1/evidence?alert_id={aid}", JPEG)
    assert r.status_code == 201, r.text
    return aid, r.json()["id"]
