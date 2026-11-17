"""Force HTTPS: with require_https on, plain-http requests are refused. The scheme is taken from X-Forwarded-Proto
only when the direct peer is a trusted proxy; anyone else's header is ignored."""
from __future__ import annotations

import ipaddress
import secrets
import socket
import threading
import time

import pytest
import requests
import uvicorn
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from airspace.app import create_app
from airspace.config import Settings


def _serve(s: Settings):
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(s), host="127.0.0.1", port=port, log_level="warning",
                                           access_log=False, proxy_headers=False))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    return server, th, f"http://127.0.0.1:{port}"


def _settings(tmp_path, proxies: str) -> Settings:
    s = Settings(db_path=str(tmp_path / "h.db"), evidence_dir=str(tmp_path / "ev"), require_https=True,
                 trust_proxy_headers=True, trusted_proxies=(ipaddress.ip_network(proxies),))
    s.pepper = secrets.token_bytes(32)
    s.jwt_keys, s.current_kid = {"k": Ed25519PrivateKey.generate()}, "k"
    s.evidence_key = Ed25519PrivateKey.generate()
    return s


@pytest.fixture
def behind_trusted_proxy(tmp_path):
    server, th, base = _serve(_settings(tmp_path, "127.0.0.1/32"))     # the test client plays the gateway
    yield base
    server.should_exit = True
    th.join(timeout=5)


@pytest.fixture
def untrusted_peer(tmp_path):
    server, th, base = _serve(_settings(tmp_path, "10.0.0.0/8"))       # gateway elsewhere; we're a random client
    yield base
    server.should_exit = True
    th.join(timeout=5)


def test_plain_http_refused(behind_trusted_proxy):
    r = requests.post(behind_trusted_proxy + "/api/v1/auth/challenge", timeout=10)
    assert r.status_code == 400 and r.json()["error"] == "https_required"
    assert "max-age=" in r.headers["strict-transport-security"]


def test_https_via_trusted_proxy_accepted(behind_trusted_proxy):
    r = requests.post(behind_trusted_proxy + "/api/v1/auth/challenge", headers={"X-Forwarded-Proto": "https"}, timeout=10)
    assert r.status_code == 200


def test_proxy_says_http_refused(behind_trusted_proxy):
    r = requests.post(behind_trusted_proxy + "/api/v1/auth/challenge", headers={"X-Forwarded-Proto": "http"}, timeout=10)
    assert r.status_code == 400


def test_forged_proto_from_untrusted_peer_ignored(untrusted_peer):
    r = requests.post(untrusted_peer + "/api/v1/auth/challenge", headers={"X-Forwarded-Proto": "https"}, timeout=10)
    assert r.status_code == 400 and r.json()["error"] == "https_required"


def test_health_probe_exempt(behind_trusted_proxy):
    assert requests.get(behind_trusted_proxy + "/healthz", timeout=10).status_code == 200
