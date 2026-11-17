"""Randomized robustness: malformed bodies, types and headers must never produce a 5xx or leak internals."""
import json
import random
import string

import requests

ENDPOINTS = ["/api/v1/auth/login", "/api/v1/auth/refresh", "/api/v1/auth/logout", "/api/v1/approvals", "/api/v1/users",
             "/api/v1/devices", "/api/v1/api-tokens", "/ingest/v1/telemetry", "/ingest/v1/evidence"]


def _rand_value(r: random.Random, depth=0):
    k = r.randrange(9 if depth < 3 else 6)
    if k == 0: return None
    if k == 1: return r.choice([True, False])
    if k == 2: return r.randint(-2 ** 63, 2 ** 63)
    if k == 3: return r.uniform(-1e308, 1e308)
    if k == 4: return "".join(r.choice(string.printable + "‮\u0000é漢") for _ in range(r.randrange(0, 300)))
    if k == 5: return r.choice(["' OR 1=1--", "<script>", "../../", "%n%n", "\\x00"])
    if k == 6: return [_rand_value(r, depth + 1) for _ in range(r.randrange(0, 5))]
    return {"".join(r.choice(string.ascii_letters) for _ in range(r.randrange(1, 12))): _rand_value(r, depth + 1) for _ in range(r.randrange(0, 6))}


def test_fuzz_json_bodies(env):
    r = random.Random(1337)
    h = env.h(env.A, "tenant_admin")
    for i in range(400):
        ep = r.choice(ENDPOINTS)
        body = _rand_value(r)
        if i % 3 == 0:
            data = json.dumps(body).encode()
        elif i % 3 == 1:
            data = bytes(r.randrange(256) for _ in range(r.randrange(0, 400)))
        else:
            data = json.dumps({"email": _rand_value(r), "password": _rand_value(r), "totp": _rand_value(r)}).encode()
        resp = requests.post(env.base + ep, data=data, headers={**h, "Content-Type": r.choice(["application/json", "text/plain", ""])}, timeout=10)
        assert resp.status_code < 500, (ep, data[:80], resp.status_code)
        assert "Traceback" not in resp.text and "sqlite" not in resp.text.lower()


def test_fuzz_headers(env):
    r = random.Random(7)
    for _ in range(150):
        hdrs = {"Authorization": "Bearer " + "".join(r.choice(string.ascii_letters + ".-_") for _ in range(r.randrange(0, 600))),
                "X-Device-Id": "".join(r.choice(string.ascii_letters) for _ in range(r.randrange(0, 80))),
                "X-Timestamp": str(r.randint(-10, 10 ** 15)), "X-Nonce": "".join(r.choice(string.ascii_letters) for _ in range(r.randrange(0, 90))),
                "X-Signature": "".join(r.choice(string.ascii_letters) for _ in range(r.randrange(0, 150))), "Content-Type": "application/json"}
        for ep in ("/api/v1/me", "/api/v1/alerts", "/ingest/v1/telemetry"):
            resp = (requests.post if ep.startswith("/ingest") else requests.get)(env.base + ep, headers=hdrs, data=b"{}" if ep.startswith("/ingest") else None, timeout=10)
            assert resp.status_code in (401, 413, 422, 429), (ep, resp.status_code)
