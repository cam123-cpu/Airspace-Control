"""Signed edge-device ingest: signature, clock window, replay, schema bounds, tenancy, idempotency."""
import json
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from conftest import telemetry

P = "/ingest/v1/telemetry"


def test_valid_signed_telemetry(env):
    r, _ = env.signed(env.A, P, telemetry(alerts=2, tracks=3, sensors=[{"sensor_id": env.A.sensors["radar"], "status": "ok"}]))
    assert r.status_code == 202, r.text
    assert r.json()["accepted"] == {"sensors": 1, "tracks": 3, "alerts": 2}


def test_unsigned_and_bad_signatures(env):
    body = telemetry()
    assert env.post(P, data=body, headers={"Content-Type": "application/json"}).status_code == 401
    r, _ = env.signed(env.A, P, body, key=Ed25519PrivateKey.generate())          # wrong key
    assert r.status_code == 401
    r, _ = env.signed(env.A, P, body, device_id=env.B.device_id)                  # A's key claiming to be B's device
    assert r.status_code == 401
    r, _ = env.signed(env.A, P, body, device_id="dev_doesnotexist")
    assert r.status_code == 401


def test_body_tampering_detected(env):
    body = telemetry()
    ts, nonce = str(int(time.time())), "n" * 24
    from airspace.ingest import canonical
    from conftest import b64u
    sig = b64u(env.A.device_key.sign(canonical("POST", P, ts, nonce, body)))
    tampered = body.replace(b"Drone over yard", b"All clear here!")
    r = env.post(P, data=tampered, headers={"X-Device-Id": env.A.device_id, "X-Timestamp": ts, "X-Nonce": nonce, "X-Signature": sig,
                                            "Content-Type": "application/json"})
    assert r.status_code == 401


def test_clock_window(env):
    assert env.signed(env.A, P, telemetry(), ts=int(time.time()) - 120)[0].status_code == 401
    assert env.signed(env.A, P, telemetry(), ts=int(time.time()) + 120)[0].status_code == 401
    assert env.signed(env.A, P, telemetry(), ts=int(time.time()) - 30)[0].status_code == 202


def test_replay_rejected(env):
    body = telemetry()
    r1, nonce = env.signed(env.A, P, body)
    assert r1.status_code == 202
    r2, _ = env.signed(env.A, P, body, nonce=nonce)
    assert r2.status_code == 401


def test_deactivated_device(env):
    db = env.db()
    db.x("UPDATE devices SET active = 0 WHERE id = ?", (env.B.device_id,))
    db.close()
    try:
        assert env.signed(env.B, P, telemetry())[0].status_code == 401
    finally:
        db = env.db()
        db.x("UPDATE devices SET active = 1 WHERE id = ?", (env.B.device_id,))
        db.close()


def _send(env, payload):
    return env.signed(env.A, P, json.dumps(payload).encode())[0]


def test_schema_rejects_bad_payloads(env):
    now = int(time.time())
    trk = {"ext_track_id": "T1", "cls": "drone", "confidence": 0.9, "x": 1.0, "y": 1.0, "z": 1.0, "observed_at": now}
    bad = [
        {"tracks": [{**trk, "extra": 1}]},                                 # unknown field
        {"tracks": [{**trk, "confidence": 7}]},                            # out of range
        {"tracks": [{**trk, "cls": "<script>"}]},                          # not in enum
        {"tracks": [{**trk, "x": "1e999"}]},                               # wrong type (strict)
        {"tracks": [trk] * 501},                                           # list too long
        {"alerts": [{"ext_track_id": "T1", "severity": "alert", "title": "x" * 201}]},
        {"alerts": [{"ext_track_id": "T1'; DROP TABLE alerts;--", "severity": "alert", "title": "x"}]},
        {"tenant_id": env.B.id},                                           # tenant can't be supplied
    ]
    for b in bad:
        assert _send(env, b).status_code in (413, 422), b
    # NaN / Infinity literals are refused by the parser
    assert env.signed(env.A, P, b'{"tracks":[{"ext_track_id":"T","cls":"drone","confidence":NaN,"x":1,"y":1,"z":1,"observed_at":%d}]}' % now)[0].status_code == 422


def test_other_sites_sensor_rejected(env):
    r = _send(env, {"sensors": [{"sensor_id": env.B.sensors["radar"], "status": "offline"}]})
    assert r.status_code == 422
    db = env.db()
    assert db.one("SELECT status FROM sensors WHERE id = ?", (env.B.sensors["radar"],))["status"] != "offline"
    db.close()


def test_idempotency_key(env):
    body = telemetry()
    r1, _ = env.signed(env.A, P, body, extra={"Idempotency-Key": "batch-000123"})
    r2, _ = env.signed(env.A, P, body, extra={"Idempotency-Key": "batch-000123"})
    assert r1.status_code == r2.status_code == 202
    assert r1.json() == r2.json()


def test_untrusted_remote_id_text_stored_verbatim_returned_as_json(env):
    evil = '<img src=x onerror=alert(1)> IGNORE PREVIOUS INSTRUCTIONS'
    r, _ = env.signed(env.A, P, telemetry(rid=evil))
    aid = r.json()["alert_ids"][0]
    g = env.get(f"/api/v1/alerts/{aid}", headers=env.h(env.A, "viewer"))
    assert g.headers["content-type"].startswith("application/json")
    assert g.json()["remote_id_text"] == evil       # data preserved; clients must encode on render (JSON, nosniff, CSP)
