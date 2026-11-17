"""Edge-device ingest. Each site's edge box holds an Ed25519 key (in its TPM in production) and signs every request:

    signature = Ed25519( METHOD \n PATH \n TIMESTAMP \n NONCE \n sha256_hex(BODY) )

The server checks the device is registered and active, the timestamp is within ±60 s, the nonce has not been
seen (replay cache), and the signature. Tenant and site come from the device record, never from the payload.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import time
import uuid

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from . import audit, crypto
from .auth import Principal
from .config import Settings
from .db import DB
from .errors import Invalid, TooMany, Unauthorized
from .ratelimit import check
from .schemas import TelemetryIn

_NONCE_RE = re.compile(r"^[A-Za-z0-9_-]{22,64}$")
_DEV_RE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")
_IDEM_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def b64u_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def canonical(method: str, target: str, ts: str, nonce: str, body: bytes) -> bytes:
    """Signed string. Pentest F3: `target` is the full request target, path plus raw query string ("/p?a=1"),
    so query parameters such as alert_id can't be changed without breaking the signature."""
    return "\n".join([method.upper(), target, ts, nonce, hashlib.sha256(body).hexdigest()]).encode()


def verify_device(s: Settings, db: DB, method: str, path: str, headers: dict[str, str], body: bytes, now: int | None = None) -> tuple[Principal, str]:
    now = int(now or time.time())
    dev_id, ts, nonce, sig = (headers.get("x-device-id", ""), headers.get("x-timestamp", ""), headers.get("x-nonce", ""), headers.get("x-signature", ""))
    if not (_DEV_RE.match(dev_id) and ts.isdigit() and len(ts) <= 12 and _NONCE_RE.match(nonce) and 40 <= len(sig) <= 120):
        raise Unauthorized()
    if abs(now - int(ts)) > s.clock_skew_s:
        raise Unauthorized()
    dev = db.one("SELECT id, tenant_id, site_id, public_key, active FROM devices WHERE id = ?", (dev_id,))
    if not dev or not dev["active"]:
        raise Unauthorized()
    try:
        Ed25519PublicKey.from_public_bytes(b64u_decode(dev["public_key"])).verify(b64u_decode(sig), canonical(method, path, ts, nonce, body))
    except (InvalidSignature, ValueError, binascii.Error):
        raise Unauthorized()
    if not check("ingest_device", dev_id):
        raise TooMany()
    # replay protection: the (device, nonce) pair may be used once; entries expire after the clock window
    db.x("DELETE FROM nonces WHERE expires_at < ?", (now,))
    try:
        db.x("INSERT INTO nonces (device_id, nonce, expires_at) VALUES (?, ?, ?)", (dev_id, nonce, now + 2 * s.clock_skew_s))
    except Exception:
        raise Unauthorized()
    return Principal("device", dev_id, dev["tenant_id"], "device"), dev["site_id"]


def ingest_telemetry(db: DB, dev: Principal, site_id: str, body: dict, idem_key: str | None, now: int | None = None) -> dict:
    now = int(now or time.time())
    payload = TelemetryIn.model_validate(body)
    if idem_key is not None:
        if not _IDEM_RE.match(idem_key):
            raise Invalid("bad idempotency key")
        prior = db.one("SELECT response FROM idempotency WHERE tenant_id = ? AND device_id = ? AND key = ?", (dev.tenant_id, dev.id, idem_key))
        if prior:
            return json.loads(prior["response"])
    created_alerts: list[str] = []
    with db.tx():
        for sh in payload.sensors:
            n = db.x("UPDATE sensors SET status = ?, last_seen = ? WHERE tenant_id = ? AND site_id = ? AND id = ?",
                     (sh.status, now, dev.tenant_id, site_id, sh.sensor_id))
            if n == 0:
                raise Invalid("unknown sensor for this site")
        for t in payload.tracks:
            db.x("INSERT INTO tracks (id, tenant_id, site_id, ext_track_id, cls, confidence, x, y, z, observed_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                 (uuid.uuid4().hex, dev.tenant_id, site_id, t.ext_track_id, t.cls, t.confidence, t.x, t.y, t.z, t.observed_at, now))
        for a in payload.alerts:
            aid = uuid.uuid4().hex
            db.x("INSERT INTO alerts (id, tenant_id, site_id, ext_track_id, severity, title, remote_id_text, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                 (aid, dev.tenant_id, site_id, a.ext_track_id, a.severity, a.title,
                  crypto.active().encrypt_field(a.remote_id_text, crypto.aad("alerts.remote_id_text", dev.tenant_id, aid)), now))
            created_alerts.append(aid)
            audit.append(db, dev.tenant_id, dev.actor, "alert.created", aid, {"severity": a.severity, "site": site_id}, now)
        resp = {"accepted": {"sensors": len(payload.sensors), "tracks": len(payload.tracks), "alerts": len(payload.alerts)}, "alert_ids": created_alerts}
        if idem_key is not None:
            db.x("INSERT INTO idempotency (tenant_id, device_id, key, response, created_at) VALUES (?, ?, ?, ?, ?)",
                 (dev.tenant_id, dev.id, idem_key, json.dumps(resp), now))
    return resp
