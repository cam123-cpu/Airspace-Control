"""Evidence snapshots: hashed and server-signed at capture, stored outside any web root under random keys,
downloadable only through a one-time grant created by a two-person-approved export."""
from __future__ import annotations

import base64
import hashlib
import os
import secrets
import time
import uuid

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from . import audit
from .auth import Principal
from .config import Settings
from .crypto import DecryptError, aad
from .db import DB
from .errors import Invalid, NotFound
from .rbac import require

_MAGIC = {b"\xff\xd8\xff": "image/jpeg", b"\x89PNG\r\n\x1a\n": "image/png"}
_EXT = {"image/jpeg": "jpg", "image/png": "png"}


def sniff(data: bytes) -> str | None:
    for magic, ctype in _MAGIC.items():
        if data.startswith(magic):
            return ctype
    return None


def _sig_payload(eid: str, sha: str, tenant: str, alert_id: str, created: int) -> bytes:
    return f"{eid}|{sha}|{tenant}|{alert_id}|{created}".encode()


def _store_path(s: Settings, key: str) -> str:
    if not (len(key) == 64 and all(c in "0123456789abcdef" for c in key)):
        raise Invalid("bad storage key")        # never derived from user input, checked anyway
    base = os.path.realpath(s.evidence_dir)
    path = os.path.realpath(os.path.join(base, key[:2], key))
    if not path.startswith(base + os.sep):
        raise Invalid("bad path")
    return path


def store(s: Settings, db: DB, dev: Principal, site_id: str, alert_id: str, data: bytes, now: int | None = None) -> dict:
    now = int(now or time.time())
    if not data or len(data) > s.max_evidence_bytes:
        raise Invalid("evidence size")
    ctype = sniff(data)
    if not ctype:
        raise Invalid("only JPEG or PNG evidence is accepted")
    if not db.one("SELECT id FROM alerts WHERE tenant_id = ? AND site_id = ? AND id = ?", (dev.tenant_id, site_id, alert_id)):
        raise NotFound()
    sha = hashlib.sha256(data).hexdigest()
    eid, key = uuid.uuid4().hex, secrets.token_hex(32)
    sig = base64.b64encode(s.evidence_key.sign(_sig_payload(eid, sha, dev.tenant_id, alert_id, now))).decode()
    path = _store_path(s, key)
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as fh:
        # encrypted at rest (AES-256-GCM), bound to tenant + evidence id; the SHA-256 and signature cover the plaintext
        fh.write(s.keyring().encrypt_blob(data, aad("evidence", dev.tenant_id, eid)))
    with db.tx():
        db.x("INSERT INTO evidence (id, tenant_id, site_id, alert_id, sha256, size, content_type, storage_key, server_sig, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
             (eid, dev.tenant_id, site_id, alert_id, sha, len(data), ctype, key, sig, now))
        audit.append(db, dev.tenant_id, dev.actor, "evidence.stored", eid, {"alert": alert_id, "sha256": sha}, now)
    return {"id": eid, "sha256": sha}


def metadata(s: Settings, db: DB, p: Principal, eid: str) -> dict:
    require(p, "evidence.read")
    r = db.one("SELECT id, alert_id, site_id, sha256, size, content_type, server_sig, created_at, legal_hold FROM evidence WHERE tenant_id = ? AND id = ?", (p.tenant_id, eid))
    if not r:
        raise NotFound()
    r["signature_valid"] = verify_signature(s, r, p.tenant_id)
    return r


def verify_signature(s: Settings, r: dict, tenant_id: str) -> bool:
    try:
        pub: Ed25519PublicKey = s.evidence_key.public_key()
        pub.verify(base64.b64decode(r["server_sig"]), _sig_payload(r["id"], r["sha256"], tenant_id, r["alert_id"], r["created_at"]))
        return True
    except (InvalidSignature, ValueError):
        return False


def create_export_grant(db: DB, tenant_id: str, evidence_id: str, user_id: str, now: int) -> str:
    raw = "eg_" + secrets.token_urlsafe(24)
    db.x("INSERT INTO export_grants (id_hash, tenant_id, evidence_id, user_id, expires_at) VALUES (?, ?, ?, ?, ?)",
         (hashlib.sha256(raw.encode()).hexdigest(), tenant_id, evidence_id, user_id, now + 600))
    return raw


def download(s: Settings, db: DB, p: Principal, eid: str, grant: str, now: int | None = None) -> tuple[bytes, str, str]:
    require(p, "evidence.read")
    now = int(now or time.time())
    if not isinstance(grant, str) or not grant.startswith("eg_") or len(grant) > 80:
        raise NotFound()
    with db.tx():
        n = db.x("UPDATE export_grants SET used_at = ? WHERE id_hash = ? AND tenant_id = ? AND evidence_id = ? AND user_id = ? AND used_at IS NULL AND expires_at > ?",
                 (now, hashlib.sha256(grant.encode()).hexdigest(), p.tenant_id, eid, p.id, now))
        if n != 1:
            raise NotFound()
        r = db.one("SELECT storage_key, content_type, sha256 FROM evidence WHERE tenant_id = ? AND id = ?", (p.tenant_id, eid))
        if not r:
            raise NotFound()
        audit.append(db, p.tenant_id, p.actor, "evidence.downloaded", eid, {"sha256": r["sha256"]}, now)
    with open(_store_path(s, r["storage_key"]), "rb") as fh:
        blob = fh.read()
    try:
        data = s.keyring().decrypt_blob(blob, aad("evidence", p.tenant_id, eid))
    except DecryptError:
        raise Invalid("evidence integrity check failed")
    if hashlib.sha256(data).hexdigest() != r["sha256"]:
        raise Invalid("evidence integrity check failed")
    return data, r["content_type"], f"evidence-{eid}.{_EXT[r['content_type']]}"


def delete_file(s: Settings, key: str) -> None:
    try:
        os.remove(_store_path(s, key))
    except FileNotFoundError:
        pass
