"""Encryption at rest: sensitive values never reach the database file or the evidence store in plaintext,
ciphertexts are bound to their row (AAD), tampering is detected, and keys rotate."""
from __future__ import annotations

import base64
import glob
import os
import secrets

import pytest

from airspace import crypto
from airspace.config import Settings
from airspace.crypto import DecryptError, Keyring, aad
from conftest import JPEG_PLAIN_MARKER, telemetry


def _db_bytes(env) -> bytes:
    """Every byte SQLite has written: main file plus WAL."""
    out = b""
    for p in (env.s.db_path, env.s.db_path + "-wal"):
        if os.path.exists(p):
            with open(p, "rb") as fh:
                out += fh.read()
    return out


def test_totp_seed_not_stored_in_plaintext(env):
    seed = env.A.users["operator"]["totp"]
    assert seed.encode() not in _db_bytes(env)
    db = env.db()
    raw = db.one("SELECT totp_secret FROM users WHERE id = ?", (env.A.users["operator"]["id"],))["totp_secret"]
    db.close()
    assert raw.startswith("enc:v1:")


def test_remote_id_and_note_encrypted_but_readable_through_api(env):
    marker = "RID-SERIAL-" + secrets.token_hex(6)
    r, _ = env.signed(env.A, "/ingest/v1/telemetry", telemetry(rid=marker))
    aid = r.json()["alert_ids"][0]
    note = "pilot seen near gate " + secrets.token_hex(6)
    assert env.post(f"/api/v1/alerts/{aid}/resolve", json={"note": note}, headers=env.h(env.A, "operator")).status_code == 200
    got = env.get(f"/api/v1/alerts/{aid}", headers=env.h(env.A, "viewer")).json()
    assert got["remote_id_text"] == marker and got["resolution_note"] == note
    blob = _db_bytes(env)
    assert marker.encode() not in blob and note.encode() not in blob


def test_ciphertext_moved_to_another_row_fails(env, new_alert):
    a1 = new_alert(rid="first-" + secrets.token_hex(4))
    a2 = new_alert(rid="second-" + secrets.token_hex(4))
    db = env.db()
    c1 = db.one("SELECT remote_id_text FROM alerts WHERE id = ?", (a1,))["remote_id_text"]
    db.x("UPDATE alerts SET remote_id_text = ? WHERE id = ?", (c1, a2))     # attacker with DB write access
    db.close()
    got = env.get(f"/api/v1/alerts/{a2}", headers=env.h(env.A, "viewer")).json()
    assert got["remote_id_text"] == "[unreadable: integrity check failed]"


def test_evidence_file_is_ciphertext(env, stored_evidence):
    _, eid = stored_evidence
    db = env.db()
    key = db.one("SELECT storage_key FROM evidence WHERE id = ?", (eid,))["storage_key"]
    db.close()
    with open(os.path.join(env.s.evidence_dir, key[:2], key), "rb") as fh:
        blob = fh.read()
    assert blob.startswith(crypto.BLOB_MAGIC)
    assert b"\xff\xd8\xff" not in blob[:16] and JPEG_PLAIN_MARKER not in blob
    for path in glob.glob(os.path.join(env.s.evidence_dir, "*", "*")):
        with open(path, "rb") as fh:
            assert JPEG_PLAIN_MARKER not in fh.read()


def test_field_tamper_and_wrong_aad_detected():
    ring = Keyring({"k1": secrets.token_bytes(32)}, "k1")
    tok = ring.encrypt_field("secret", aad("users.totp_secret", "tn_a", "u1"))
    assert ring.decrypt_field(tok, aad("users.totp_secret", "tn_a", "u1")) == "secret"
    with pytest.raises(DecryptError):
        ring.decrypt_field(tok, aad("users.totp_secret", "tn_b", "u1"))          # other tenant
    with pytest.raises(DecryptError):
        ring.decrypt_field(tok, aad("alerts.remote_id_text", "tn_a", "u1"))      # other column
    flipped = tok[:-3] + ("A" if tok[-3] != "A" else "B") + tok[-2:]
    with pytest.raises(DecryptError):
        ring.decrypt_field(flipped, aad("users.totp_secret", "tn_a", "u1"))
    with pytest.raises(DecryptError):
        ring.decrypt_field("plaintext-value", aad("users.totp_secret", "tn_a", "u1"))   # no silent plaintext fallback


def test_blob_tamper_detected():
    ring = Keyring({"k1": secrets.token_bytes(32)}, "k1")
    blob = bytearray(ring.encrypt_blob(b"evidence bytes", "evidence|tn_a|e1"))
    blob[-1] ^= 1
    with pytest.raises(DecryptError):
        ring.decrypt_blob(bytes(blob), "evidence|tn_a|e1")


def test_key_rotation_rewrap(tmp_path, monkeypatch):
    """Add a new key, re-encrypt everything, then retire the old key: data stays readable."""
    from airspace import admin, cli
    from airspace.db import DB, migrate
    old, new = secrets.token_bytes(32), secrets.token_bytes(32)
    s = Settings(db_path=str(tmp_path / "rot.db"), evidence_dir=str(tmp_path / "ev"))
    s.pepper = secrets.token_bytes(32)
    s.data_keys, s.current_data_kid = {"old": old}, "old"
    migrate(s.db_path)
    db = DB(s.db_path)
    crypto.install(s.keyring())
    boot = admin.bootstrap_tenant(s, db, "Rot", "rot@a.example", "correct horse battery staple 9")
    assert db.one("SELECT totp_secret FROM users WHERE id = ?", (boot["admin_user_id"],))["totp_secret"].startswith("enc:v1:old:")
    s.data_keys, s.current_data_kid = {"new": new, "old": old}, "new"
    crypto.install(s.keyring())
    res = cli.rewrap(s, db)
    assert res["rewrapped"]["users.totp_secret"] == 1
    s.data_keys, s.current_data_kid = {"new": new}, "new"                       # old key retired
    raw = db.one("SELECT tenant_id, totp_secret FROM users WHERE id = ?", (boot["admin_user_id"],))
    assert s.keyring().decrypt_field(raw["totp_secret"], aad("users.totp_secret", raw["tenant_id"], boot["admin_user_id"])) == boot["totp_secret"]
    db.close()


def test_keyring_rejects_weak_keys():
    with pytest.raises(ValueError):
        Keyring({"k1": b"short"}, "k1")
    with pytest.raises(ValueError):
        Keyring({"k1": secrets.token_bytes(32)}, "missing")


@pytest.fixture(autouse=True)
def _restore_ring(env):
    yield
    crypto.install(env.s.keyring())       # the rotation test installs its own ring; put the server's back
