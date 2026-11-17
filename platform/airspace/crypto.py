"""Encryption at rest: AES-256-GCM with a versioned key ring (envelope encryption).

What is encrypted
  * users.totp_secret          MFA seeds
  * alerts.remote_id_text      over-the-air Remote ID strings (drone serials, operator data)
  * alerts.resolution_note     free-text notes that can name people
  * evidence files on disk     camera snapshots

Format
  field  "enc:v1:<kid>:<base64url(nonce || ciphertext+tag)>"
  file   b"AEB1" || len(kid) || kid || nonce(12) || ciphertext+tag

Every value is bound to where it lives through the AES-GCM associated data (AAD):
"<table>.<column>|<tenant_id>|<record_id>". A ciphertext copied into another row, column or tenant fails to decrypt.

Keys
  AIRSPACE_DATA_KEYS = "kid2:<base64 32 bytes>,kid1:<base64 32 bytes>"  (newest first; all listed keys decrypt,
  the first encrypts). In production these are data keys unwrapped from a KMS/HSM at start-up and supplied
  through *_FILE mounts; `python -m airspace.cli rewrap-data` re-encrypts everything under the newest key so
  an old key can be retired.
"""
from __future__ import annotations

import base64
import os
import struct

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

FIELD_PREFIX = "enc:v1:"
BLOB_MAGIC = b"AEB1"


class DecryptError(Exception):
    """Ciphertext is malformed, was tampered with, was moved to another record, or its key is not loaded."""


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class Keyring:
    def __init__(self, keys: dict[str, bytes], current: str):
        if not keys or current not in keys:
            raise ValueError("key ring needs at least one key and a current key id")
        for kid, k in keys.items():
            if len(k) != 32 or not kid.replace("-", "").replace("_", "").isalnum() or len(kid) > 32:
                raise ValueError("data keys must be 32 bytes with short alphanumeric ids")
        self._keys = dict(keys)
        self.current = current

    # ---- fields (TEXT columns) ----
    def encrypt_field(self, value: str | None, aad: str) -> str | None:
        if value is None:
            return None
        nonce = os.urandom(12)
        ct = AESGCM(self._keys[self.current]).encrypt(nonce, value.encode("utf-8"), aad.encode("utf-8"))
        return f"{FIELD_PREFIX}{self.current}:{_b64e(nonce + ct)}"

    def decrypt_field(self, token: str | None, aad: str) -> str | None:
        if token is None:
            return None
        if not isinstance(token, str) or not token.startswith(FIELD_PREFIX):
            raise DecryptError("not an encrypted field")
        try:
            kid, payload = token[len(FIELD_PREFIX):].split(":", 1)
            raw = _b64d(payload)
            key = self._keys[kid]
            return AESGCM(key).decrypt(raw[:12], raw[12:], aad.encode("utf-8")).decode("utf-8")
        except (KeyError, ValueError, InvalidTag, UnicodeDecodeError) as e:
            raise DecryptError("field failed authentication") from e

    def field_kid(self, token: str) -> str | None:
        if isinstance(token, str) and token.startswith(FIELD_PREFIX):
            return token[len(FIELD_PREFIX):].split(":", 1)[0]
        return None

    # ---- blobs (files) ----
    def encrypt_blob(self, data: bytes, aad: str) -> bytes:
        nonce = os.urandom(12)
        kid = self.current.encode()
        ct = AESGCM(self._keys[self.current]).encrypt(nonce, data, aad.encode("utf-8"))
        return BLOB_MAGIC + struct.pack(">B", len(kid)) + kid + nonce + ct

    def decrypt_blob(self, blob: bytes, aad: str) -> bytes:
        try:
            if not blob.startswith(BLOB_MAGIC):
                raise ValueError("magic")
            n = blob[4]
            kid = blob[5:5 + n].decode()
            rest = blob[5 + n:]
            return AESGCM(self._keys[kid]).decrypt(rest[:12], rest[12:], aad.encode("utf-8"))
        except (IndexError, KeyError, ValueError, InvalidTag, UnicodeDecodeError) as e:
            raise DecryptError("blob failed authentication") from e

    def blob_kid(self, blob: bytes) -> str | None:
        if blob.startswith(BLOB_MAGIC) and len(blob) > 5:
            return blob[5:5 + blob[4]].decode(errors="replace")
        return None


def aad(table_column: str, tenant_id: str | None, record_id: str) -> str:
    return f"{table_column}|{tenant_id or '_platform'}|{record_id}"


# Process-wide key ring for service functions that do not receive Settings (alerts, ingest).
# Installed by create_app(), the MCP server and the CLI from the same Settings object.
_ACTIVE: Keyring | None = None


def install(ring: Keyring) -> None:
    global _ACTIVE
    _ACTIVE = ring


def active() -> Keyring:
    if _ACTIVE is None:
        raise RuntimeError("data key ring not installed")
    return _ACTIVE
