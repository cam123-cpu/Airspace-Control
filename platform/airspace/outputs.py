"""Explicit response shapes. Every JSON body the API returns passes through `trim(name, body)`.

A spec maps each allowed key to:
  None        a scalar value (string, number, bool, null)
  ANY         free-form JSON produced by the platform itself (only `detail`/`params`, already redacted)
  {..}        a nested object with its own spec
  [{..}]      a list of objects with that spec

Keys not in the spec are dropped and logged (production) or raise (strict mode, used by the test suite and CI),
so a new database column can never leak through an existing endpoint by accident. Independently of the specs,
names in FORBIDDEN are never allowed anywhere in a response.
"""
from __future__ import annotations

from typing import Any

ANY = object()

FORBIDDEN = frozenset({
    "pw_hash", "password", "password_hash", "salt", "totp_secret", "totp_last_step", "token_hash", "id_hash",
    "code_hash", "refresh_token_hash", "family_id", "storage_key", "public_key", "private_key", "pepper",
    "failed_count", "locked_until", "tenant_secret", "data_key", "jwt_key", "seq_internal",
})

_ALERT = {"id": None, "site_id": None, "ext_track_id": None, "severity": None, "title": None, "remote_id_text": None,
          "status": None, "created_at": None, "ack_by": None, "ack_at": None, "resolved_by": None, "resolved_at": None,
          "resolution_note": None, "legal_hold": None}
_TOKENS = {"access_token": None, "token_type": None, "expires_in": None, "refresh_token": None, "csrf_token": None}
_APPROVAL_RESULT = {"id": None, "status": None, "expires_at": None, "grant_for_requester": None}

SPECS: dict[str, dict] = {
    "healthz": {"ok": None},
    "challenge": {"kind": None, "algorithm": None, "token": None, "difficulty": None, "expires_at": None},
    "login": _TOKENS,
    "refresh": _TOKENS,
    "logout": {"ok": None},
    "me": {"id": None, "tenant_id": None, "role": None},
    "sites": {"items": [{"id": None, "name": None, "alert_threshold": None, "created_at": None}]},
    "sensors": {"items": [{"id": None, "kind": None, "enabled": None, "status": None, "last_seen": None}]},
    "alert_list": {"items": [_ALERT], "next_cursor": None},
    "alert_get": {**_ALERT, "evidence": [{"id": None, "sha256": None, "size": None, "content_type": None, "created_at": None}]},
    "tracks": {"items": [{"id": None, "ext_track_id": None, "cls": None, "confidence": None, "x": None, "y": None, "z": None,
                          "observed_at": None}]},
    "alert_ack": {"id": None, "status": None},
    "alert_resolve": {"id": None, "status": None},
    "alert_hold": {"id": None, "legal_hold": None},
    "mcp_confirmation": {"code": None, "expires_in": None},
    "evidence_meta": {"id": None, "alert_id": None, "site_id": None, "sha256": None, "size": None, "content_type": None,
                      "server_sig": None, "created_at": None, "legal_hold": None, "signature_valid": None},
    "approval_create": _APPROVAL_RESULT,
    "approval_list": {"items": [{"id": None, "action": None, "params": ANY, "requested_by": None, "requested_at": None,
                                 "expires_at": None, "status": None}]},
    "approval_decide": _APPROVAL_RESULT,
    "user_create": {"id": None, "role": None, "mfa_required": None, "totp_provisioning_uri": None},
    "device_create": {"id": None, "site_id": None},
    "token_create": {"id": None, "token": None, "expires_at": None},
    "token_revoke": {"id": None, "revoked": None},
    "audit_list": {"items": [{"seq": None, "ts": None, "actor": None, "action": None, "target": None, "detail": ANY, "hash": None}]},
    "audit_verify": {"ok": None, "records": None, "head": None, "broken_at_seq": None},
    "ingest_telemetry": {"accepted": {"sensors": None, "tracks": None, "alerts": None}, "alert_ids": ANY},
    "ingest_evidence": {"id": None, "sha256": None},
    "error": {"error": None, "message": None, "request_id": None,
              "challenge": {"kind": None, "algorithm": None, "token": None, "difficulty": None, "expires_at": None}},
}


class OutputViolation(RuntimeError):
    pass


def _walk(spec: Any, data: Any, path: str, strict: bool, dropped: list[str]) -> Any:
    if spec is ANY:
        return data
    if isinstance(spec, list):
        if not isinstance(data, list):
            raise OutputViolation(f"{path}: expected a list")
        return [_walk(spec[0], v, f"{path}[]", strict, dropped) for v in data]
    if isinstance(spec, dict):
        if data is None:
            return None
        if not isinstance(data, dict):
            raise OutputViolation(f"{path}: expected an object")
        out = {}
        for k, v in data.items():
            if k in FORBIDDEN or k not in spec:
                if strict:
                    raise OutputViolation(f"{path}.{k}: field not allowed in this response")
                dropped.append(f"{path}.{k}")
                continue
            out[k] = _walk(spec[k], v, f"{path}.{k}", strict, dropped)
        return out
    if isinstance(data, (dict, list)):
        raise OutputViolation(f"{path}: expected a scalar")
    return data


def trim(name: str, data: Any, strict: bool = False) -> tuple[Any, list[str]]:
    dropped: list[str] = []
    return _walk(SPECS[name], data, name, strict, dropped), dropped
