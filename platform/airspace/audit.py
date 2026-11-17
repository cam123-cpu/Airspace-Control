"""Append-only, per-tenant hash-chained audit log. UPDATE/DELETE are blocked by database triggers;
tampering with any stored row breaks the chain, which verify() detects. Production also streams each
record to WORM storage / the SIEM so a database-level attacker cannot rewrite the whole chain unnoticed."""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from .db import DB

GENESIS = "0" * 64
PLATFORM = "_platform"  # chain for events that belong to no tenant (e.g. failed logins for unknown accounts)


def _record_hash(prev: str, tenant_id: str, ts: int, actor: str, action: str, target: str, detail: str) -> str:
    payload = json.dumps([prev, tenant_id, ts, actor, action, target, detail], separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def append(db: DB, tenant_id: str | None, actor: str, action: str, target: str = "", detail: dict[str, Any] | None = None,
           now: int | None = None) -> None:
    """Call inside the same transaction as the change being audited."""
    tid = tenant_id or PLATFORM
    ts = int(now or time.time())
    det = json.dumps(_scrub(detail or {}), separators=(",", ":"), sort_keys=True, ensure_ascii=True)[:4000]
    last = db.one("SELECT hash FROM audit_log WHERE tenant_id = ? ORDER BY seq DESC LIMIT 1", (tid,))
    prev = last["hash"] if last else GENESIS
    h = _record_hash(prev, tid, ts, actor[:120], action[:80], target[:200], det)
    db.x("INSERT INTO audit_log (tenant_id, ts, actor, action, target, detail, prev_hash, hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
         (tid, ts, actor[:120], action[:80], target[:200], det, prev, h))


_SECRET_KEYS = {"password", "token", "refresh_token", "access_token", "totp", "secret", "code", "authorization"}


def _scrub(d: dict[str, Any]) -> dict[str, Any]:
    return {k: ("[redacted]" if k.lower() in _SECRET_KEYS else v) for k, v in d.items()}


def verify(db: DB, tenant_id: str) -> dict[str, Any]:
    prev = GENESIS
    n = 0
    for r in db.q("SELECT seq, tenant_id, ts, actor, action, target, detail, prev_hash, hash FROM audit_log WHERE tenant_id = ? ORDER BY seq", (tenant_id,)):
        if r["prev_hash"] != prev or _record_hash(prev, r["tenant_id"], r["ts"], r["actor"], r["action"], r["target"], r["detail"]) != r["hash"]:
            return {"ok": False, "records": n, "broken_at_seq": r["seq"]}
        prev = r["hash"]
        n += 1
    return {"ok": True, "records": n, "head": prev}
