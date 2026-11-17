"""Alert, track, site and sensor queries. Shared by the HTTP API and the MCP server so both enforce the same
tenant scoping and permissions. tenant_id always comes from the authenticated Principal."""
from __future__ import annotations

import hashlib
import secrets
import time
from typing import Any

from . import audit, crypto
from .auth import Principal
from .db import DB
from .errors import Conflict, Invalid, NotFound
from .rbac import require

MAX_PAGE = 100


def _open(r: dict, tenant_id: str) -> dict:
    """Decrypt the encrypted columns of an alert row. A row that fails authentication is reported, not hidden."""
    ring = crypto.active()
    for col in ("remote_id_text", "resolution_note"):
        try:
            r[col] = ring.decrypt_field(r.get(col), crypto.aad("alerts." + col, tenant_id, r["id"]))
        except crypto.DecryptError:
            r[col] = "[unreadable: integrity check failed]"
    return r

def _limit(n: Any) -> int:
    try:
        n = int(n)
    except (TypeError, ValueError):
        raise Invalid("limit must be an integer")
    return max(1, min(MAX_PAGE, n))


def _cursor(c: Any) -> int:
    if c in (None, ""):
        return 2 ** 62
    try:
        v = int(c)
    except (TypeError, ValueError):
        raise Invalid("bad cursor")
    if not 0 < v < 2 ** 62:
        raise Invalid("bad cursor")
    return v


def list_sites(db: DB, p: Principal) -> list[dict]:
    require(p, "sites.read")
    return db.q("SELECT id, name, alert_threshold, created_at FROM sites WHERE tenant_id = ? ORDER BY name LIMIT 500", (p.tenant_id,))


def site_sensors(db: DB, p: Principal, site_id: str) -> list[dict]:
    require(p, "sensors.read")
    if not db.one("SELECT id FROM sites WHERE tenant_id = ? AND id = ?", (p.tenant_id, site_id)):
        raise NotFound()
    return db.q("SELECT id, kind, enabled, status, last_seen FROM sensors WHERE tenant_id = ? AND site_id = ? ORDER BY kind, id", (p.tenant_id, site_id))


def list_alerts(db: DB, p: Principal, status: str | None = None, site_id: str | None = None, limit: Any = 50, cursor: Any = None) -> dict:
    require(p, "alerts.read")
    lim, cur = _limit(limit), _cursor(cursor)
    if status not in (None, "", "open", "acknowledged", "resolved", "active"):
        raise Invalid("bad status")
    st = status or ""
    sid = site_id or ""
    # one fixed statement; optional filters are expressed as parameters, never by editing SQL text
    rows = db.q(
        "SELECT id, site_id, ext_track_id, severity, title, remote_id_text, status, created_at, ack_by, ack_at, resolved_by, resolved_at, resolution_note, legal_hold, seq "
        "FROM alerts WHERE tenant_id = ? AND seq < ? "
        "AND (? = '' OR status = ? OR (? = 'active' AND status IN ('open','acknowledged'))) "
        "AND (? = '' OR site_id = ?) ORDER BY seq DESC LIMIT ?",
        (p.tenant_id, cur, st, st, st, sid, sid, lim))
    nxt = rows[-1]["seq"] if len(rows) == lim else None
    for r in rows:
        r.pop("seq", None)
        _open(r, p.tenant_id)
    return {"items": rows, "next_cursor": nxt}


def get_alert(db: DB, p: Principal, alert_id: str) -> dict:
    require(p, "alerts.read")
    r = db.one(
        "SELECT id, site_id, ext_track_id, severity, title, remote_id_text, status, created_at, ack_by, ack_at, resolved_by, resolved_at, resolution_note, legal_hold "
        "FROM alerts WHERE tenant_id = ? AND id = ?", (p.tenant_id, alert_id))
    if not r:
        raise NotFound()
    _open(r, p.tenant_id)
    r["evidence"] = db.q("SELECT id, sha256, size, content_type, created_at FROM evidence WHERE tenant_id = ? AND alert_id = ? ORDER BY created_at", (p.tenant_id, alert_id))
    return r


def acknowledge(db: DB, p: Principal, alert_id: str, now: int | None = None, via: str = "api") -> dict:
    require(p, "alerts.ack")
    now = int(now or time.time())
    with db.tx():
        n = db.x("UPDATE alerts SET status = 'acknowledged', ack_by = ?, ack_at = ? WHERE tenant_id = ? AND id = ? AND status = 'open'",
                 (p.actor, now, p.tenant_id, alert_id))
        if n == 0:
            if not db.one("SELECT id FROM alerts WHERE tenant_id = ? AND id = ?", (p.tenant_id, alert_id)):
                raise NotFound()
            raise Conflict("alert is not open")
        audit.append(db, p.tenant_id, p.actor, "alert.ack", alert_id, {"via": via}, now)
    return {"id": alert_id, "status": "acknowledged"}


def resolve(db: DB, p: Principal, alert_id: str, note: str, now: int | None = None) -> dict:
    require(p, "alerts.resolve")
    now = int(now or time.time())
    with db.tx():
        n = db.x("UPDATE alerts SET status = 'resolved', resolved_by = ?, resolved_at = ?, resolution_note = ? "
                 "WHERE tenant_id = ? AND id = ? AND status IN ('open','acknowledged')",
                 (p.actor, now, crypto.active().encrypt_field(note, crypto.aad("alerts.resolution_note", p.tenant_id, alert_id)), p.tenant_id, alert_id))
        if n == 0:
            if not db.one("SELECT id FROM alerts WHERE tenant_id = ? AND id = ?", (p.tenant_id, alert_id)):
                raise NotFound()
            raise Conflict("alert already resolved")
        audit.append(db, p.tenant_id, p.actor, "alert.resolve", alert_id, {"note_len": len(note)}, now)
    return {"id": alert_id, "status": "resolved"}


def set_legal_hold(db: DB, p: Principal, alert_id: str, hold: bool, now: int | None = None) -> dict:
    require(p, "alerts.legal_hold")
    with db.tx():
        if db.x("UPDATE alerts SET legal_hold = ? WHERE tenant_id = ? AND id = ?", (1 if hold else 0, p.tenant_id, alert_id)) == 0:
            raise NotFound()
        db.x("UPDATE evidence SET legal_hold = ? WHERE tenant_id = ? AND alert_id = ?", (1 if hold else 0, p.tenant_id, alert_id))
        audit.append(db, p.tenant_id, p.actor, "alert.legal_hold", alert_id, {"hold": hold}, now)
    return {"id": alert_id, "legal_hold": hold}


def list_tracks(db: DB, p: Principal, site_id: str, since: Any = 0, limit: Any = 100) -> list[dict]:
    require(p, "tracks.read")
    try:
        since_i = int(since or 0)
    except (TypeError, ValueError):
        raise Invalid("bad since")
    if not db.one("SELECT id FROM sites WHERE tenant_id = ? AND id = ?", (p.tenant_id, site_id)):
        raise NotFound()
    return db.q("SELECT id, ext_track_id, cls, confidence, x, y, z, observed_at FROM tracks WHERE tenant_id = ? AND site_id = ? AND observed_at >= ? "
                "ORDER BY observed_at DESC LIMIT ?", (p.tenant_id, site_id, since_i, _limit(limit)))


# ---- human confirmation codes for agent (MCP) writes ----
def new_mcp_confirmation(db: DB, p: Principal, alert_id: str, now: int | None = None) -> dict:
    """A human operator issues a short-lived, single-use code that lets an AI agent acknowledge ONE alert."""
    require(p, "alerts.mcp_confirm")
    now = int(now or time.time())
    code = "".join(secrets.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8))
    with db.tx():
        if not db.one("SELECT id FROM alerts WHERE tenant_id = ? AND id = ? AND status = 'open'", (p.tenant_id, alert_id)):
            raise NotFound()
        db.x("INSERT INTO confirmations (code_hash, tenant_id, alert_id, created_by, expires_at) VALUES (?, ?, ?, ?, ?)",
             (hashlib.sha256(code.encode()).hexdigest(), p.tenant_id, alert_id, p.actor, now + 300))
        audit.append(db, p.tenant_id, p.actor, "mcp.confirmation_issued", alert_id, {}, now)
    return {"code": code, "expires_in": 300}


def consume_mcp_confirmation(db: DB, p: Principal, alert_id: str, code: str, now: int | None = None) -> bool:
    now = int(now or time.time())
    h = hashlib.sha256(str(code).encode()).hexdigest()
    return db.x("UPDATE confirmations SET used_at = ? WHERE code_hash = ? AND tenant_id = ? AND alert_id = ? AND used_at IS NULL AND expires_at > ?",
                (now, h, p.tenant_id, alert_id, now)) == 1
