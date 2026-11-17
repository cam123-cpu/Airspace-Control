"""Two-person rule. Sensitive actions are requested by one supervisor and executed only when a DIFFERENT
supervisor/tenant_admin of the same tenant approves before the request expires."""
from __future__ import annotations

import json
import time
import uuid

from . import audit
from .auth import Principal
from .config import Settings
from .db import DB
from .errors import Conflict, ForbiddenErr, Invalid, NotFound
from .evidence import create_export_grant
from .rbac import require
from .schemas import ApprovalIn

_REQUIRED = {
    "disable_sensor": ("sensor_id",), "enable_sensor": ("sensor_id",),
    "set_alert_threshold": ("site_id", "threshold"), "set_retention": ("retention_days",),
    "export_evidence": ("evidence_id",),
}


def request(s: Settings, db: DB, p: Principal, body: dict, now: int | None = None) -> dict:
    require(p, "approvals.request")
    now = int(now or time.time())
    a = ApprovalIn.model_validate(body)
    params = {k: getattr(a, k) for k in _REQUIRED[a.action]}
    if any(v is None for v in params.values()):
        raise Invalid(f"{a.action} needs {', '.join(_REQUIRED[a.action])}")
    # the target must exist in the requester's tenant
    if "sensor_id" in params and not db.one("SELECT id FROM sensors WHERE tenant_id = ? AND id = ?", (p.tenant_id, params["sensor_id"])):
        raise NotFound()
    if "site_id" in params and not db.one("SELECT id FROM sites WHERE tenant_id = ? AND id = ?", (p.tenant_id, params["site_id"])):
        raise NotFound()
    if "evidence_id" in params and not db.one("SELECT id FROM evidence WHERE tenant_id = ? AND id = ?", (p.tenant_id, params["evidence_id"])):
        raise NotFound()
    params["reason"] = a.reason
    aid = uuid.uuid4().hex
    with db.tx():
        db.x("INSERT INTO approvals (id, tenant_id, action, params, requested_by, requested_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
             (aid, p.tenant_id, a.action, json.dumps(params), p.id, now, now + s.approval_ttl_s))
        audit.append(db, p.tenant_id, p.actor, "approval.requested", aid, {"action": a.action}, now)
    return {"id": aid, "status": "pending", "expires_at": now + s.approval_ttl_s}


def list_pending(db: DB, p: Principal) -> list[dict]:
    require(p, "approvals.read")
    return db.q("SELECT id, action, params, requested_by, requested_at, expires_at, status FROM approvals WHERE tenant_id = ? AND status = 'pending' ORDER BY requested_at DESC LIMIT 100",
                (p.tenant_id,))


def decide(s: Settings, db: DB, p: Principal, approval_id: str, approve: bool, now: int | None = None) -> dict:
    require(p, "approvals.approve")
    now = int(now or time.time())
    with db.tx():
        ap = db.one("SELECT id, action, params, requested_by, expires_at, status FROM approvals WHERE tenant_id = ? AND id = ?", (p.tenant_id, approval_id))
        if not ap:
            raise NotFound()
        if ap["status"] != "pending":
            raise Conflict("already decided")
        if ap["expires_at"] <= now:
            db.x("UPDATE approvals SET status = 'expired' WHERE tenant_id = ? AND id = ?", (p.tenant_id, approval_id))
            raise Conflict("approval expired")
        if ap["requested_by"] == p.id:
            raise ForbiddenErr("the requester cannot approve their own request")
        result: dict = {}
        if approve:
            result = _execute(db, p.tenant_id, ap["action"], json.loads(ap["params"]), ap["requested_by"], now)
        db.x("UPDATE approvals SET status = ?, decided_by = ?, decided_at = ? WHERE tenant_id = ? AND id = ?",
             ("executed" if approve else "rejected", p.id, now, p.tenant_id, approval_id))
        audit.append(db, p.tenant_id, p.actor, "approval.executed" if approve else "approval.rejected", approval_id,
                     {"action": ap["action"], "requested_by": ap["requested_by"]}, now)
    return {"id": approval_id, "status": "executed" if approve else "rejected", **result}


def _execute(db: DB, tenant_id: str, action: str, params: dict, requester: str, now: int) -> dict:
    if action in ("disable_sensor", "enable_sensor"):
        db.x("UPDATE sensors SET enabled = ? WHERE tenant_id = ? AND id = ?", (0 if action == "disable_sensor" else 1, tenant_id, params["sensor_id"]))
        return {}
    if action == "set_alert_threshold":
        db.x("UPDATE sites SET alert_threshold = ? WHERE tenant_id = ? AND id = ?", (float(params["threshold"]), tenant_id, params["site_id"]))
        return {}
    if action == "set_retention":
        db.x("UPDATE tenants SET retention_days = ? WHERE id = ?", (int(params["retention_days"]), tenant_id))
        return {}
    if action == "export_evidence":
        # the one-time grant goes to the REQUESTER; it is delivered through their approvals view, never logged
        grant = create_export_grant(db, tenant_id, params["evidence_id"], requester, now)
        return {"grant_for_requester": grant}
    raise Invalid("unknown action")
