"""Retention: purge tracks, resolved alerts and evidence older than the tenant's retention period
(default 180 days, the SAFER SKIES rule's limit on keeping drone-communication records) unless under legal hold.
The audit log is never purged by this job (separate, longer retention under the CJIS policy)."""
from __future__ import annotations

import time

from . import audit
from .config import Settings
from .db import DB
from .evidence import delete_file


def purge(s: Settings, db: DB, now: int | None = None) -> dict:
    now = int(now or time.time())
    totals = {"tracks": 0, "alerts": 0, "evidence": 0}
    for t in db.q("SELECT id, retention_days FROM tenants ORDER BY id"):
        cutoff = now - int(t["retention_days"]) * 86400
        files = db.q("SELECT e.storage_key FROM evidence e JOIN alerts a ON a.tenant_id = e.tenant_id AND a.id = e.alert_id "
                     "WHERE e.tenant_id = ? AND e.created_at < ? AND e.legal_hold = 0 AND a.legal_hold = 0", (t["id"], cutoff))
        with db.tx():
            ne = db.x("DELETE FROM evidence WHERE tenant_id = ? AND created_at < ? AND legal_hold = 0 "
                      "AND alert_id IN (SELECT id FROM alerts WHERE tenant_id = ? AND legal_hold = 0)", (t["id"], cutoff, t["id"]))
            db.x("DELETE FROM confirmations WHERE tenant_id = ? AND expires_at < ?", (t["id"], now))
            na = db.x("DELETE FROM alerts WHERE tenant_id = ? AND created_at < ? AND legal_hold = 0 AND status = 'resolved' "
                      "AND id NOT IN (SELECT alert_id FROM evidence WHERE tenant_id = ?)", (t["id"], cutoff, t["id"]))
            nt = db.x("DELETE FROM tracks WHERE tenant_id = ? AND observed_at < ?", (t["id"], cutoff))
            audit.append(db, t["id"], "system:retention", "retention.purge", t["id"], {"tracks": nt, "alerts": na, "evidence": ne, "cutoff": cutoff}, now)
        for f in files:
            delete_file(s, f["storage_key"])
        totals["tracks"] += nt
        totals["alerts"] += na
        totals["evidence"] += ne
    db.x("DELETE FROM nonces WHERE expires_at < ?", (now,))
    db.x("DELETE FROM idempotency WHERE created_at < ?", (now - 86400,))
    return totals
