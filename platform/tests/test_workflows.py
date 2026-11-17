"""Two-person rule, evidence chain of custody, audit chain, retention."""
import hashlib
import time

import pytest

from airspace import audit, retention

JPEG = b"\xff\xd8\xff\xe0" + b"evidence-bytes" * 50


def _approve(env, t, req_role, body, approver="supervisor2"):
    r = env.post("/api/v1/approvals", headers=env.h(t, req_role), json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"], env.post(f"/api/v1/approvals/{r.json()['id']}/approve", headers=env.h(t, approver))


# ---------- two-person rule ----------
def test_requester_cannot_self_approve(env):
    r = env.post("/api/v1/approvals", headers=env.h(env.A, "supervisor"),
                 json={"action": "disable_sensor", "sensor_id": env.A.sensors["acoustic"], "reason": "maintenance"})
    aid = r.json()["id"]
    assert env.post(f"/api/v1/approvals/{aid}/approve", headers=env.h(env.A, "supervisor")).status_code == 403
    db = env.db()
    assert db.one("SELECT enabled FROM sensors WHERE id = ?", (env.A.sensors["acoustic"],))["enabled"] == 1
    db.close()


def test_second_supervisor_executes(env):
    _, r = _approve(env, env.A, "supervisor", {"action": "set_alert_threshold", "site_id": env.A.site, "threshold": 0.9, "reason": "tuning"})
    assert r.status_code == 200 and r.json()["status"] == "executed"
    db = env.db()
    assert db.one("SELECT alert_threshold FROM sites WHERE id = ?", (env.A.site,))["alert_threshold"] == 0.9
    db.close()


def test_operator_cannot_request_or_approve(env):
    assert env.post("/api/v1/approvals", headers=env.h(env.A, "operator"),
                    json={"action": "set_retention", "retention_days": 30, "reason": "x"}).status_code == 403


def test_approval_expires(env):
    r = env.post("/api/v1/approvals", headers=env.h(env.A, "supervisor"),
                 json={"action": "set_retention", "retention_days": 90, "reason": "x"})
    aid = r.json()["id"]
    db = env.db()
    db.x("UPDATE approvals SET expires_at = ? WHERE id = ?", (int(time.time()) - 1, aid))
    db.close()
    assert env.post(f"/api/v1/approvals/{aid}/approve", headers=env.h(env.A, "supervisor2")).status_code == 409


def test_approval_cannot_be_executed_twice(env):
    aid, r = _approve(env, env.A, "supervisor", {"action": "enable_sensor", "sensor_id": env.A.sensors["rf"], "reason": "x"})
    assert r.status_code == 200
    assert env.post(f"/api/v1/approvals/{aid}/approve", headers=env.h(env.A, "tenant_admin")).status_code == 409


def test_bad_threshold_rejected(env):
    r = env.post("/api/v1/approvals", headers=env.h(env.A, "supervisor"),
                 json={"action": "set_alert_threshold", "site_id": env.A.site, "threshold": 0.01, "reason": "x"})
    assert r.status_code == 422


# ---------- evidence ----------
@pytest.fixture
def stored_evidence(env, new_alert):
    aid = new_alert()
    r, _ = env.signed(env.A, f"/ingest/v1/evidence?alert_id={aid}", JPEG)
    assert r.status_code == 201, r.text
    return aid, r.json()["id"]


def test_evidence_rejects_non_images_and_oversize(env, new_alert):
    aid = new_alert()
    assert env.signed(env.A, f"/ingest/v1/evidence?alert_id={aid}", b"<svg onload=alert(1)>")[0].status_code == 422
    big = b"\xff\xd8\xff" + b"0" * (env.s.max_evidence_bytes + 10)
    assert env.signed(env.A, f"/ingest/v1/evidence?alert_id={aid}", big)[0].status_code == 413


def test_evidence_cross_tenant_alert(env, new_alert):
    aid_b = new_alert(env.B)
    assert env.signed(env.A, f"/ingest/v1/evidence?alert_id={aid_b}", JPEG)[0].status_code == 404


def test_evidence_metadata_signed(env, stored_evidence):
    _, eid = stored_evidence
    m = env.get(f"/api/v1/evidence/{eid}", headers=env.h(env.A, "operator")).json()
    assert m["sha256"] == hashlib.sha256(JPEG).hexdigest() and m["signature_valid"] is True
    assert "storage_key" not in m


def test_evidence_export_needs_two_people_and_one_time_grant(env, stored_evidence):
    _, eid = stored_evidence
    # no grant -> nothing
    assert env.get(f"/api/v1/evidence/{eid}/download", headers=env.h(env.A, "supervisor")).status_code == 404
    _, r = _approve(env, env.A, "supervisor", {"action": "export_evidence", "evidence_id": eid, "reason": "court request 22-118"})
    grant = r.json()["grant_for_requester"]
    # the approver can't use the requester's grant
    assert env.get(f"/api/v1/evidence/{eid}/download", headers={**env.h(env.A, "supervisor2"), "X-Export-Grant": grant}).status_code == 404
    d = env.get(f"/api/v1/evidence/{eid}/download", headers={**env.h(env.A, "supervisor"), "X-Export-Grant": grant})
    assert d.status_code == 200 and d.content == JPEG
    assert d.headers["content-type"] == "image/jpeg" and d.headers["x-content-type-options"] == "nosniff"
    assert d.headers["content-disposition"].startswith("attachment;")
    # single use
    assert env.get(f"/api/v1/evidence/{eid}/download", headers={**env.h(env.A, "supervisor"), "X-Export-Grant": grant}).status_code == 404


def test_evidence_file_tamper_detected(env, stored_evidence):
    _, eid = stored_evidence
    db = env.db()
    key = db.one("SELECT storage_key FROM evidence WHERE id = ?", (eid,))["storage_key"]
    db.close()
    import os
    with open(os.path.join(env.s.evidence_dir, key[:2], key), "r+b") as fh:
        fh.write(b"\xff\xd8\xffX")
        fh.seek(20)
        fh.write(b"TAMPERED")
    _, r = _approve(env, env.A, "supervisor", {"action": "export_evidence", "evidence_id": eid, "reason": "x"})
    d = env.get(f"/api/v1/evidence/{eid}/download", headers={**env.h(env.A, "supervisor"), "X-Export-Grant": r.json()["grant_for_requester"]})
    assert d.status_code == 422


# ---------- audit ----------
def test_audit_chain_verifies_and_detects_tampering(env, new_alert):
    new_alert()
    h = env.h(env.A, "tenant_admin")
    v = env.get("/api/v1/audit/verify", headers=h).json()
    assert v["ok"] and v["records"] > 3
    db = env.db()
    with pytest.raises(Exception):
        db.x("UPDATE audit_log SET action = 'nothing' WHERE tenant_id = ?", (env.A.id,))
    with pytest.raises(Exception):
        db.x("DELETE FROM audit_log WHERE tenant_id = ?", (env.A.id,))
    # simulate an attacker with raw DB access who drops the triggers and edits a record
    db.conn.execute("DROP TRIGGER audit_no_update")
    db.x("UPDATE audit_log SET actor = 'someone-else' WHERE seq = (SELECT MIN(seq) FROM audit_log WHERE tenant_id = ?)", (env.A.id,))
    res = audit.verify(db, env.A.id)
    db.conn.execute("CREATE TRIGGER audit_no_update BEFORE UPDATE ON audit_log BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END")
    db.close()
    assert res["ok"] is False


def test_audit_never_contains_secrets(env):
    db = env.db()
    rows = db.q("SELECT detail FROM audit_log")
    db.close()
    blob = " ".join(r["detail"] for r in rows)
    import re
    for pat in (r"rt_[A-Za-z0-9_-]{20,}", r"ast_[A-Za-z0-9_-]{20,}", r"eg_[A-Za-z0-9_-]{20,}", r"scrypt\$", r"eyJ[A-Za-z0-9_-]{10,}\."):
        assert not re.search(pat, blob), pat


# ---------- retention ----------
def test_retention_purges_but_respects_legal_hold(env, new_alert):
    old = int(time.time()) - 200 * 86400
    a_keep, a_purge = new_alert(), new_alert()
    for aid in (a_keep, a_purge):
        env.post(f"/api/v1/alerts/{aid}/resolve", headers=env.h(env.A, "operator"), json={"note": "handled"})
    db = env.db()
    db.x("UPDATE alerts SET created_at = ? WHERE id IN (?, ?)", (old, a_keep, a_purge))
    db.x("UPDATE tracks SET observed_at = ? WHERE tenant_id = ?", (old, env.A.id))
    db.close()
    assert env.post(f"/api/v1/alerts/{a_keep}/legal-hold", headers=env.h(env.A, "tenant_admin"), json={"hold": True}).status_code == 200
    db = env.db()
    totals = retention.purge(env.s, db)
    ids = {r["id"] for r in db.q("SELECT id FROM alerts WHERE tenant_id = ?", (env.A.id,))}
    db.close()
    assert a_keep in ids and a_purge not in ids and totals["tracks"] > 0
