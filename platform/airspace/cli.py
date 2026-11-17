"""Operator CLI (run inside the deployment, never exposed over HTTP).

  python -m airspace.cli gen-key                      # new Ed25519 key (base64 PEM) for AIRSPACE_JWT_KEYS / EVIDENCE_KEY
  python -m airspace.cli gen-pepper
  python -m airspace.cli bootstrap --tenant "Georgia DOC" --email admin@example.gov
  python -m airspace.cli create-site --tenant tn_... --name "Site A" --sensors radar,rf,acoustic,camera_ir
  python -m airspace.cli create-auditor --email auditor@example.gov
  python -m airspace.cli purge                         # retention job (cron / k8s CronJob, daily)
  python -m airspace.cli gen-data-key                  # new AES-256 data key for AIRSPACE_DATA_KEYS
  python -m airspace.cli rewrap-data                   # re-encrypt everything under the newest data key
"""
from __future__ import annotations

import argparse
import base64
import getpass
import json
import secrets
import sys

from . import admin, crypto, retention
from .config import generate_data_key_b64, generate_key_b64, load_settings
from .db import DB, migrate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="airspace")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gen-key")
    sub.add_parser("gen-pepper")
    b = sub.add_parser("bootstrap")
    b.add_argument("--tenant", required=True)
    b.add_argument("--email", required=True)
    cs = sub.add_parser("create-site")
    cs.add_argument("--tenant", required=True)
    cs.add_argument("--name", required=True)
    cs.add_argument("--sensors", default="radar,rf,acoustic,camera_vis,camera_ir,remote_id")
    ca = sub.add_parser("create-auditor")
    ca.add_argument("--email", required=True)
    sub.add_parser("purge")
    sub.add_parser("gen-data-key")
    sub.add_parser("rewrap-data")
    a = ap.parse_args(argv)

    if a.cmd == "gen-key":
        print(generate_key_b64())
        return 0
    if a.cmd == "gen-data-key":
        print(generate_data_key_b64())
        return 0
    if a.cmd == "gen-pepper":
        print(base64.b64encode(secrets.token_bytes(32)).decode())
        return 0
    s = load_settings()
    migrate(s.db_path)
    crypto.install(s.keyring())
    db = DB(s.db_path)
    try:
        if a.cmd == "bootstrap":
            pw = getpass.getpass("Initial admin password (12+ chars): ")
            print(json.dumps(admin.bootstrap_tenant(s, db, a.tenant, a.email, pw), indent=2))
        elif a.cmd == "create-site":
            kinds = [k.strip() for k in a.sensors.split(",") if k.strip()]
            print(json.dumps(admin.create_site(db, a.tenant, a.name, kinds), indent=2))
        elif a.cmd == "create-auditor":
            pw = getpass.getpass("Auditor password (12+ chars): ")
            print(json.dumps(admin.create_platform_auditor(s, db, a.email, pw), indent=2))
        elif a.cmd == "purge":
            print(json.dumps(retention.purge(s, db)))
        elif a.cmd == "rewrap-data":
            print(json.dumps(rewrap(s, db)))
    finally:
        db.close()
    return 0


def rewrap(s, db: DB) -> dict:
    """Key rotation: decrypt every encrypted value with whichever key wrote it and re-encrypt under the current key.
    After it reports zero values under old keys, the old key can be removed from AIRSPACE_DATA_KEYS."""
    import os
    from .evidence import _store_path
    ring, n = s.keyring(), {"users.totp_secret": 0, "alerts": 0, "evidence_files": 0}
    with db.tx():
        for u in db.q("SELECT id, tenant_id, totp_secret FROM users WHERE totp_secret IS NOT NULL"):
            ad = crypto.aad("users.totp_secret", u["tenant_id"], u["id"])
            if ring.field_kid(u["totp_secret"]) != ring.current:
                db.x("UPDATE users SET totp_secret = ? WHERE id = ?", (ring.encrypt_field(ring.decrypt_field(u["totp_secret"], ad), ad), u["id"]))
                n["users.totp_secret"] += 1
        for r in db.q("SELECT id, tenant_id, remote_id_text, resolution_note FROM alerts"):
            vals = []
            for col in ("remote_id_text", "resolution_note"):
                ad = crypto.aad("alerts." + col, r["tenant_id"], r["id"])
                v = r[col]
                vals.append(ring.encrypt_field(ring.decrypt_field(v, ad), ad) if v is not None and ring.field_kid(v) != ring.current else v)
            if vals != [r["remote_id_text"], r["resolution_note"]]:
                db.x("UPDATE alerts SET remote_id_text = ?, resolution_note = ? WHERE id = ?", (vals[0], vals[1], r["id"]))
                n["alerts"] += 1
    for e in db.q("SELECT id, tenant_id, storage_key FROM evidence"):
        path = _store_path(s, e["storage_key"])
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            blob = fh.read()
        if ring.blob_kid(blob) == ring.current:
            continue
        ad = crypto.aad("evidence", e["tenant_id"], e["id"])
        tmp = path + ".rewrap"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(ring.encrypt_blob(ring.decrypt_blob(blob, ad), ad))
        os.replace(tmp, path)
        n["evidence_files"] += 1
    return {"rewrapped": n, "current_key": ring.current}


if __name__ == "__main__":
    sys.exit(main())
