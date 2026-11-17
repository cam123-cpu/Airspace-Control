# Airspace Control platform (reference implementation)

A secure, multi-tenant backend for counter-drone deployments. It serves operator dashboards, edge sensor boxes, and AI agents through MCP. For the design, read [ARCHITECTURE.md](../docs/ARCHITECTURE.md). For the security tests and how to report a vulnerability, read `SECURITY.md`.

```
airspace/
  app.py              HTTP routes (thin): authenticate → validate → service layer
  db.py               migrations + DB wrapper; literal SQL with bound parameters only
  auth.py             scrypt+pepper, TOTP, EdDSA JWT, rotating refresh tokens, API tokens
  sessions.py         login / refresh / logout, lockout, MFA enforcement
  rbac.py             deny-by-default permission table, two-person action list
  approvals.py        two-person rule (request → different supervisor approves → atomic execute)
  alerts.py           tenant-scoped reads and alert actions (shared by HTTP and MCP)
  ingest.py           Ed25519-signed device ingest, replay cache, idempotency
  evidence.py         hashed + server-signed evidence, one-time export grants
  audit.py            per-tenant hash-chained, append-only audit log
  retention.py        180-day purge with legal hold
  admin.py, cli.py    users / devices / API tokens; bootstrap CLI
  mcp_server.py       MCP wrapper (stdio) with scoped tokens and human-confirmed writes
  security_headers.py ASGI hardening: headers, CORS allowlist, body caps, safe errors, JSON logs
  schemas.py          strict, bounded pydantic input models
  crypto.py           AES-256-GCM encryption at rest (TOTP seeds, Remote ID text, notes, evidence files), key ring + rewrap
  challenge.py        bot protection: proof-of-work (built in) or Turnstile/hCaptcha after repeated failed logins
  outputs.py          response allowlists; unknown or secret-shaped fields never leave the API
  net.py              client IP and scheme from trusted proxies only
tests/                160 tests (pytest): SQL-safety scan, injection probes, fuzzing, route and model inventories,
                      encryption, cookie/CSRF, bot challenge, HTTPS, repo secret scan, Postgres RLS (needs a server)
ops/postgres/         production schema with row-level security, roles and SECURITY DEFINER auth lookups
ops/                  Dockerfile (distroless, non-root), compose, key-rotation runbook
```

## Run locally
```bash
pip install -r requirements-dev.txt
python -m pytest -q tests/                      # 160 tests (RLS tests run when AIRSPACE_PG_DSN is set)
export AIRSPACE_DB_PATH=dev.db                  # dev mode generates ephemeral keys
python -m airspace.cli bootstrap --tenant "Demo DOC" --email admin@demo.example
python -m airspace.cli create-site --tenant tn_... --name "Site A"
uvicorn airspace.asgi:app --port 8080 --no-server-header --no-proxy-headers
```

## API

| Method | Path | Who |
|---|---|---|
| POST | `/api/v1/auth/challenge` | anyone: a proof-of-work challenge, needed after repeated failed logins |
| POST | `/api/v1/auth/login` `{email, password, totp, challenge?}` | anyone (rate-limited; 428 + challenge after 3 failures) |
| POST | `/api/v1/auth/refresh` · `/api/v1/auth/logout` | holder of a refresh token |
| GET | `/api/v1/me` | any user |
| GET | `/api/v1/sites` · `/api/v1/sites/{id}/sensors` | viewer+ |
| GET | `/api/v1/alerts?status=&site_id=&limit=&cursor=` · `/api/v1/alerts/{id}` | viewer+ |
| POST | `/api/v1/alerts/{id}/ack` · `/resolve {note}` | operator+ |
| POST | `/api/v1/alerts/{id}/mcp-confirmation` | operator+ (one-time code for an AI agent) |
| POST | `/api/v1/alerts/{id}/legal-hold {hold}` | tenant_admin |
| GET | `/api/v1/tracks?site_id=&since=&limit=` | viewer+ |
| GET | `/api/v1/evidence/{id}` · `/download` (header `X-Export-Grant`) | operator+ (download needs an approved export) |
| POST/GET | `/api/v1/approvals` · POST `/{id}/approve` · `/{id}/reject` | supervisor+ (approver ≠ requester) |
| POST | `/api/v1/users` · `/api/v1/devices` · `/api/v1/api-tokens` · `/api/v1/api-tokens/{id}/revoke` | tenant_admin |
| GET | `/api/v1/audit?after=` · `/api/v1/audit/verify` | tenant_admin, platform_auditor (`?tenant_id=`) |
| POST | `/ingest/v1/telemetry` · `/ingest/v1/evidence?alert_id=` | edge devices (Ed25519-signed headers) |

Device signature: `Ed25519(METHOD\nPATH?QUERY\nTIMESTAMP\nNONCE\nsha256_hex(body))` (the full request target is signed) in `X-Signature`, with `X-Device-Id`, `X-Timestamp` (±60 s) and `X-Nonce` (single use); see `ingest.py`.

## Browser sessions

Send `X-Session-Mode: cookie` with login, refresh and logout.

- **Refresh token:** set as `__Host-airspace_rt`, which is Secure, HttpOnly, SameSite=Strict and Path=/. It is never returned in JSON.
- **CSRF token:** set as `__Host-airspace_csrf` and also returned in the body.
- **Refresh and logout:** need an allowlisted `Origin` header and `X-CSRF-Token`. The token must equal the CSRF cookie and the value bound to the refresh token.
- **Same-site only:** use cookie mode only for a console on the same site as the API. Machine clients keep using body tokens.

## MCP (AI agents)
A tenant_admin creates a scoped token: `POST /api/v1/api-tokens {"name":"soc-agent","role":"viewer","allowed_tools":["list_active_alerts","get_alert"],"ttl_hours":8}`. Then configure the MCP client:
```json
{"mcpServers": {"airspace": {"command": "python", "args": ["-m", "airspace.mcp_server"],
  "env": {"AIRSPACE_MCP_TOKEN": "ast_…", "AIRSPACE_DB_PATH": "/data/airspace.db"}}}}
```
The tools are `list_active_alerts`, `get_alert`, `site_sensor_health` and `search_tracks`, plus `acknowledge_alert` for operator tokens. `acknowledge_alert` only works with a code a human operator issued for that specific alert. Text from sensors or users comes back wrapped in `UNTRUSTED_DATA` markers.
In production the MCP server calls the HTTP API with the token, instead of opening the database directly, so it runs on the agent host with no DB credentials. The service-layer calls in this repo are already the same functions the API uses.
