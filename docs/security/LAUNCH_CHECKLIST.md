# Launch security checklist

The 20 common pre-launch checks, applied to the Airspace Command platform (`platform/`). Each row says what's in place and where to verify it. "Done + test" means an automated test fails if the control regresses. Every test listed runs in CI on each push.

| # | Check | Status | What is in place | Where |
|---|---|---|---|---|
| 1 | Hide API keys | Done + test | No secrets in code or config files; all keys come from environment variables or `*_FILE` mounts; production refuses to start without them; the static web apps hold no keys at all. | `airspace/config.py`, `tests/test_config.py`, `tests/test_repo_secrets.py` |
| 2 | Purge Git secrets | Done + test | Repo scan for private keys, cloud and SaaS tokens, JWTs and hard-coded platform secrets; gitleaks over full history in CI and as a pre-commit hook; pentest keys are generated per run. | `tests/test_repo_secrets.py`, `.github/workflows/security.yml`, `.pre-commit-config.yaml`, `platform/pentest/env.sh` |
| 3 | Use public DB key | N/A by design | No client ever talks to the database. Browsers and AI agents only reach the API, which connects as a least-privilege role with no BYPASSRLS and no schema rights. There is no anon or service key to leak. | `platform/ops/postgres/schema.sql` (roles), `tests/test_postgres_rls.py::test_app_role_cannot_bypass_rls` |
| 4 | Enable row-level security | Done + test | RLS is enabled and forced (applies to the owner too) on all 16 tenant tables; policies key on `app.tenant_id`, set per transaction; pre-tenant auth lookups go through 5 narrow SECURITY DEFINER functions. Verified on PostgreSQL 16. | `platform/ops/postgres/schema.sql`, `platform/ops/postgres/README.md`, `tests/test_postgres_rls.py`, CI job `postgres-rls` |
| 5 | Encrypt sensitive data | Done + test | AES-256-GCM at rest for TOTP seeds, Remote ID text, resolution notes and evidence files; AAD binds each value to its table, tenant and record; versioned key ring with `rewrap-data` for rotation; tampering and copy-to-another-row are detected. | `airspace/crypto.py`, `tests/test_encryption.py` |
| 6 | Enforce server-side auth | Done + test | Every route except an explicit public list needs a verified credential (user JWT or device signature); the test enumerates all routes, so a new one can't ship unprotected. | `airspace/app.py`, `tests/test_inventory.py` |
| 7 | Lock record access | Done + test | Tenant comes only from verified identity; every query filters by it; other tenants' objects return 404; composite foreign keys plus RLS make cross-tenant rows impossible at the schema level. | `airspace/alerts.py`, `tests/test_rbac_tenancy.py`, `tests/test_postgres_rls.py` |
| 8 | Block field tampering | Done + test | Every input model is strict with `extra="forbid"` (enumerated by test); bodies carrying `tenant_id`, `role` or other unknown fields are rejected. | `airspace/schemas.py`, `tests/test_inventory.py::test_every_input_model_forbids_unknown_fields`, `tests/test_rbac_tenancy.py` |
| 9 | Secure session cookies | Done + test | Optional browser mode: refresh token only in `__Host-airspace_rt` (Secure, HttpOnly, SameSite=Strict, Path=/), never in JSON; CSRF needs an allowlisted Origin plus a double-submit token bound to the session. | `airspace/app.py`, `airspace/sessions.py`, `tests/test_cookies_csrf.py` |
| 10 | Hash passwords | Done + test | scrypt (N=2^15) with a per-user salt and an HMAC pepper; constant-time compare; dummy hash for unknown accounts so timing doesn't reveal them. | `airspace/auth.py`, `tests/test_auth.py` |
| 11 | Rate limit login | Done + test | Token buckets per IP and per account; exponential account lockout; client IP taken only from trusted proxies (pentest F2). | `airspace/ratelimit.py`, `airspace/net.py`, `tests/test_auth.py`, `pentest/attack_xff_spoofing.py` |
| 12 | Add bot protection | Done + test | After 3 failed logins per account or per IP, login needs a challenge: built-in proof of work (signed, IP-bound, 2-minute expiry, single use) or Turnstile/hCaptcha behind the same interface, failing closed. | `airspace/challenge.py`, `tests/test_bot_challenge.py` |
| 13 | Parameterize queries | Done + test | Every SQL statement is a string literal with bound parameters; an AST scan fails the build on f-strings, concatenation or `%` formatting; 10 injection families are probed against every parameter. | `airspace/db.py`, `tests/test_sql_safety.py`, `tests/test_hardening.py` |
| 14 | Validate all input | Done + test | Strict pydantic schemas with bounded strings, numbers and lists; NaN and Infinity rejected; JSON depth and body size capped; path ids pattern-checked; fuzzing finds no 5xx. | `airspace/schemas.py`, `airspace/security_headers.py`, `tests/test_fuzz.py` |
| 15 | Escape user content | Done + test | The API returns JSON only, with `nosniff` and a `default-src 'none'` CSP; the web apps escape every dynamic string before rendering; AI agents get attacker-controlled text inside an unforgeable data fence (pentest F1). | `airspace/mcp_server.py`, `apps/simulator/index.html`, `tests/test_ingest.py`, `pentest/attack_mcp_prompt_injection.py` |
| 16 | Restrict file uploads | Done + test | Evidence only from signed devices; JPEG or PNG by magic bytes; 5 MB cap; random storage keys outside any web root; `O_EXCL|O_NOFOLLOW`, mode 0600; encrypted at rest; download only through a two-person-approved, single-use grant. | `airspace/evidence.py`, `tests/test_workflows.py`, `tests/test_encryption.py` |
| 17 | Trim API responses | Done + test | Every response passes an explicit allowlist; forbidden names (hashes, salts, seeds, keys) are refused at any depth; the whole suite runs in strict mode, so any unexpected field fails a test; a crawler checks every readable route as each role. | `airspace/outputs.py`, `tests/test_response_trimming.py`, `tests/test_inventory.py` |
| 18 | Add security headers | Done + test | CSP, HSTS (2 years, preload), nosniff, frame denial, Referrer-Policy, Permissions-Policy, COOP and CORP, plus `no-store` on every response, errors included. | `airspace/security_headers.py`, `tests/test_hardening.py` |
| 19 | Force HTTPS | Done + test | With `AIRSPACE_REQUIRE_HTTPS=1` (the production default) plain-http requests get 400 `https_required`; the scheme comes from a trusted proxy's X-Forwarded-Proto or the direct connection, never from an arbitrary client header; HSTS pins browsers to https. | `airspace/security_headers.py`, `airspace/net.py`, `tests/test_https.py` |
| 20 | Scan dependencies | Done (CI) | Pinned versions; pip-audit, Bandit and CodeQL on every push and weekly; Dependabot for pip and Actions; Bandit also runs as a pre-commit hook. Hash-locked requirements need a machine with PyPI access (`pip-compile --generate-hashes`). | `platform/requirements.txt`, `.github/workflows/security.yml`, `.github/dependabot.yml` |

## Run it yourself

```bash
cd platform
python -m pytest -q                       # 160 tests; RLS tests also need AIRSPACE_PG_DSN and psycopg
bash pentest/run_all.sh                   # 76 regression attacks + the 3 fixed pen-test findings
pip install pre-commit && pre-commit install
```

## Still outside this repository

These need real infrastructure and are tracked in `docs/ARCHITECTURE.md`:

- **mTLS gateway** with device certificates
- **Key storage:** data keys unwrapped from a KMS or HSM, not environment variables
- **Shared state:** Redis-backed rate limits and failure counters shared across replicas
- **Perimeter:** WAF and DDoS protection
- **Evidence:** WORM storage
- **Login:** WebAuthn for operators
- **Outside review:** an external penetration test and a CJIS audit
