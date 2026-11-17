# Security policy

## Reporting a vulnerability
Email security@airspace.example, encrypted with the PGP key published at https://airspace.example/.well-known/security.txt. Include the affected component and version, reproduction steps, and the impact you observed.
- We acknowledge within 2 business days and give an initial assessment within 5.
- Fix targets: critical 7 days, high 30 days, others 90 days.
- We don't pursue good-faith research that respects these rules:
  - no access to other customers' data
  - no denial of service
  - no social engineering
  - no testing of live correctional facilities

## Supported versions
| Version | Supported |
|---|---|
| 1.x | Yes, with security fixes |
| < 1.0 | No |

## Security testing in this repository
`python -m pytest tests/` runs 106 tests. Highlights:
- **SQL safety:** a static scan rejects any SQL that isn't a string literal, and any use of eval, exec or subprocess.
- **Authentication:**
  - scrypt with pepper
  - the RFC 6238 test vector, and blocking of TOTP replay
  - lockout, and rate limits by IP and by account
  - forged tokens rejected: JWT `alg=none`, HS256 signed with the public key, a foreign signing key, a claimed role escalation, expired tokens, wrong audience, unknown key id
  - refresh-token rotation with reuse detection
  - immediate revocation of a disabled user
- **Authorization:** the role-by-endpoint matrix, and that nothing is allowed unless listed.
- **Tenant isolation:**
  - another tenant's alerts, sites, sensors, tracks, devices and approval targets are all invisible
  - schema-level composite-FK block
  - the platform auditor is read-only
- **Device ingest:** signature, tampered body, wrong or foreign key, ±60 s clock window, nonce replay, deactivated device, strict schema bounds, NaN/Infinity, another site's sensors, idempotency.
- **Workflows:**
  - two-person rule: no self-approval, expiry, no double execution
  - evidence: type sniffing, size cap, server signature, one-time export grant, file tampering detected
  - audit chain: append-only triggers, tampering detected, no secrets in records
  - retention with legal hold
- **HTTP hardening:**
  - security headers on every response, CORS allowlist, request-id sanitising, body and JSON-depth caps, content-type enforcement, bounded pagination
  - 10 injection payload families sent through every parameter
  - 400 fuzzed bodies plus 150 fuzzed header sets with no 5xx responses
- **MCP:**
  - an invalid token refuses to start the server
  - schema bounds, tool allowlist, read-only default
  - writes need a human confirmation code that is single-use, bound to one alert and to one tenant
  - free text marked UNTRUSTED, control characters stripped
  - instant revocation, audited calls, rate limits

These tests are a regression floor. Before any production deployment, commission an independent penetration test and a CJIS security audit.
