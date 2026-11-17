# Security policy

## Reporting a vulnerability

Please **do not open a public issue.** Use GitHub's private vulnerability reporting (Security → *Report a vulnerability*) and include reproduction steps. Expect an acknowledgement within 3 business days.

## Scope

| Component | Security docs |
|---|---|
| `platform/` (API, device ingest, MCP wrapper) | [platform/SECURITY.md](platform/SECURITY.md), [ARCHITECTURE](docs/ARCHITECTURE.md), [pen test](docs/security/PENTEST_REPORT.md) |
| `apps/*` (static web apps) | Strict CSP, no third-party runtime code, escaped output |
| `sim-core/`, `ml/` | [Security audit](docs/security/SECURITY_AUDIT.md), [red team](docs/security/REDTEAM.md) |

## What's verified, and what isn't

- **Platform (done):** unit and integration tests, plus 76 regression attacks (`platform/pentest/run_all.sh`). All 3 pen-test findings are fixed.
- **CI (done):** runs tests, CodeQL, Bandit, pip-audit and secret scanning on every push.
- **Not done yet:** an external penetration test, a CJIS audit, and production infrastructure (mTLS gateway and device PKI, KMS/HSM, Postgres RLS, WAF/DDoS protection, WORM evidence storage).

## Handling sensitive data

Never commit real site layouts, sensor coverage maps, blind spots, keys, or captured footage of real people. The threat model treats those as restricted data. Test keys are generated fresh on every run.
