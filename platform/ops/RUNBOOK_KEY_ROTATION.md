# Runbook: key and secret rotation

## JWT signing key (scheduled every 90 days, or immediately if compromised)
1. `python -m airspace.cli gen-key` prints a new base64 PEM. In production, create the key in KMS instead.
2. Prepend it to the ring: `AIRSPACE_JWT_KEYS="k2026q4:<new>,k2026q3:<current>"`. Deploy.
   - New tokens are signed with `k2026q4`.
   - Tokens signed with `k2026q3` stay valid until they expire (at most 10 minutes).
3. After 15 minutes, remove `k2026q3` from the ring and deploy again.
4. **Compromise:** skip the overlap. Deploy with only the new key. Every outstanding access token fails at once, and users re-authenticate with their refresh tokens.

## Evidence signing key (yearly, or if compromised)
1. Generate a new key, set `AIRSPACE_EVIDENCE_KEY`, deploy.
2. Publish the new public key, with an effective date, to the evidence-verification page used by prosecutors.
3. Keep every old public key published forever: evidence signed earlier must stay verifiable.
4. **Compromise:** publish a revocation notice with a timestamp. Evidence signed after that time with the old key is not trustworthy. Re-verify affected items against the edge device's TPM signature and its SHA-256 recorded in the audit chain.

## Pepper (if compromised)
1. Add the new pepper next to the old one. This needs the two-pepper verify path, planned for v1.1.
2. On each successful login, re-hash the password with the new pepper.
3. After 90 days, force a password reset for every account not yet re-hashed, then remove the old pepper.

## Edge device key (on tamper alarm, theft or RMA)
1. A tenant_admin deactivates the device: `UPDATE devices SET active = 0`, done through the admin API in v1.1. Its requests fail with 401 at once.
2. Re-enrol the replacement: a new TPM-generated key, registered with `POST /api/v1/devices`.
3. Review the audit log for anything ingested from that device since the last good heartbeat.

## API tokens for MCP and automation
- Revoke: `POST /api/v1/api-tokens/{id}/revoke`. Takes effect on the agent's next call.
- Rotate: create a new token, then update the agent's secret store (never commit tokens to a prompt or repository). Revoke the old one.
- Maximum lifetime is 90 days. Prefer hours for interactive agents.
