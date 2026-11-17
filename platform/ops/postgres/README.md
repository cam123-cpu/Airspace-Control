# PostgreSQL with row-level security

`schema.sql` is the production schema. It mirrors the SQLite schema in `airspace/db.py`, and it adds row-level security (RLS) as a second tenant-isolation layer that doesn't depend on application code.

## Apply

```bash
psql "$ADMIN_DSN" -v ON_ERROR_STOP=1 -f ops/postgres/schema.sql
# the API logs in as its own user that only holds airspace_app:
psql "$ADMIN_DSN" -c "CREATE ROLE airspace_api LOGIN PASSWORD '...' IN ROLE airspace_app"
```

Run it as a superuser or a role with `CREATEROLE`. It creates four roles idempotently.

## Roles

| Role | Login | BYPASSRLS | Used for |
|---|---|---|---|
| `airspace_owner` | no | no | Owns the tables. `FORCE ROW LEVEL SECURITY` binds the owner too. |
| `airspace_app` | through a login user | no | The API. CRUD on tenant tables, append-only on `audit_log`. |
| `airspace_definer` | no | yes | Owns 5 `SECURITY DEFINER` lookups that run before the tenant is known (login by email, refresh token, device, API token, user by id). Each returns one row and only the auth columns. |
| `airspace_maintenance` | through a separate login user | yes | CLI jobs that span tenants: bootstrap, retention purge, `rewrap-data`. Never used by the API. |

## Contract for the data layer

Every transaction on the app role sets the tenant first, transaction-locally:

```sql
BEGIN;
SELECT set_config('app.tenant_id', $1, true);   -- tenant id from the verified token or device
-- queries
COMMIT;
```

With `app.tenant_id` unset, tenant tables read as empty. A row for another tenant can't be inserted, and an existing row can't be moved to another tenant (`WITH CHECK`). Auth lookups call the `auth_*` functions instead of reading `users`, `refresh_tokens`, `devices` or `api_tokens` directly.

Moving `airspace/db.py` to psycopg is mechanical: replace `?` with `%s`, and have `DB.tx()` run the `set_config` call above with the principal's tenant.

## Verification

`tests/test_postgres_rls.py` proves the policies against a real server:

- cross-tenant SELECT returns nothing
- cross-tenant UPDATE and DELETE affect 0 rows
- cross-tenant INSERT and tenant-switching UPDATE are refused
- the audit log can't be changed
- the owner is bound
- the app role can't bypass RLS

CI runs it on `postgres:16`. Locally:

```bash
pip install "psycopg[binary]"
AIRSPACE_PG_DSN=postgresql://postgres:postgres@localhost:5432/postgres python -m pytest -q tests/test_postgres_rls.py
```
