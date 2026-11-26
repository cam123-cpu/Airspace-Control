"""Row-level security in PostgreSQL proves tenant isolation at the database layer, independent of application code.

Runs only when AIRSPACE_PG_DSN points at a disposable database (superuser or CREATEROLE) and psycopg is installed.
CI runs it against a postgres:16 service and sets AIRSPACE_PG_REQUIRED=1, which turns a skip into a failure.
"""
from __future__ import annotations

import os

import pytest

DSN = os.environ.get("AIRSPACE_PG_DSN", "")
REQUIRED = os.environ.get("AIRSPACE_PG_REQUIRED") == "1"
try:
    import psycopg
    from psycopg import errors as pgerr
except ImportError:  # pragma: no cover
    psycopg = None

if not (DSN and psycopg):
    if REQUIRED:
        raise RuntimeError("AIRSPACE_PG_REQUIRED=1 but AIRSPACE_PG_DSN or psycopg is missing")
    pytest.skip("set AIRSPACE_PG_DSN and install psycopg to run the RLS tests", allow_module_level=True)

SCHEMA = os.path.join(os.path.dirname(__file__), "..", "ops", "postgres", "schema.sql")


@pytest.fixture(scope="module")
def pg():
    conn = psycopg.connect(DSN, autocommit=True)
    conn.execute("DROP SCHEMA IF EXISTS airspace CASCADE")
    with open(SCHEMA, encoding="utf-8") as fh:
        conn.execute(fh.read())
    conn.execute("SET search_path = airspace")
    with conn.transaction():
        conn.execute("SET LOCAL ROLE airspace_maintenance")      # cross-tenant seeding is a CLI-only job
        conn.execute("INSERT INTO tenants (id, name, created_at) VALUES ('tn_a','A',1), ('tn_b','B',1)")
        conn.execute("INSERT INTO sites (id, tenant_id, name, created_at) VALUES ('site_a','tn_a','SA',1), ('site_b','tn_b','SB',1)")
        conn.execute("INSERT INTO alerts (id, tenant_id, site_id, ext_track_id, severity, title, created_at) VALUES "
                     "('al_a','tn_a','site_a','T1','alert','a',1), ('al_b','tn_b','site_b','T1','alert','b',1)")
        conn.execute("INSERT INTO users (id, tenant_id, email, pw_hash, role, created_at) VALUES "
                     "('u_a','tn_a','a@x.example','h','operator',1), ('u_b','tn_b','b@x.example','h','operator',1)")
        conn.execute("INSERT INTO audit_log (tenant_id, ts, actor, action, target, detail, prev_hash, hash) VALUES "
                     "('tn_b',1,'s','x','t','{}','0','1')")
    yield conn
    conn.close()


class _As:
    """One transaction as the application role, scoped to a tenant (or to none)."""

    def __init__(self, conn, tenant: str | None):
        self.conn, self.tenant = conn, tenant

    def __enter__(self):
        self.tx = self.conn.transaction()
        self.tx.__enter__()
        self.conn.execute("SET LOCAL ROLE airspace_app")
        if self.tenant is not None:
            self.conn.execute("SELECT set_config('app.tenant_id', %s, true)", (self.tenant,))
        return self.conn

    def __exit__(self, *exc):
        return self.tx.__exit__(*exc)


def _count(conn, sql, params=()):
    return conn.execute(sql, params).fetchone()[0]


def test_select_sees_only_own_tenant(pg):
    with _As(pg, "tn_a") as c:
        assert _count(c, "SELECT count(*) FROM alerts") == 1
        assert _count(c, "SELECT count(*) FROM alerts WHERE tenant_id = %s", ("tn_b",)) == 0
        assert _count(c, "SELECT count(*) FROM users") == 1
        assert _count(c, "SELECT count(*) FROM audit_log") == 0


def test_no_tenant_means_no_rows(pg):
    with _As(pg, None) as c:
        assert _count(c, "SELECT count(*) FROM alerts") == 0
        assert _count(c, "SELECT count(*) FROM tenants") == 0


def test_cross_tenant_update_and_delete_affect_nothing(pg):
    with _As(pg, "tn_a") as c:
        assert c.execute("UPDATE alerts SET title = 'pwned' WHERE tenant_id = %s", ("tn_b",)).rowcount == 0
        assert c.execute("DELETE FROM alerts WHERE tenant_id = %s", ("tn_b",)).rowcount == 0
    with _As(pg, "tn_b") as c:
        assert c.execute("SELECT title FROM alerts WHERE id = 'al_b'").fetchone()[0] == "b"


def test_cross_tenant_insert_refused(pg):
    with _As(pg, "tn_a") as c:
        with pytest.raises(pgerr.InsufficientPrivilege):
            with c.transaction():
                c.execute("INSERT INTO alerts (id, tenant_id, site_id, ext_track_id, severity, title, created_at) "
                          "VALUES ('al_x','tn_b','site_b','T9','alert','x',1)")


def test_moving_a_row_to_another_tenant_refused(pg):
    with _As(pg, "tn_a") as c:
        with pytest.raises(pgerr.InsufficientPrivilege):
            with c.transaction():
                c.execute("UPDATE alerts SET tenant_id = 'tn_b' WHERE id = 'al_a'")


def test_audit_log_is_append_only_for_the_app(pg):
    with _As(pg, "tn_b") as c:
        with pytest.raises(pgerr.InsufficientPrivilege):
            with c.transaction():
                c.execute("UPDATE audit_log SET actor = 'x'")


def test_owner_is_bound_too(pg):
    with pg.transaction():
        pg.execute("SET LOCAL ROLE airspace_owner")
        assert _count(pg, "SELECT count(*) FROM alerts") == 0       # FORCE ROW LEVEL SECURITY


def test_app_role_cannot_bypass_rls(pg):
    row = pg.execute("SELECT rolbypassrls, rolsuper FROM pg_roles WHERE rolname = 'airspace_app'").fetchone()
    assert row == (False, False)


def test_definer_lookup_works_before_tenant_is_known(pg):
    with _As(pg, None) as c:
        assert c.execute("SELECT tenant_id FROM auth_user_by_email(%s)", ("A@X.example",)).fetchone()[0] == "tn_a"
        # it returns exactly the auth columns, nothing else
        cols = [d.name for d in c.execute("SELECT * FROM auth_user_by_id(%s)", ("u_b",)).description]
        assert cols == ["id", "tenant_id", "role", "active"]


def test_every_tenant_table_has_forced_rls(pg):
    rows = pg.execute("SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class c "
                      "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'airspace' AND c.relkind = 'r'").fetchall()
    # no tenant data, and both are used before login knows the tenant:
    exempt = {"pow_used",       # hashes of spent challenges
              "rate_buckets"}   # auth rate-limit buckets keyed by SHA-256 of bucket:key
    for name, enabled, forced in rows:
        if name not in exempt:
            assert enabled and forced, name
