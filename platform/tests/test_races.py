"""Check-then-act races in login and refresh (pentest F4).

Each thread gets its own database connection, the way separate uvicorn workers or replicas would, and all
threads are released at once. Before the fix, one TOTP code produced several sessions and parallel wrong
passwords were under-counted, so the lockout fired late.
"""
from __future__ import annotations

import threading
import time

from airspace import sessions
from airspace.auth import totp_now
from airspace.errors import AppError
from airspace.ratelimit import LIMITER

from conftest import PW
from test_auth import fresh_user  # noqa: F401  (fixture)

N = 6


def _burst(env, body_for):
    LIMITER.reset()
    gate = threading.Barrier(N)
    results = [None] * N

    def run(i):
        db = env.db()
        try:
            gate.wait()
            results[i] = sessions.login(env.s, db, body_for(i), f"203.0.113.{10 + i}")
        except AppError as e:
            results[i] = e
        finally:
            db.close()

    ts = [threading.Thread(target=run, args=(i,)) for i in range(N)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    LIMITER.reset()
    return results


def _wait_for_fresh_step():
    while int(time.time()) % 30 > 25:
        time.sleep(0.25)


def test_one_totp_code_gives_one_session(env, fresh_user):
    _wait_for_fresh_step()
    code = totp_now(fresh_user["totp"])
    rs = _burst(env, lambda i: {"email": fresh_user["email"], "password": PW, "totp": code})
    assert sum(isinstance(r, dict) for r in rs) == 1, rs


def test_parallel_wrong_passwords_are_all_counted(env, fresh_user):
    rs = _burst(env, lambda i: {"email": fresh_user["email"], "password": f"wrong password {i}"})
    checked = sum(isinstance(r, AppError) and r.status == 401 for r in rs)
    db = env.db()
    row = db.one("SELECT failed_count, locked_until FROM users WHERE id = ?", (fresh_user["id"],))
    db.close()
    # a request that arrives after the bot challenge kicks in is refused before the password check (428);
    # every request that did reach the check must be in the counter
    assert row["failed_count"] == checked >= env.s.lockout_threshold
    assert row["locked_until"] > time.time()


def test_parallel_refresh_yields_one_token(env, fresh_user):
    LIMITER.reset()
    db = env.db()
    try:
        rt = sessions.login(env.s, db, {"email": fresh_user["email"], "password": PW,
                                        "totp": totp_now(fresh_user["totp"])}, "203.0.113.1")["refresh_token"]
    finally:
        db.close()
    gate = threading.Barrier(N)
    ok = []

    def run(i):
        d = env.db()
        try:
            gate.wait()
            ok.append(sessions.refresh(env.s, d, {"refresh_token": rt}, f"203.0.113.{40 + i}"))
        except AppError:
            pass
        finally:
            d.close()

    ts = [threading.Thread(target=run, args=(i,)) for i in range(N)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    LIMITER.reset()
    assert len(ok) <= 1


def test_shared_login_limit_holds_across_connections(env):
    """Each connection stands in for a worker; together they still get one bucket's worth of attempts."""
    from airspace.ratelimit import LIMITS, check_shared
    LIMITER.reset()
    rate, burst = LIMITS["login_account"]
    dbs = [env.db() for _ in range(4)]
    try:
        allowed = sum(check_shared(dbs[i % 4], "login_account", "victim@a.example", now=1_000_000.0) for i in range(20))
        refilled = check_shared(dbs[0], "login_account", "victim@a.example", now=1_000_000.0 + 1 / rate)
    finally:
        for d in dbs:
            d.close()
    LIMITER.reset()
    assert allowed == burst
    assert refilled


def test_shared_buckets_store_no_raw_keys(env):
    from airspace.ratelimit import check_shared
    db = env.db()
    try:
        check_shared(db, "login_ip", "198.51.100.77")
        keys = [r["key_hash"] for r in db.q("SELECT key_hash FROM rate_buckets")]
    finally:
        db.close()
    assert keys and all(len(k) == 64 and "198.51" not in k for k in keys)
