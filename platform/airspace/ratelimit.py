"""Token-bucket rate limiting.

Two backends share one interface:
  * `check()`: in-process buckets for high-volume paths (API calls, ingest). Each worker enforces its own
    copy, so the effective limit scales with the worker count. That is acceptable for throughput limits.
  * `check_shared()`: buckets stored in the database, updated inside a write transaction, so a burst split
    across workers or replicas still draws from one bucket. Used for the login and refresh limits, where
    the limit is a security control (credential stuffing, MFA guessing) and traffic is low.
Keys are hashed before storage, so the table holds no emails or IP addresses.
"""
from __future__ import annotations

import hashlib
import secrets
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .db import DB


class RateLimiter:
    def __init__(self, max_keys: int = 100_000):
        self._b: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()
        self._max = max_keys

    def allow(self, key: str, rate_per_s: float, burst: int, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._b.get(key, (float(burst), now))
            tokens = min(float(burst), tokens + (now - last) * rate_per_s)
            ok = tokens >= 1.0
            self._b[key] = (tokens - 1.0 if ok else tokens, now)
            if len(self._b) > self._max:  # bounded memory under key-flooding
                for k in list(self._b)[: len(self._b) // 10]:
                    self._b.pop(k, None)
            return ok

    def reset(self) -> None:
        with self._lock:
            self._b.clear()


LIMITER = RateLimiter()

# (rate per second, burst)
LIMITS = {
    "login_ip": (0.2, 10),        # ~12/min per source IP
    "login_account": (0.05, 5),   # ~3/min per account (plus lockout)
    "refresh_ip": (0.5, 20),
    "api_user": (10.0, 60),
    "ingest_device": (20.0, 100),
    "mcp_token": (2.0, 20),
}


def check(bucket: str, key: str) -> bool:
    rate, burst = LIMITS[bucket]
    return LIMITER.allow(f"{bucket}:{key}", rate, burst)


SHARED_IDLE_S = 3600        # a bucket untouched this long is full again, so its row can go


def check_shared(db: "DB", bucket: str, key: str, now: float | None = None) -> bool:
    """Take one token from a database-backed bucket. Must not be called inside another transaction."""
    rate, burst = LIMITS[bucket]
    now = time.time() if now is None else now
    kh = hashlib.sha256(f"{bucket}:{key}".encode()).hexdigest()
    with db.tx():
        row = db.one("SELECT tokens, updated_at FROM rate_buckets WHERE key_hash = ?", (kh,))
        tokens = float(burst) if row is None else min(float(burst), row["tokens"] + max(0.0, now - row["updated_at"]) * rate)
        ok = tokens >= 1.0
        left = tokens - 1.0 if ok else tokens
        db.x("INSERT INTO rate_buckets (key_hash, tokens, updated_at) VALUES (?, ?, ?) "
             "ON CONFLICT (key_hash) DO UPDATE SET tokens = excluded.tokens, updated_at = excluded.updated_at",
             (kh, left, now))
        if secrets.randbelow(100) == 0:           # amortized cleanup keeps the table small under key flooding
            db.x("DELETE FROM rate_buckets WHERE updated_at < ?", (now - SHARED_IDLE_S,))
    return ok


def reset_shared(db: "DB") -> None:
    db.x("DELETE FROM rate_buckets")
