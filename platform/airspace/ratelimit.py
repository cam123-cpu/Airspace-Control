"""Token-bucket rate limiting. In-process here; production uses the same interface backed by Redis
(atomic Lua script) so limits hold across replicas."""
from __future__ import annotations

import threading
import time


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
