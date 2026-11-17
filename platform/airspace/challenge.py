"""Bot protection for login.

After `challenge_after` failed logins for an account or a source IP (counted over 15 minutes), a login must carry a
solved challenge. Two kinds of verifier sit behind one interface:

  * built-in proof of work (default, no third party): the server issues a signed, short-lived challenge bound to
    the client's IP; the client finds a `solution` so that SHA-256(challenge ":" solution) starts with
    `difficulty` zero bits. Each challenge can be redeemed once (pow_used table, shared across replicas).
  * hosted CAPTCHA (Cloudflare Turnstile or hCaptcha): the browser widget returns a response token that the
    server checks with the provider. Network calls go only to the provider's fixed https endpoint.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
import time
import urllib.parse
import urllib.request
from typing import Protocol

from .config import Settings
from .db import DB

WINDOW_S = 15 * 60


# ---------------- failure counting ----------------
class FailureTracker:
    """In-process counter of recent login failures per key (account email or IP). Production backs this
    with the same Redis used for rate limits so every replica sees the same counts."""

    def __init__(self, max_keys: int = 100_000):
        self._f: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self._max = max_keys

    def record(self, key: str, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            lst = [t for t in self._f.get(key, []) if now - t < WINDOW_S]
            lst.append(now)
            self._f[key] = lst[-50:]
            if len(self._f) > self._max:
                for k in list(self._f)[: len(self._f) // 10]:
                    self._f.pop(k, None)

    def count(self, key: str, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with self._lock:
            return sum(1 for t in self._f.get(key, []) if now - t < WINDOW_S)

    def clear(self, key: str) -> None:
        with self._lock:
            self._f.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._f.clear()


FAILURES = FailureTracker()


def challenge_needed(s: Settings, email: str, ip: str) -> bool:
    return FAILURES.count("acct:" + email) >= s.challenge_after or FAILURES.count("ip:" + ip) >= s.challenge_after


def record_failure(email: str, ip: str) -> None:
    FAILURES.record("acct:" + email)
    FAILURES.record("ip:" + ip)


# ---------------- proof of work ----------------
def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _pow_key(s: Settings) -> bytes:
    return hmac.new(s.pepper, b"airspace-pow-v1", hashlib.sha256).digest()


def _ip_tag(s: Settings, ip: str) -> str:
    return _b64e(hmac.new(_pow_key(s), ("ip|" + ip).encode(), hashlib.sha256).digest()[:12])


def issue_pow(s: Settings, ip: str, now: int | None = None) -> dict:
    now = int(now or time.time())
    body = {"n": _b64e(secrets.token_bytes(16)), "d": s.pow_difficulty, "exp": now + s.pow_ttl_s, "ip": _ip_tag(s, ip)}
    raw = _b64e(json.dumps(body, separators=(",", ":"), sort_keys=True).encode())
    sig = _b64e(hmac.new(_pow_key(s), raw.encode(), hashlib.sha256).digest())
    return {"kind": "pow", "algorithm": "sha256-leading-zero-bits", "token": f"{raw}.{sig}",
            "difficulty": s.pow_difficulty, "expires_at": body["exp"]}


def leading_zero_bits(digest: bytes) -> int:
    n = 0
    for byte in digest:
        if byte == 0:
            n += 8
            continue
        return n + (8 - byte.bit_length())
    return n


def solve_pow(token: str, difficulty: int, limit: int = 1 << 26) -> str:
    """Reference solver (what the browser runs in a Web Worker)."""
    for i in range(limit):
        cand = format(i, "x")
        if leading_zero_bits(hashlib.sha256(f"{token}:{cand}".encode()).digest()) >= difficulty:
            return cand
    raise RuntimeError("no solution found")


def verify_pow(s: Settings, db: DB, token: str, solution: str, ip: str, now: int | None = None) -> bool:
    now = int(now or time.time())
    if not (isinstance(token, str) and isinstance(solution, str) and len(token) <= 400 and 1 <= len(solution) <= 32):
        return False
    try:
        raw, sig = token.split(".", 1)
        if not hmac.compare_digest(_b64e(hmac.new(_pow_key(s), raw.encode(), hashlib.sha256).digest()), sig):
            return False
        body = json.loads(_b64d(raw))
    except (ValueError, TypeError):
        return False
    if body.get("exp", 0) < now or not hmac.compare_digest(str(body.get("ip", "")), _ip_tag(s, ip)):
        return False
    d = int(body.get("d", 99))
    if d < s.pow_difficulty:
        return False
    if leading_zero_bits(hashlib.sha256(f"{token}:{solution}".encode()).digest()) < d:
        return False
    th = hashlib.sha256(token.encode()).hexdigest()
    with db.tx():
        db.x("DELETE FROM pow_used WHERE expires_at < ?", (now,))
        if db.one("SELECT token_hash FROM pow_used WHERE token_hash = ?", (th,)):
            return False                       # single use
        db.x("INSERT INTO pow_used (token_hash, expires_at) VALUES (?, ?)", (th, int(body["exp"]) + 60))
    return True


# ---------------- verifier interface ----------------
class ChallengeVerifier(Protocol):
    kind: str

    def verify(self, s: Settings, db: DB, payload: dict, ip: str) -> bool: ...


class PowVerifier:
    kind = "pow"

    def verify(self, s: Settings, db: DB, payload: dict, ip: str) -> bool:
        return verify_pow(s, db, payload.get("token", ""), payload.get("solution", ""), ip)


class HostedCaptchaVerifier:
    """Turnstile / hCaptcha server-side check. Fixed provider URL, short timeout, fails closed."""
    ENDPOINTS = {"turnstile": "https://challenges.cloudflare.com/turnstile/v0/siteverify",
                 "hcaptcha": "https://api.hcaptcha.com/siteverify"}

    def __init__(self, kind: str):
        if kind not in self.ENDPOINTS:
            raise ValueError("unknown provider")
        self.kind = kind

    def verify(self, s: Settings, db: DB, payload: dict, ip: str) -> bool:
        token = payload.get("token", "")
        if not isinstance(token, str) or not 1 <= len(token) <= 2048:
            return False
        data = urllib.parse.urlencode({"secret": s.challenge_secret, "response": token, "remoteip": ip}).encode()
        req = urllib.request.Request(self.ENDPOINTS[self.kind], data=data, method="POST",
                                     headers={"content-type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:  # nosec B310: fixed https provider URL, not user input
                body = json.loads(r.read(65536))
        except (OSError, ValueError):
            return False                       # provider unreachable: fail closed
        return body.get("success") is True


def verifier_for(s: Settings) -> ChallengeVerifier:
    return PowVerifier() if s.challenge_provider == "pow" else HostedCaptchaVerifier(s.challenge_provider)
