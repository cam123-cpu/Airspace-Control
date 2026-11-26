"""Edge-of-app hardening as a pure ASGI middleware:
request IDs, body-size cap (streaming), strict CORS allowlist, security headers on every response,
and a last-resort handler that never leaks stack traces. Logs are structured JSON without secrets."""
from __future__ import annotations

import json
import logging
import re
import secrets
import sys
import time

from . import net
from .config import Settings

log = logging.getLogger("airspace")
if not log.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)

_RID = re.compile(r"^[A-Za-z0-9-]{8,64}$")
_SINGLE_HEADERS = {b"authorization", b"content-length", b"content-type", b"transfer-encoding", b"host", b"origin",
                   b"cookie", b"x-device-id", b"x-signature", b"x-timestamp", b"x-nonce", b"idempotency-key",
                   b"x-csrf-token", b"x-session-mode", b"x-forwarded-proto"}

SECURITY_HEADERS = [
    (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"),
    (b"strict-transport-security", b"max-age=63072000; includeSubDomains; preload"),
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=(), payment=(), usb=()"),
    (b"cross-origin-opener-policy", b"same-origin"),
    (b"cross-origin-resource-policy", b"same-origin"),
    (b"cache-control", b"no-store"),
]


def jlog(**kw) -> None:
    kw.setdefault("ts", round(time.time(), 3))
    log.info(json.dumps(kw, separators=(",", ":"), default=str))


class HardeningMiddleware:
    def __init__(self, app, settings: Settings):
        self.app = app
        self.s = settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        raw_headers = scope.get("headers", [])
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in raw_headers}
        rid = headers.get("x-request-id", "")
        rid = rid if _RID.match(rid) else secrets.token_hex(8)
        scope.setdefault("state", {})["request_id"] = rid
        # Header limits come before anything reads them. A repeated security header is refused outright: the
        # dict above keeps the last copy, other layers (proxies, Starlette) may keep the first, and an attacker
        # who can make two layers read two different values gets to choose which check sees which.
        total = sum(len(k) + len(v) for k, v in raw_headers)
        names = [k.lower() for k, _ in raw_headers]
        dup = {n for n in names if names.count(n) > 1} & _SINGLE_HEADERS
        if total > self.s.max_header_bytes or len(raw_headers) > self.s.max_header_count or dup:
            body = json.dumps({"error": "header_fields_rejected", "request_id": rid}).encode()
            await send({"type": "http.response.start", "status": 431,
                        "headers": SECURITY_HEADERS + [(b"content-type", b"application/json"), (b"x-request-id", rid.encode())]})
            return await send({"type": "http.response.body", "body": body})
        origin = headers.get("origin")
        allowed_origin = origin if origin and origin in self.s.cors_origins else None
        started = time.monotonic()
        status_holder = {"status": 500}
        limit = self.s.max_evidence_bytes if scope["path"].startswith("/ingest/v1/evidence") else self.s.max_body_bytes

        async def send_wrapped(message):
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                hs = [(k, v) for k, v in message.get("headers", []) if k.lower() not in {h for h, _ in SECURITY_HEADERS}]
                hs += SECURITY_HEADERS + [(b"x-request-id", rid.encode())]
                if allowed_origin:
                    hs += [(b"access-control-allow-origin", allowed_origin.encode()), (b"vary", b"Origin")]
                message = {**message, "headers": hs}
            await send(message)

        async def error(status: int, code: str):
            body = json.dumps({"error": code, "request_id": rid}).encode()
            await send_wrapped({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json")]})
            await send_wrapped({"type": "http.response.body", "body": body})

        # HTTPS only: plain-http requests are refused (the scheme comes from a trusted proxy's X-Forwarded-Proto
        # or from the direct connection; never from an arbitrary client header). /healthz stays reachable for probes.
        if self.s.require_https and scope["path"] != "/healthz":
            peer = (scope.get("client") or ("", 0))[0]
            if net.request_scheme(self.s, peer, scope.get("scheme", "http"), headers) != "https":
                return await error(400, "https_required")

        # CORS preflight: answer only for allowlisted origins; never wildcard, never credentials
        if scope["method"] == "OPTIONS":
            if not allowed_origin:
                return await error(403, "cors_denied")
            await send_wrapped({"type": "http.response.start", "status": 204, "headers": [
                (b"access-control-allow-methods", b"GET, POST"),
                (b"access-control-allow-headers", b"authorization, content-type, x-request-id, idempotency-key, x-csrf-token, x-session-mode"),
                (b"access-control-max-age", b"600")]})
            return await send_wrapped({"type": "http.response.body", "body": b""})

        cl = headers.get("content-length")
        if cl is not None and (not cl.isdigit() or int(cl) > limit):
            return await error(413, "payload_too_large")

        received = {"n": 0}

        async def receive_limited():
            msg = await receive()
            if msg["type"] == "http.request":
                received["n"] += len(msg.get("body", b""))
                if received["n"] > limit:
                    raise _TooLarge()
            return msg

        try:
            await self.app(scope, receive_limited, send_wrapped)
        except _TooLarge:
            await error(413, "payload_too_large")
        except Exception as e:  # never leak internals
            jlog(level="error", request_id=rid, path=scope["path"], err=type(e).__name__)
            await error(500, "internal_error")
        finally:
            jlog(level="info", request_id=rid, method=scope["method"], path=scope["path"], status=status_holder["status"],
                 ms=round((time.monotonic() - started) * 1000, 1))


class _TooLarge(Exception):
    pass


def json_depth_ok(obj, max_depth: int) -> bool:
    stack = [(obj, 1)]
    while stack:
        o, d = stack.pop()
        if d > max_depth:
            return False
        if isinstance(o, dict):
            stack.extend((v, d + 1) for v in o.values())
        elif isinstance(o, list):
            stack.extend((v, d + 1) for v in o)
    return True
