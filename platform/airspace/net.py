"""Client address and scheme, read only from sources the platform can trust.

X-Forwarded-For and X-Forwarded-Proto are accepted only when the direct TCP peer is one of our own proxies
(AIRSPACE_TRUSTED_PROXIES). For X-Forwarded-For the entry appended by our proxy chain is used, counted from the
right; the client-controlled leftmost entry is never trusted (pentest F2). uvicorn must run with
--no-proxy-headers so this module is the only place these headers are interpreted.
"""
from __future__ import annotations

import ipaddress

from .config import Settings


def peer_trusted(peer: str, nets) -> bool:
    try:
        ip = ipaddress.ip_address(peer)
    except ValueError:
        return False
    return any(ip in n for n in nets)


def _from_trusted_proxy(s: Settings, peer: str) -> bool:
    return s.trust_proxy_headers and peer_trusted(peer, s.trusted_proxies)


def client_ip(s: Settings, peer: str, headers: dict[str, str]) -> str:
    if not _from_trusted_proxy(s, peer):
        return peer
    parts = [p.strip() for p in headers.get("x-forwarded-for", "").split(",") if p.strip()]
    if len(parts) < s.trusted_proxy_hops:
        return peer
    try:
        return str(ipaddress.ip_address(parts[-s.trusted_proxy_hops]))
    except ValueError:
        return peer


def request_scheme(s: Settings, peer: str, scope_scheme: str, headers: dict[str, str]) -> str:
    """The scheme the client actually used. Behind a trusted TLS-terminating proxy, its X-Forwarded-Proto
    (last value, the one our proxy wrote); otherwise the scheme of the direct connection."""
    if _from_trusted_proxy(s, peer):
        vals = [v.strip().lower() for v in headers.get("x-forwarded-proto", "").split(",") if v.strip()]
        if vals and vals[-1] in ("http", "https"):
            return vals[-1]
    return scope_scheme
