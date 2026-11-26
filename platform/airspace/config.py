"""Runtime configuration. Secrets come only from the environment (or a mounted secrets file in prod).

Fail-closed: in production every secret must be provided; development generates ephemeral keys and says so.
"""
from __future__ import annotations

import base64
import os
import secrets
from dataclasses import dataclass, field

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


class ConfigError(RuntimeError):
    pass


def _env(name: str, default: str | None = None) -> str | None:
    file_var = os.environ.get(name + "_FILE")  # docker/k8s secrets mounted as files
    if file_var:
        with open(file_var, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    return os.environ.get(name, default)


def _load_key(pem: str) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(pem.encode(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ConfigError("signing keys must be Ed25519")
    return key


@dataclass
class Settings:
    env: str = "development"
    db_path: str = "airspace.db"
    evidence_dir: str = "evidence_store"
    issuer: str = "https://airspace.example/auth"
    audience: str = "airspace-api"
    access_ttl_s: int = 600            # 10 min
    refresh_ttl_s: int = 12 * 3600     # 12 h, rotating
    pepper: bytes = b""
    # JWT signing key ring: kid -> private key. `current_kid` signs; all listed keys verify (rotation overlap).
    jwt_keys: dict[str, Ed25519PrivateKey] = field(default_factory=dict)
    current_kid: str = ""
    evidence_key: Ed25519PrivateKey | None = None
    cors_origins: tuple[str, ...] = ()
    max_body_bytes: int = 256 * 1024
    max_header_bytes: int = 16 * 1024
    max_header_count: int = 64
    max_evidence_bytes: int = 5 * 1024 * 1024
    max_json_depth: int = 12
    clock_skew_s: int = 60
    lockout_threshold: int = 5
    approval_ttl_s: int = 15 * 60
    trust_proxy_headers: bool = False   # only behind the gateway that strips/sets X-Forwarded-For
    trusted_proxies: tuple = ()          # CIDRs of our own proxies (direct peers allowed to set X-Forwarded-For)
    trusted_proxy_hops: int = 1          # how many of our proxies append to X-Forwarded-For
    # encryption at rest (AES-256-GCM key ring: kid -> 32-byte key; current_data_kid encrypts)
    data_keys: dict[str, bytes] = field(default_factory=dict)
    current_data_kid: str = ""
    require_https: bool = False          # production default: on
    # bot protection: after this many failed logins for an account or a source IP, a challenge is required
    challenge_after: int = 3
    challenge_provider: str = "pow"      # pow | turnstile | hcaptcha
    challenge_secret: str = ""           # provider secret (Turnstile/hCaptcha), via *_FILE in production
    pow_difficulty: int = 18             # leading zero bits; ~260k SHA-256 evaluations for the client
    pow_ttl_s: int = 120
    # responses: in strict mode an unexpected output field is a server error (used in tests/CI)
    strict_outputs: bool = False

    @property
    def production(self) -> bool:
        return self.env == "production"

    def keyring(self):
        """AES-GCM key ring for encryption at rest. Development without configured keys gets one ephemeral
        key per Settings object (load_settings persists it next to the dev database instead)."""
        from .crypto import Keyring
        if not self.data_keys:
            if self.production:
                raise ConfigError("production requires AIRSPACE_DATA_KEYS")
            self.data_keys = {"dev": secrets.token_bytes(32)}
            self.current_data_kid = "dev"
        if getattr(self, "_ring", None) is None or self._ring.current != self.current_data_kid:
            self._ring = Keyring(self.data_keys, self.current_data_kid)
        return self._ring


def load_settings() -> Settings:
    s = Settings()
    s.env = _env("AIRSPACE_ENV", "development") or "development"
    s.db_path = _env("AIRSPACE_DB_PATH", s.db_path) or s.db_path
    s.evidence_dir = _env("AIRSPACE_EVIDENCE_DIR", s.evidence_dir) or s.evidence_dir
    s.issuer = _env("AIRSPACE_ISSUER", s.issuer) or s.issuer
    s.audience = _env("AIRSPACE_AUDIENCE", s.audience) or s.audience
    s.cors_origins = tuple(o.strip() for o in (_env("AIRSPACE_CORS_ORIGINS", "") or "").split(",") if o.strip())
    s.trust_proxy_headers = (_env("AIRSPACE_TRUST_PROXY", "0") == "1")
    import ipaddress as _ip
    s.trusted_proxies = tuple(_ip.ip_network(c.strip(), strict=False) for c in _env("AIRSPACE_TRUSTED_PROXIES", "").split(",") if c.strip())
    s.trusted_proxy_hops = max(1, min(5, int(_env("AIRSPACE_TRUSTED_PROXY_HOPS", "1") or "1")))
    if s.trust_proxy_headers and not s.trusted_proxies:
        raise ValueError("AIRSPACE_TRUST_PROXY=1 requires AIRSPACE_TRUSTED_PROXIES (CIDRs of the gateway)")

    s.require_https = (_env("AIRSPACE_REQUIRE_HTTPS", "1" if s.production else "0") == "1")
    s.challenge_after = max(1, min(20, int(_env("AIRSPACE_CHALLENGE_AFTER", "3") or "3")))
    s.challenge_provider = (_env("AIRSPACE_CHALLENGE_PROVIDER", "pow") or "pow").lower()
    if s.challenge_provider not in ("pow", "turnstile", "hcaptcha"):
        raise ConfigError("AIRSPACE_CHALLENGE_PROVIDER must be pow, turnstile or hcaptcha")
    s.challenge_secret = _env("AIRSPACE_CHALLENGE_SECRET", "") or ""
    if s.challenge_provider != "pow" and not s.challenge_secret:
        raise ConfigError("a hosted challenge provider needs AIRSPACE_CHALLENGE_SECRET")
    s.pow_difficulty = max(8, min(24, int(_env("AIRSPACE_POW_DIFFICULTY", "18") or "18")))

    data_keys = _env("AIRSPACE_DATA_KEYS")         # "kid2:<base64 32 bytes>,kid1:<...>" newest first
    if data_keys:
        for i, item in enumerate(data_keys.split(",")):
            kid, b64 = item.strip().split(":", 1)
            key = base64.b64decode(b64)
            if len(key) != 32:
                raise ConfigError("each data key must be 32 bytes (AES-256)")
            s.data_keys[kid] = key
            if i == 0:
                s.current_data_kid = kid
    elif s.production:
        raise ConfigError("production requires AIRSPACE_DATA_KEYS")
    else:
        # development: persist one key next to the dev database so the CLI and the server agree
        kpath = s.db_path + ".datakey"
        if not os.path.exists(kpath):
            fd = os.open(kpath, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as fh:
                fh.write(base64.b64encode(secrets.token_bytes(32)).decode())
        with open(kpath, "r", encoding="utf-8") as fh:
            s.data_keys = {"dev": base64.b64decode(fh.read().strip())}
        s.current_data_kid = "dev"

    pepper = _env("AIRSPACE_PEPPER")
    keys_spec = _env("AIRSPACE_JWT_KEYS")          # "kid1:<base64 PEM>,kid0:<base64 PEM>" newest first
    evidence_pem = _env("AIRSPACE_EVIDENCE_KEY")   # base64 PEM

    if s.production and not (pepper and keys_spec and evidence_pem):
        raise ConfigError("production requires AIRSPACE_PEPPER, AIRSPACE_JWT_KEYS and AIRSPACE_EVIDENCE_KEY")
    if s.production and any(o == "*" or not o.startswith("https://") for o in s.cors_origins):
        raise ConfigError("production CORS origins must be explicit https origins")

    s.pepper = base64.b64decode(pepper) if pepper else secrets.token_bytes(32)
    if len(s.pepper) < 32:
        raise ConfigError("pepper must be >= 32 bytes")
    if keys_spec:
        for i, item in enumerate(keys_spec.split(",")):
            kid, b64 = item.split(":", 1)
            s.jwt_keys[kid] = _load_key(base64.b64decode(b64).decode())
            if i == 0:
                s.current_kid = kid
    else:
        kid = "dev-" + secrets.token_hex(4)
        s.jwt_keys[kid] = Ed25519PrivateKey.generate()
        s.current_kid = kid
    s.evidence_key = _load_key(base64.b64decode(evidence_pem).decode()) if evidence_pem else Ed25519PrivateKey.generate()
    return s


def generate_data_key_b64() -> str:
    """Helper for operators: a new AES-256 data key, base64."""
    return base64.b64encode(secrets.token_bytes(32)).decode()


def generate_key_b64() -> str:
    """Helper for operators: a new Ed25519 private key as base64(PEM)."""
    pem = Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    return base64.b64encode(pem).decode()
