"""Hide keys / purge secrets: fail the build if anything key-shaped is committed anywhere in the repository.
CI and the pre-commit hook also run gitleaks over the full git history; this test is the fast local guard."""
from __future__ import annotations

import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SKIP_DIRS = {".git", "node_modules", "vendor", "__pycache__", ".pytest_cache", "data", "cache", ".venv", "venv"}
SKIP_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".onnx", ".npy", ".zip", ".gz", ".pdf", ".woff", ".woff2"}

# Patterns are assembled from pieces so this file doesn't match itself.
B = "-----" + "BEGIN"
PATTERNS = {
    "pem private key": re.compile(B + r" (RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY"),
    "base64 pem private key": re.compile("LS0tLS1CRUdJTi" + r"BQUklWQVRF|LS0tLS1CRUdJTi" + "BFQyBQUklWQVRF"),
    "aws access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github token": re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})\b"),
    "slack token": re.compile(r"\bxox[abpors]-[A-Za-z0-9-]{10,}"),
    "stripe live key": re.compile(r"\b[sr]k_live_[A-Za-z0-9]{20,}"),
    "google api key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{20,}"),
    "hardcoded platform secret": re.compile(r"AIRSPACE_(PEPPER|DATA_KEYS|JWT_KEYS|EVIDENCE_KEY|CHALLENGE_SECRET)\s*=\s*['\"]?[A-Za-z0-9+/]{30,}"),
    "db url with password": re.compile(r"postgres(ql)?://[^:/\s]+:(?!postgres@)[^@\s]{6,}@"),
}


def _files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            if os.path.splitext(f)[1].lower() in SKIP_EXT:
                continue
            p = os.path.join(dirpath, f)
            if os.path.getsize(p) > 5_000_000:
                continue
            yield p


def test_no_secrets_in_repository():
    hits = []
    for p in _files():
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except OSError:
            continue
        for name, rx in PATTERNS.items():
            m = rx.search(text)
            if m:
                hits.append(f"{os.path.relpath(p, ROOT)}: {name}")
    assert not hits, "secret-like content committed:\n" + "\n".join(hits)


def test_env_example_has_only_placeholders():
    p = os.path.join(ROOT, "platform", ".env.example")
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if "=" in line and not line.lstrip().startswith("#"):
                key, val = line.split("=", 1)
                if any(w in key for w in ("PEPPER", "KEY", "SECRET", "PASSWORD")):
                    v = val.strip()
                    assert "<" in v or v in ("", "0", "1"), key                   # placeholders only
                    assert not re.search(r"[A-Za-z0-9+/=]{30,}", v), key          # no real key material
