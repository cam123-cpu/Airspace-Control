"""Password policy, applied wherever an account password is set (API, CLI bootstrap, auditor creation).

Length beats composition rules (NIST SP 800-63B), so the policy is: 12 to 256 characters, and not
guessable from the obvious places an attacker starts: a common base word padded with digits or symbols,
the account's own email, or a run of one or two characters. No "must contain a symbol" rules.
"""
from __future__ import annotations

import re
import unicodedata

from .errors import Invalid

MIN_LEN, MAX_LEN = 12, 256

# Base words at the top of public breach corpora, plus words an attacker would try against this product.
_COMMON = frozenset("""
password passw0rd passwort motdepasse contrasena qwerty qwertyuiop asdfgh asdfghjkl zxcvbnm azerty
letmein welcome admin administrator root login iloveyou monkey dragon football baseball soccer hockey
sunshine princess master shadow superman batman trustno starwars whatever freedom secret changeme default
abc abcdef abcdefg abcdefgh test tester testing guest user summer winter spring autumn
airspace prison corrections correctional warden officer drone counterdrone security
""".split())

_LEET = str.maketrans("013457@$!", "oleastasi")


def _letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s)


def _cores(p: str) -> set[str]:
    """The base word under the padding, read two ways: digits and symbols dropped, or the padding trimmed
    from both ends and leetspeak inside it undone (p@ssw0rd!2026 -> password)."""
    inner = re.sub(r"^[^a-z]+|[^a-z]+$", "", p)
    return {_letters(p), _letters(inner.translate(_LEET))}


def _is_common(core: str) -> bool:
    return bool(core) and (core in _COMMON) or bool(core) and any(
        len(core) % len(w) == 0 and core == w * (len(core) // len(w)) for w in _COMMON)


def check_password(password: str, email: str | None = None) -> None:
    if not isinstance(password, str) or not (MIN_LEN <= len(password) <= MAX_LEN):
        raise Invalid(f"password must be {MIN_LEN} to {MAX_LEN} characters")
    p = unicodedata.normalize("NFKC", password).casefold()
    if len(set(p)) < 6:
        raise Invalid("password repeats too few characters")
    if not any(ch.isalpha() for ch in p):
        raise Invalid("password needs at least one word, not only digits and symbols")
    cores = _cores(p)
    if any(_is_common(c) for c in cores):
        raise Invalid("password is a common word padded with digits or symbols")
    if email:
        local, _, domain = email.casefold().partition("@")
        parts = [local] + domain.split(".")[:-1]
        if any(len(x) >= 4 and (x in p or (_letters(x) and any(_letters(x) in c for c in cores))) for x in parts):
            raise Invalid("password must not contain the account's email")
