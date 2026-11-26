"""Password policy: long passphrases pass; padded common words, the account's email and low-variety strings fail."""
import pytest

from airspace.errors import Invalid
from airspace.passwords import check_password

GOOD = ["correct horse battery staple 9", "tangerine-orbit-gravel-42", "Ü-bahn nach Kreuzberg 1989", "x7#Qm!29vLpa", "погода в москве сегодня"]
BAD = [
    ("short", None),
    ("Password1234!", None),          # common word + padding
    ("P@ssw0rd!2026", None),          # leetspeak of a common word
    ("qwertyuiop12", None),
    ("passwordpassword", None),       # repeated common word
    ("aaaaaaaaaaaab", None),          # too few distinct characters
    ("121212121212", None),           # digits only
    ("Airspace2026!!", None),         # product word
    ("jordan.smith!2026", "jordan.smith@doc.state.example"),   # contains email local part
    ("statedoc!!2026xx", "ops@statedoc.example"),                # contains email domain label
    ("x" * 257, None),
]


@pytest.mark.parametrize("pw", GOOD)
def test_good_passwords_pass(pw):
    check_password(pw, "someone@a.example")


@pytest.mark.parametrize("pw,email", BAD)
def test_weak_passwords_fail(pw, email):
    with pytest.raises(Invalid):
        check_password(pw, email)


def test_api_rejects_weak_password(env):
    r = env.post("/api/v1/users", headers=env.h(env.A, "tenant_admin"),
                 json={"email": "weak@a.example", "role": "viewer", "password": "Password1234!"})
    assert r.status_code == 422
