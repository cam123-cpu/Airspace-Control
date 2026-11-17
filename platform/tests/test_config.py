"""Fail-closed configuration."""
import base64

import pytest

from airspace.config import ConfigError, generate_data_key_b64, generate_key_b64, load_settings


@pytest.fixture(autouse=True)
def _tmp_db(monkeypatch, tmp_path):
    # development mode persists a data key next to the database: keep it out of the source tree
    monkeypatch.setenv("AIRSPACE_DB_PATH", str(tmp_path / "cfg.db"))


def _clear(monkeypatch):
    for k in ["AIRSPACE_ENV", "AIRSPACE_PEPPER", "AIRSPACE_JWT_KEYS", "AIRSPACE_EVIDENCE_KEY", "AIRSPACE_CORS_ORIGINS",
              "AIRSPACE_DATA_KEYS", "AIRSPACE_REQUIRE_HTTPS", "AIRSPACE_TRUST_PROXY", "AIRSPACE_CHALLENGE_PROVIDER"]:
        monkeypatch.delenv(k, raising=False)


def _prod_secrets(monkeypatch):
    monkeypatch.setenv("AIRSPACE_ENV", "production")
    monkeypatch.setenv("AIRSPACE_PEPPER", base64.b64encode(b"p" * 32).decode())
    monkeypatch.setenv("AIRSPACE_JWT_KEYS", "k1:" + generate_key_b64())
    monkeypatch.setenv("AIRSPACE_EVIDENCE_KEY", generate_key_b64())
    monkeypatch.setenv("AIRSPACE_DATA_KEYS", "d1:" + generate_data_key_b64())


def test_production_requires_secrets(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("AIRSPACE_ENV", "production")
    with pytest.raises(ConfigError):
        load_settings()


def test_production_rejects_wildcard_or_http_cors(monkeypatch):
    _clear(monkeypatch)
    _prod_secrets(monkeypatch)
    for bad in ("*", "http://console.example"):
        monkeypatch.setenv("AIRSPACE_CORS_ORIGINS", bad)
        with pytest.raises(ConfigError):
            load_settings()
    monkeypatch.setenv("AIRSPACE_CORS_ORIGINS", "https://console.example")
    s = load_settings()
    assert s.production and s.current_kid == "k1"
    assert s.require_https, "production must refuse plain http by default"
    assert s.current_data_kid == "d1"


def test_production_requires_data_keys(monkeypatch):
    _clear(monkeypatch)
    _prod_secrets(monkeypatch)
    monkeypatch.setenv("AIRSPACE_CORS_ORIGINS", "https://console.example")
    monkeypatch.delenv("AIRSPACE_DATA_KEYS")
    with pytest.raises(ConfigError):
        load_settings()


def test_data_keys_must_be_aes256(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("AIRSPACE_DATA_KEYS", "d1:" + base64.b64encode(b"k" * 16).decode())
    with pytest.raises(ConfigError):
        load_settings()


def test_hosted_captcha_needs_secret(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("AIRSPACE_CHALLENGE_PROVIDER", "turnstile")
    monkeypatch.delenv("AIRSPACE_CHALLENGE_SECRET", raising=False)
    with pytest.raises(ConfigError):
        load_settings()


def test_short_pepper_rejected(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("AIRSPACE_PEPPER", base64.b64encode(b"short").decode())
    with pytest.raises(ConfigError):
        load_settings()


def test_rotation_ring_order(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("AIRSPACE_JWT_KEYS", f"new:{generate_key_b64()},old:{generate_key_b64()}")
    s = load_settings()
    assert s.current_kid == "new" and set(s.jwt_keys) == {"new", "old"}
