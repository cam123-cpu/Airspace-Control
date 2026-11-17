import hashlib
import io
import os
import tarfile
import zipfile
from pathlib import Path

import pytest

from training import common, fetch


def test_catalog_loads_and_is_consistent():
    cat = common.load_catalog()
    ids = [d["id"] for d in cat["datasets"]]
    assert len(ids) == len(set(ids)) >= 20
    for d in cat["datasets"]:
        assert str(d["commercial_ok"]) in common.COMMERCIAL_VALUES
        if d.get("converter"):
            from training import convert
            convert.get(d["converter"])  # every referenced converter exists
        if d["access"] == "direct" and d.get("fetch"):
            assert d["fetch"]["method"] in fetch.METHODS


def test_catalog_rejects_bad_values():
    with pytest.raises(common.CatalogError):
        common.validate_catalog({"datasets": [{"id": "a", "name": "a", "license": "x", "commercial_ok": "maybe",
                                               "access": "direct"}]})


def test_class_map():
    assert common.map_class("UAV") == "drone"
    assert common.map_class("Flock") == "bird"
    assert common.map_class("Helicopter") == "other"
    assert common.map_class("zeppelin") == "other"


def test_safe_join_blocks_traversal(tmp_path):
    assert common.safe_join(tmp_path, "a", "b.txt").parent.parent == tmp_path.resolve()
    with pytest.raises(ValueError):
        common.safe_join(tmp_path, "../evil")
    with pytest.raises(ValueError):
        common.safe_join(tmp_path, "/etc/passwd")


def test_safe_extract_zip_ok_and_traversal(tmp_path):
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("dir/a.txt", "hello")
    out = common.safe_extract(good, tmp_path / "x")
    assert (tmp_path / "x" / "dir" / "a.txt").read_text() == "hello" and len(out) == 1
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../../escape.txt", "x")
    with pytest.raises(ValueError):
        common.safe_extract(bad, tmp_path / "y")
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_tar_rejects_symlink(tmp_path):
    t = tmp_path / "bad.tar"
    with tarfile.open(t, "w") as tf:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    with pytest.raises(ValueError):
        common.safe_extract(t, tmp_path / "z")


def test_safe_extract_size_cap(tmp_path):
    z = tmp_path / "big.zip"
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("zeros.bin", b"\0" * 100_000)
    with pytest.raises(ValueError):
        common.safe_extract(z, tmp_path / "o", max_bytes=1000)


def test_manifest_roundtrip(tmp_path):
    r = common.ManifestRow(image="i.jpg", label="l.txt", source="s", license="L", commercial_ok="yes", sequence="q")
    p = tmp_path / "m.jsonl"
    common.write_manifest([r, r], p)
    assert list(common.read_manifest(p)) == [r, r]


def test_check_url():
    fetch.check_url("https://zenodo.org/api/records/1")
    fetch.check_url("https://ndownloader.figshare.com/files/1")
    for bad in ("http://zenodo.org/x", "https://evil.example.com/x", "https://zenodo.org.evil.com/x", "file:///etc/passwd"):
        with pytest.raises(fetch.FetchError):
            fetch.check_url(bad)


class _Resp:
    def __init__(self, data, status=200, url="https://zenodo.org/f"):
        self.data, self.status_code, self.url = data, status, url

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def iter_content(self, n):
        for i in range(0, len(self.data), 7):
            yield self.data[i:i + 7]


class _Sess:
    def __init__(self, data, url="https://zenodo.org/f"):
        self.data, self.url, self.calls = data, url, 0

    def get(self, url, **kw):
        self.calls += 1
        return _Resp(self.data, url=self.url)


def test_download_md5_ok_and_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "MIN_INTERVAL_S", 0)
    monkeypatch.setattr(fetch, "_throttle", lambda *a, **k: None)
    data = b"dataset bytes" * 10
    md5 = hashlib.md5(data).hexdigest()
    p = fetch.download("https://zenodo.org/f", tmp_path / "f.bin", md5=md5, session=_Sess(data))
    assert p.read_bytes() == data
    with pytest.raises(fetch.FetchError):
        fetch.download("https://zenodo.org/g", tmp_path / "g.bin", md5="0" * 32, session=_Sess(data))
    assert not (tmp_path / "g.bin").exists()


def test_download_rejects_redirect_off_allowlist(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "_throttle", lambda *a, **k: None)
    with pytest.raises(fetch.FetchError):
        fetch.download("https://zenodo.org/f", tmp_path / "f.bin", session=_Sess(b"x", url="https://evil.example.com/f"))


def test_pexels_refuses_without_permission(tmp_path):
    ds = common.dataset_by_id(common.load_catalog(), "pexels_video")
    with pytest.raises(fetch.FetchError, match="machine learning"):
        fetch.fetch_pexels(ds, tmp_path)


def test_pixabay_requires_terms_and_env_key(tmp_path, monkeypatch):
    ds = common.dataset_by_id(common.load_catalog(), "pixabay_video")
    with pytest.raises(fetch.FetchError, match="mass downloads"):
        fetch.fetch_pixabay(ds, tmp_path)
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    with pytest.raises(fetch.FetchError, match="PIXABAY_API_KEY"):
        fetch.fetch_pixabay(ds, tmp_path, accept_pixabay_terms=True)


def test_manual_datasets_print_instructions(tmp_path, capsys):
    ds = common.dataset_by_id(common.load_catalog(), "drone_vs_bird")
    assert fetch.fetch(ds, tmp_path) == []
    assert "wosdetc@googlegroups.com" in capsys.readouterr().out


def test_cli_list(capsys):
    assert fetch.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "aot" in out and "commercial_ok=yes" in out
