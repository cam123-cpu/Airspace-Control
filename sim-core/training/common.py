"""Shared helpers: catalog loading, license policy, safe paths, safe archive extraction,
and the JSONL manifest every downloaded/converted file is recorded in."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tarfile
import zipfile
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Iterator

import yaml

HERE = Path(__file__).resolve().parent
CATALOG_PATH = HERE / "datasets.yaml"
CLASSES = ["drone", "drone_payload", "bird", "other"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
COMMERCIAL_VALUES = {"yes", "no", "unclear", "permission_required"}
ACCESS_VALUES = {"direct", "manual", "request", "unavailable"}

# Source class names (lower-cased) -> training class. Anything unknown -> "other".
DEFAULT_CLASS_MAP = {
    "drone": "drone", "uav": "drone", "mav": "drone", "quadcopter": "drone", "target": "drone",
    "drone_payload": "drone_payload", "drone+payload": "drone_payload",
    "bird": "bird", "birds": "bird", "flock": "bird", "n01503061": "bird",
    "airplane": "other", "aeroplane": "other", "plane": "other", "helicopter": "other",
    "aircraft": "other", "balloon": "other", "bag": "other", "other": "other",
}


class CatalogError(ValueError):
    pass


def load_catalog(path: Path = CATALOG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cat = yaml.safe_load(f)  # never yaml.load: no arbitrary object construction
    validate_catalog(cat)
    return cat


def validate_catalog(cat: dict) -> None:
    if not isinstance(cat, dict) or "datasets" not in cat:
        raise CatalogError("catalog must be a mapping with a 'datasets' list")
    seen = set()
    for d in cat["datasets"]:
        for k in ("id", "name", "license", "commercial_ok", "access"):
            if k not in d:
                raise CatalogError(f"dataset entry missing '{k}': {d.get('id', d)}")
        if d["id"] in seen:
            raise CatalogError(f"duplicate id {d['id']}")
        seen.add(d["id"])
        if str(d["commercial_ok"]) not in COMMERCIAL_VALUES:
            raise CatalogError(f"{d['id']}: commercial_ok must be one of {sorted(COMMERCIAL_VALUES)}")
        if d["access"] not in ACCESS_VALUES:
            raise CatalogError(f"{d['id']}: access must be one of {sorted(ACCESS_VALUES)}")


def dataset_by_id(cat: dict, ds_id: str) -> dict:
    for d in cat["datasets"]:
        if d["id"] == ds_id:
            return d
    raise CatalogError(f"unknown dataset id '{ds_id}'")


def commercial_ok(cat: dict, ds_id: str) -> bool:
    return str(dataset_by_id(cat, ds_id)["commercial_ok"]) == "yes"


def map_class(name: str, class_map: dict | None = None) -> str:
    m = {**DEFAULT_CLASS_MAP, **{k.lower(): v for k, v in (class_map or {}).items()}}
    out = m.get(str(name).strip().lower(), "other")
    if out not in CLASS_ID:
        raise ValueError(f"class map points to unknown training class '{out}'")
    return out


# ------------------------------------------------------------------ paths

def safe_join(root: Path, *parts: str) -> Path:
    """Join and refuse anything that escapes root (../, absolute paths, symlink tricks)."""
    root = Path(root).resolve()
    p = root.joinpath(*parts).resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"path escapes {root}: {'/'.join(parts)}")
    return p


def safe_name(s: str, maxlen: int = 120) -> str:
    keep = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s))
    keep = keep.strip("._") or "x"
    return keep[:maxlen]


# ------------------------------------------------------------------ archives

MAX_EXTRACT_BYTES = 2 * 1024 ** 4  # 2 TiB hard ceiling against zip bombs
MAX_MEMBERS = 5_000_000


def safe_extract(archive: Path, dest: Path, max_bytes: int = MAX_EXTRACT_BYTES) -> list[Path]:
    """Extract zip/tar without path traversal, absolute paths, links or device files."""
    archive, dest = Path(archive), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    total = 0
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            infos = z.infolist()
            if len(infos) > MAX_MEMBERS:
                raise ValueError("archive has too many members")
            for info in infos:
                mode = (info.external_attr >> 16) & 0o170000
                if mode == stat.S_IFLNK:
                    raise ValueError(f"refusing symlink in zip: {info.filename}")
                target = safe_join(dest, info.filename)
                total += info.file_size
                if total > max_bytes:
                    raise ValueError("archive expands beyond the size limit")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
                out.append(target)
        return out
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            members = t.getmembers()
            if len(members) > MAX_MEMBERS:
                raise ValueError("archive has too many members")
            for m in members:
                if not (m.isfile() or m.isdir()):
                    raise ValueError(f"refusing link/device in tar: {m.name}")
                target = safe_join(dest, m.name)
                total += m.size
                if total > max_bytes:
                    raise ValueError("archive expands beyond the size limit")
                if m.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                src = t.extractfile(m)
                if src is None:
                    continue
                with src, open(target, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
                os.chmod(target, 0o644)
                out.append(target)
        return out
    raise ValueError(f"not a zip or tar archive: {archive}")


def file_md5(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)  # matches publisher-provided MD5 checksums; never used for security decisions
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ manifest

@dataclass
class ManifestRow:
    """One labelled image (or one unlabelled background frame) and where it came from."""
    image: str               # path relative to the manifest's root
    label: str | None        # YOLO txt relative to root, None for unlabelled plates
    source: str              # catalog id
    license: str
    commercial_ok: str
    sequence: str            # video / flight / folder id: the unit splits are made over
    modality: str = "rgb"    # rgb | ir | gray
    context: str = ""        # free text: night, urban, sea, ...
    frame: int = -1
    extra: dict = field(default_factory=dict)


def write_manifest(rows: Iterable[ManifestRow], path: Path, append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_manifest(path: Path) -> Iterator[ManifestRow]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            try:
                yield ManifestRow(**d)
            except TypeError as e:
                raise ValueError(f"{path}:{i}: bad manifest row: {e}") from None
