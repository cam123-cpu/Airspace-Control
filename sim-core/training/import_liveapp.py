"""Convert a Drone Watch Live label export (JSON) into a YOLO dataset of your own footage.

Usage:
    python3 -m training.import_liveapp export.json --out /data/raw/own_footage

Event items (one object, a 128x128 crop centred on it) become crop images with one box covering the
object's centre region. "Frame" items (full frames saved with every tracked object) become full images
with one box per object. Items labelled "false alarm" become negative images (empty label file),
which is exactly what a detector needs to stop repeating that mistake.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import sys
from pathlib import Path

CLASSES = ["drone", "drone_payload", "bird", "other"]
MAX_ITEMS = 100_000
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def _decode_jpeg(data_url: str) -> bytes:
    prefix = "data:image/jpeg;base64,"
    if not isinstance(data_url, str) or not data_url.startswith(prefix):
        raise ValueError("image is not a base64 JPEG data URL")
    raw = base64.b64decode(data_url[len(prefix):], validate=True)
    if len(raw) > MAX_IMAGE_BYTES or raw[:2] != b"\xff\xd8":
        raise ValueError("image too large or not a JPEG")
    return raw


def _box_ok(b) -> bool:
    return (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)
            and 0 <= b[0] <= 1 and 0 <= b[1] <= 1 and 0 < b[2] <= 1 and 0 < b[3] <= 1)


def convert(export_path: Path, out_dir: Path) -> dict:
    data = json.loads(export_path.read_text(encoding="utf-8"))
    if data.get("app") != "drone-watch-live":
        raise ValueError("not a Drone Watch Live export")
    items = data.get("items", [])[:MAX_ITEMS]
    img_dir, lbl_dir = out_dir / "images", out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "boxes": 0, "negatives": 0, "skipped": 0}
    stem_base = export_path.stem.replace(os.sep, "_")[:60]
    for i, it in enumerate(items):
        try:
            jpg = _decode_jpeg(it.get("image_jpeg_dataurl", ""))
        except (ValueError, binascii.Error):
            stats["skipped"] += 1
            continue
        lines = []
        if it.get("objects"):  # full frame with several objects
            for o in it["objects"]:
                lab = o.get("label")
                if lab in CLASSES and _box_ok(o.get("box")):
                    lines.append(f"{CLASSES.index(lab)} " + " ".join(f"{v:.5f}" for v in o["box"]))
        else:  # 128x128 crop centred on one object (snapshot = 4x the object size)
            lab = it.get("label")
            if lab == "false alarm":
                stats["negatives"] += 1
            elif lab in CLASSES:
                lines.append(f"{CLASSES.index(lab)} 0.50000 0.50000 0.25000 0.25000")
            else:
                stats["skipped"] += 1
                continue
        stem = f"{stem_base}_{i:05d}"
        (img_dir / f"{stem}.jpg").write_bytes(jpg)
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats["images"] += 1
        stats["boxes"] += len(lines)
    (out_dir / "SOURCE.txt").write_text(
        "Own footage labelled in Drone Watch Live. License: owned by you (commercial_ok: yes).\n", encoding="utf-8")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("export", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if not a.export.is_file():
        print(f"not found: {a.export}", file=sys.stderr)
        return 2
    print(json.dumps(convert(a.export, a.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
