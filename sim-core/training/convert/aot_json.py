"""Amazon Airborne Object Tracking (AOT). Each training part has
    ImageSets/groundtruth.json  and  Images/<flight_id>/<img_name>.png

Assumed groundtruth layout (SCHEMA UNVERIFIED - check docs/DATASET.md of the AIcrowd starter
kit linked from registry.opendata.aws/airborne-object-tracking; the converter validates
every key it reads):

    {"samples": {"<flight_id>": {
        "metadata": {...},
        "entities": [{"img_name": "...png", "bb": [x, y, w, h], "id": "Airplane1",
                      "blob": {"frame": 12}, "labels": {"is_above_horizon": 1}}, ...]}}}

Entity ids start with the object type (Airplane, Helicopter, Bird, Drone, Flock, ...).
Frames with entities but no "bb" are skipped. Images are 2448x2048 grayscale: tile them
in build_mix (--tile 1280) or the drones are only a few pixels wide."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import cv2

from .base import Box, SchemaError, Writer, cls_of

_TYPE = re.compile(r"^[A-Za-z]+")


def convert(src: Path, out_root: Path, ds: dict, class_map: dict | None = None, frame_step: int = 10,
            keep_empty: bool = False, max_flights: int | None = None, **_) -> list:
    src = Path(src)
    gts = sorted(src.rglob("groundtruth.json"))
    if not gts:
        raise SchemaError(f"no groundtruth.json under {src}")
    w = Writer(out_root, ds, modality="gray", keep_empty=keep_empty)
    flights = 0
    for gt in gts:
        d = json.loads(gt.read_text(encoding="utf-8"))
        samples = d.get("samples") if isinstance(d, dict) else None
        if not isinstance(samples, dict):
            raise SchemaError(f"{gt}: expected top-level 'samples' mapping")
        img_root = gt.parent.parent / "Images"
        for fid, s in samples.items():
            if max_flights is not None and flights >= max_flights:
                return w.rows
            flights += 1
            ents = s.get("entities")
            if not isinstance(ents, list):
                raise SchemaError(f"{gt}: flight {fid} has no 'entities' list")
            per_img: dict[str, list[Box]] = defaultdict(list)
            for e in ents:
                name = e.get("img_name")
                if not name:
                    raise SchemaError(f"{gt}: entity without img_name in flight {fid}")
                bb = e.get("bb")
                per_img.setdefault(name, [])
                if isinstance(bb, list) and len(bb) == 4:
                    m = _TYPE.match(str(e.get("id", "other")))
                    per_img[name].append(Box.from_xywh(cls_of(m.group(0) if m else "other", class_map), *bb))
            for k, name in enumerate(sorted(per_img)):
                if k % max(1, frame_step):
                    continue
                ip = img_root / fid / name
                if not ip.is_file():
                    continue  # partial download: only some flights fetched
                img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise SchemaError(f"cannot read {ip}")
                w.add(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), per_img[name], fid, k, name=Path(name).stem)
    return w.rows
