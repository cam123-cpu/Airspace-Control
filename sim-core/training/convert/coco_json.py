"""COCO detection JSON (images / annotations / categories, bbox = [x, y, w, h]).

Sequence = images[].video_id / sequence if present, else the image's parent folder."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .base import Box, SchemaError, Writer, cls_of, index_images, read_image


def convert(src: Path, out_root: Path, ds: dict, class_map: dict | None = None, frame_step: int = 1,
            keep_empty: bool = True, json_glob: str = "*.json", **_) -> list:
    src = Path(src)
    files = [p for p in sorted(src.rglob(json_glob)) if p.stat().st_size > 0]
    w = Writer(out_root, ds, keep_empty=keep_empty)
    imgs = index_images(src)
    used = 0
    for jp in files:
        with open(jp, "r", encoding="utf-8") as f:
            try:
                coco = json.load(f)
            except json.JSONDecodeError as e:
                raise SchemaError(f"{jp}: bad JSON: {e}") from None
        if not (isinstance(coco, dict) and {"images", "annotations", "categories"} <= coco.keys()):
            continue
        used += 1
        cats = {c["id"]: c.get("name", "other") for c in coco["categories"]}
        anns = defaultdict(list)
        for a in coco["annotations"]:
            if a.get("iscrowd"):
                continue
            bb = a.get("bbox")
            if not (isinstance(bb, list) and len(bb) == 4):
                raise SchemaError(f"{jp}: annotation {a.get('id')} has no [x,y,w,h] bbox")
            anns[a["image_id"]].append(Box.from_xywh(cls_of(cats.get(a["category_id"], "other"), class_map), *bb))
        for k, im in enumerate(sorted(coco["images"], key=lambda i: str(i.get("file_name")))):
            if k % max(1, frame_step):
                continue
            fn = str(im.get("file_name", ""))
            cand = [jp.parent / fn, jp.parent / "images" / fn, jp.parent.parent / "images" / fn]
            ip = next((c for c in cand if c.is_file()), None) or imgs.get(Path(fn).name)
            if ip is None:
                raise SchemaError(f"{jp}: image '{fn}' not found")
            seq = str(im.get("video_id") or im.get("sequence") or Path(fn).parent.as_posix() or jp.stem)
            w.add(read_image(ip), anns.get(im["id"], []), f"{jp.stem}/{seq}", int(im.get("frame_id", k)),
                  name=Path(fn).stem)
    if not used:
        raise SchemaError(f"no COCO-style json (images/annotations/categories) under {src}")
    return w.rows
