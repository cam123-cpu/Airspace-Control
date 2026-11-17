"""Halmstad multi-sensor drone dataset. Its labels are MATLAB groundTruth objects (.mat
class instances that scipy cannot read), so run export_halmstad_labels.m once in MATLAB
to write labels.csv:

    video,frame,class,x,y,w,h        (frame 1-based, pixels)

then this converter extracts the frames from the matching IR_/V_ .mp4 files."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .base import VID_EXT, Box, SchemaError, Writer, cls_of, iter_video

REQUIRED = {"video", "frame", "class", "x", "y", "w", "h"}


def convert(src: Path, out_root: Path, ds: dict, csv_path: Path | None = None, class_map: dict | None = None,
            frame_step: int = 3, keep_empty: bool = True, **_) -> list:
    src = Path(src)
    csv_path = Path(csv_path) if csv_path else next(iter(sorted(src.rglob("labels.csv"))), None)
    if csv_path is None or not csv_path.is_file():
        raise SchemaError("labels.csv not found: export it with export_halmstad_labels.m first")
    ann: dict[str, dict[int, list[Box]]] = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if not REQUIRED <= set(rd.fieldnames or []):
            raise SchemaError(f"{csv_path}: needs columns {sorted(REQUIRED)}")
        for r in rd:
            v, fr = Path(r["video"]).stem, int(r["frame"])
            if r["class"] and float(r["w"]) > 0 and float(r["h"]) > 0:
                ann[v][fr].append(Box.from_xywh(cls_of(r["class"], class_map), r["x"], r["y"], r["w"], r["h"]))
            else:
                ann[v].setdefault(fr, [])
    vids = {p.stem: p for p in src.rglob("*") if p.suffix.lower() in VID_EXT}
    writers: dict[str, Writer] = {}
    for stem, frames in sorted(ann.items()):
        vp = vids.get(stem)
        if vp is None:
            raise SchemaError(f"video for '{stem}' listed in labels.csv not found under {src}")
        mod = "ir" if stem.upper().startswith("IR") else "rgb"
        wr = writers.setdefault(mod, Writer(out_root, ds, modality=mod, keep_empty=keep_empty))
        for fr, img in iter_video(vp, frame_step, wanted=set(frames), start_index=1):
            wr.add(img, frames[fr], stem, fr)
    return [r for wr in writers.values() for r in wr.rows]
