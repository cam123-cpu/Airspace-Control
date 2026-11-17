"""Per-sequence rectangle files: OTB / tracking style (DUT Anti-UAV tracking, CST Anti-UAV,
USC drone). One folder per sequence holding frames (or a video) plus a ground-truth txt.

Options
    gt_name   file name of the ground truth inside each sequence folder (default groundtruth*.txt)
    fmt       xywh (default) | xyxy | cxcywh   (USC uses cxcywh, pixels)
    frame_col True when each line starts with a frame number (USC); otherwise line i = frame i
    cls       training class for every box (single-target sets), default drone
Blank, NaN or zero-size rows mean 'no target in this frame' (kept as negatives)."""
from __future__ import annotations

import math
import re
from pathlib import Path

from .base import IMG_EXT, VID_EXT, Box, SchemaError, Writer, iter_video, read_image

_SPLIT = re.compile(r"[,\s;]+")


def parse_lines(text: str, fmt: str = "xywh", frame_col: bool = False, cls: str = "drone") -> dict[int, list[Box]]:
    out: dict[int, list[Box]] = {}
    for ln, line in enumerate(text.splitlines()):
        t = [v for v in _SPLIT.split(line.strip()) if v]
        if not t:
            out.setdefault(ln, [])
            continue
        try:
            vals = [float(v) for v in t]
        except ValueError:
            if ln == 0:
                continue  # header row
            raise SchemaError(f"line {ln + 1}: non-numeric '{line.strip()[:60]}'") from None
        fr = int(vals.pop(0)) if frame_col else ln
        if len(vals) < 4 or any(math.isnan(v) for v in vals[:4]) or vals[2] <= 0 or vals[3] <= 0:
            out.setdefault(fr, [])
            continue
        a, b, c, d = vals[:4]
        box = {"xywh": Box.from_xywh, "cxcywh": Box.from_cxcywh,
               "xyxy": lambda k, *v: Box(k, *v)}[fmt](cls, a, b, c, d)
        out.setdefault(fr, []).append(box)
    return out


def convert(src: Path, out_root: Path, ds: dict, fmt: str = "xywh", frame_col: bool = False,
            cls: str = "drone", gt_name: str = "groundtruth*.txt", frame_step: int = 1,
            keep_empty: bool = True, **_) -> list:
    if fmt not in ("xywh", "xyxy", "cxcywh"):
        raise ValueError("fmt must be xywh, xyxy or cxcywh")
    src = Path(src)
    gts = sorted(src.rglob(gt_name))
    if not gts:
        raise SchemaError(f"no '{gt_name}' files under {src}")
    w = Writer(out_root, ds, keep_empty=keep_empty)
    for gt in gts:
        ann = parse_lines(gt.read_text(encoding="utf-8", errors="replace"), fmt, frame_col, cls)
        seq_dir = gt.parent
        seq = seq_dir.relative_to(src).as_posix() or gt.stem
        frames = sorted(p for p in seq_dir.rglob("*") if p.suffix.lower() in IMG_EXT and p.is_file())
        if frames:
            for i, fp in enumerate(frames):
                fr = i
                if frame_col:  # frame numbers in the file refer to the number in the image name
                    m = re.findall(r"\d+", fp.stem)
                    fr = int(m[-1]) if m else i
                if fr not in ann or i % max(1, frame_step):
                    continue
                w.add(read_image(fp), ann[fr], seq, fr, name=fp.stem)
            continue
        vids = sorted(p for p in seq_dir.iterdir() if p.suffix.lower() in VID_EXT)
        if not vids:
            raise SchemaError(f"{seq_dir}: no frames or video next to {gt.name}")
        for fr, img in iter_video(vids[0], frame_step, wanted=set(ann)):
            w.add(img, ann[fr], seq, fr)
    return w.rows
