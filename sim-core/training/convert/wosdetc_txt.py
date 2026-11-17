"""Drone-vs-Bird (WOSDETC): one annotation txt per video, same stem as the video.

Line format (from github.com/wosdetc/challenge):
    framenum num_objs_in_frame obj1_x_left obj1_y_top obj1_w obj1_h obj1_class ...
Frames absent from the txt are treated as unlabelled and skipped (not as empty)."""
from __future__ import annotations

from pathlib import Path

from .base import VID_EXT, Box, SchemaError, Writer, cls_of, iter_video


def parse(txt: Path, class_map: dict | None = None) -> dict[int, list[Box]]:
    out: dict[int, list[Box]] = {}
    for ln, line in enumerate(txt.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        t = line.split()
        if not t:
            continue
        try:
            fr, n = int(t[0]), int(t[1])
        except (ValueError, IndexError):
            raise SchemaError(f"{txt}:{ln}: expected 'framenum num_objs ...'") from None
        if len(t) < 2 + 5 * n:
            raise SchemaError(f"{txt}:{ln}: {n} objects declared but only {(len(t) - 2) // 5} present")
        boxes = []
        for i in range(n):
            x, y, w, h, c = t[2 + 5 * i: 7 + 5 * i]
            boxes.append(Box.from_xywh(cls_of(c, class_map), x, y, w, h))
        out[fr] = boxes
    return out


def convert(src: Path, out_root: Path, ds: dict, class_map: dict | None = None, frame_step: int = 5,
            keep_empty: bool = True, start_index: int = 0, **_) -> list:
    src = Path(src)
    vids = sorted(p for p in src.rglob("*") if p.suffix.lower() in VID_EXT)
    if not vids:
        raise SchemaError(f"no videos under {src}")
    w = Writer(out_root, ds, keep_empty=keep_empty)
    paired = 0
    for vp in vids:
        cands = [vp.with_suffix(".txt"), *vp.parent.rglob(vp.stem + ".txt"), *src.rglob(vp.stem + ".txt")]
        txt = next((c for c in cands if c.is_file()), None)
        if txt is None:
            continue
        paired += 1
        ann = parse(txt, class_map)
        for fr, img in iter_video(vp, frame_step, wanted=set(ann), start_index=start_index):
            w.add(img, ann[fr], vp.stem, fr)
    if not paired:
        raise SchemaError(f"no video has a matching <video_stem>.txt annotation under {src}")
    return w.rows
