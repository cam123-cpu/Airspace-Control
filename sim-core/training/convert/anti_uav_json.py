"""Anti-UAV (300 / 410 / 600). Each sequence folder holds IR.mp4 / RGB.mp4 (300) or frame
images (410/600) plus IR_label.json / RGB_label.json:

    {"exist": [1, 1, 0, ...], "gt_rect": [[x, y, w, h], [x, y, w, h], [], ...]}

SCHEMA UNVERIFIED: taken from the public challenge starter code, not from a licence-bearing
dataset page. The converter checks the keys and lengths and stops with a SchemaError if
they differ, so a mismatch is caught on the first sequence."""
from __future__ import annotations

import json
from pathlib import Path

from .base import IMG_EXT, VID_EXT, Box, SchemaError, Writer, iter_video, read_image


def parse(label: Path) -> list[list[Box]]:
    d = json.loads(label.read_text(encoding="utf-8"))
    if not isinstance(d, dict) or "gt_rect" not in d:
        raise SchemaError(f"{label}: expected keys 'exist' and 'gt_rect'")
    rects = d["gt_rect"]
    exist = d.get("exist", [1 if r else 0 for r in rects])
    if len(exist) != len(rects):
        raise SchemaError(f"{label}: exist ({len(exist)}) and gt_rect ({len(rects)}) lengths differ")
    out = []
    for e, r in zip(exist, rects):
        if e and isinstance(r, list) and len(r) == 4 and r[2] > 0 and r[3] > 0:
            out.append([Box.from_xywh("drone", *r)])
        else:
            out.append([])
    return out


def convert(src: Path, out_root: Path, ds: dict, frame_step: int = 5, keep_empty: bool = True,
            modalities: tuple[str, ...] = ("IR", "RGB"), **_) -> list:
    src = Path(src)
    labels = [p for m in modalities for p in sorted(src.rglob(f"{m}_label.json"))]
    if not labels:
        raise SchemaError(f"no IR_label.json / RGB_label.json under {src}")
    writers: dict[str, Writer] = {}
    for lab in labels:
        mod = lab.name.split("_")[0].upper()
        wr = writers.setdefault(mod, Writer(out_root, ds, modality="ir" if mod == "IR" else "rgb",
                                            keep_empty=keep_empty))
        ann = parse(lab)
        seq_dir = lab.parent
        seq = f"{seq_dir.relative_to(src).as_posix() or seq_dir.name}_{mod}"
        vid = next((p for p in seq_dir.iterdir() if p.stem.upper() == mod and p.suffix.lower() in VID_EXT), None)
        if vid is not None:
            for fr, img in iter_video(vid, frame_step):
                if fr < len(ann):
                    wr.add(img, ann[fr], seq, fr)
            continue
        fdir = next((d for d in (seq_dir / mod, seq_dir) if d.is_dir()), seq_dir)
        frames = sorted(p for p in fdir.iterdir() if p.suffix.lower() in IMG_EXT)
        if not frames:
            raise SchemaError(f"{seq_dir}: no {mod} video or frames")
        if len(frames) != len(ann):
            raise SchemaError(f"{seq_dir}: {len(frames)} frames but {len(ann)} labels")
        for i, fp in enumerate(frames):
            if i % max(1, frame_step) == 0:
                wr.add(read_image(fp), ann[i], seq, i, name=fp.stem)
    return [r for wr in writers.values() for r in wr.rows]
