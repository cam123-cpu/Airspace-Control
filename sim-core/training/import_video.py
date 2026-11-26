"""Turn plain video clips into a YOLO dataset of your own footage, with pre-labels for human review.

    python3 -m training.import_video clips/ --out /data/raw/own_footage --notes notes.csv \\
        [--model models/rgb.onnx] [--fps 2] [--modality rgb|ir] [--context "night, floodlit"]
    python3 -m training.import_video --accept /data/raw/own_footage      # after fixing boxes in CVAT

What goes into training, and what doesn't:
  * A frame inside a notes segment labelled bird, empty or other is a negative (an empty label file).
    You said what is in the frame, so it is trusted. Negatives are what teach a detector to stop
    flagging gulls.
  * A frame the model pre-labelled gets those boxes, marked needs_review. build_mix.py leaves
    unreviewed rows out until you correct the boxes (CVAT or Label Studio, YOLO format) and run --accept.
  * A frame inside a drone segment where the model found nothing, or any drone frame with no model,
    is written to review_queue.csv only. It never enters training as empty sky.

notes.csv columns: clip,start_s,end_s,label   (label: drone | drone_payload | bird | other | empty)
    flight03.mp4,0,12.5,empty
    flight03.mp4,12.5,41,drone
Clips without notes are still imported, and all of their frames go to the review queue.

Clips are untrusted input: only video extensions are read, symlinks are skipped, size, resolution and
duration are capped, output names come from a hash of the clip's content (never from its file name),
and frames are re-encoded from decoded pixels, so container metadata such as GPS tags is not copied.
Frames are split by clip downstream (build_mix.py), never by frame, so one flight can't sit in both
train and validation. The tool cannot blur faces: don't import footage that shows people.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .common import CLASSES, ManifestRow, read_manifest, write_manifest
from .convert.base import VID_EXT, Box, Writer

MAX_BYTES = 8 * 1024 ** 3
MAX_SIDE = 7680
MAX_SECONDS = 30 * 60
NEGATIVE = {"bird", "other", "empty"}
POSITIVE = {"drone", "drone_payload"}
DS = {"id": "own_footage", "license": "owned", "commercial_ok": "yes", "modality": ["rgb"]}


@dataclass
class Segment:
    start: float
    end: float
    label: str


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def find_clips(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        cands = sorted(p.rglob("*")) if p.is_dir() else [p]
        for c in cands:
            if c.is_symlink() or not c.is_file() or c.suffix.lower() not in VID_EXT:
                continue
            out.append(c)
    return out


def read_notes(path: Path | None) -> dict[str, list[Segment]]:
    notes: dict[str, list[Segment]] = {}
    if path is None:
        return notes
    with path.open(newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            try:
                clip, label = row["clip"].strip(), row["label"].strip().lower()
                a, b = float(row["start_s"]), float(row["end_s"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"{path}:{i}: expected clip,start_s,end_s,label") from None
            if label not in NEGATIVE | POSITIVE or not (0 <= a < b) or not np.isfinite(b):
                raise ValueError(f"{path}:{i}: bad segment {row}")
            notes.setdefault(Path(clip).name, []).append(Segment(a, b, label))
    return notes


def label_at(segs: list[Segment], t: float) -> str | None:
    for s in segs:
        if s.start <= t < s.end:
            return s.label
    return None


def dhash(img: np.ndarray) -> int:
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def import_clip(clip: Path, writer: Writer, segs: list[Segment], fps: float, detector=None,
                conf: float = 0.25, dedupe: int = 3, digest: str | None = None) -> tuple[list[ManifestRow], list[dict], dict]:
    if clip.stat().st_size > MAX_BYTES:
        raise ValueError(f"{clip.name}: larger than {MAX_BYTES // 1024 ** 3} GB")
    digest = digest or sha256_file(clip)
    seq = "clip-" + digest[:16]
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise ValueError(f"{clip.name}: not a readable video")
    rows, queue = [], []
    stats = {"clip": clip.name, "sequence": seq, "sha256": digest, "sampled": 0, "negatives": 0,
             "prelabelled": 0, "queued": 0, "duplicates": 0}
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        src_fps = src_fps if np.isfinite(src_fps) and 1 <= src_fps <= 1000 else 30.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if max(w, h) > MAX_SIDE:
            raise ValueError(f"{clip.name}: {w}x{h} is above the {MAX_SIDE}px cap")
        step = max(1, int(round(src_fps / fps)))
        last_hash, i = None, -1
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            i += 1
            t = i / src_fps
            if t > MAX_SECONDS:
                stats["truncated_at_s"] = MAX_SECONDS
                break
            if i % step or fr is None or fr.ndim != 3:
                continue
            hsh = dhash(fr)
            if last_hash is not None and bin(hsh ^ last_hash).count("1") <= dedupe:
                stats["duplicates"] += 1
                continue
            last_hash = hsh
            stats["sampled"] += 1
            told = label_at(segs, t)
            boxes: list[Box] = []
            best = 0.0
            if detector is not None:
                for c, s, (x0, y0, x1, y1) in detector.detect(fr):
                    if s >= conf and 0 <= c < len(CLASSES):
                        boxes.append(Box(CLASSES[c], x0, y0, x1, y1))
                        best = max(best, s)
            item = {"clip": clip.name, "sequence": seq, "frame": i, "t_s": round(t, 3), "told": told or ""}
            if told in NEGATIVE and not boxes:
                rows.append(writer.add(fr, [], seq, i, extra={"t_s": round(t, 3), "told": told, "needs_review": False}))
                stats["negatives"] += 1
            elif boxes:
                # pre-labels always need a human; a box inside a "bird"/"empty" segment needs one most of all
                rows.append(writer.add(fr, boxes, seq, i, extra={"t_s": round(t, 3), "told": told or "",
                                                                 "needs_review": True, "max_conf": round(best, 3)}))
                stats["prelabelled"] += 1
                queue.append({**item, "reason": "contradicts notes" if told in NEGATIVE else "check pre-label",
                              "max_conf": round(best, 3), "image": rows[-1].image})
            else:
                img_rel = writer.root / "review" / seq / f"{i:07d}.jpg"
                img_rel.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(img_rel), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                stats["queued"] += 1
                queue.append({**item, "reason": "drone segment, no box" if told in POSITIVE else "no notes, no box",
                              "max_conf": 0.0, "image": str(img_rel.relative_to(writer.root))})
    finally:
        cap.release()
    return rows, queue, stats


def _uncertainty(q: dict) -> tuple:
    order = {"contradicts notes": 0, "drone segment, no box": 1, "check pre-label": 2, "no notes, no box": 3}
    return order.get(q["reason"], 9), abs(q["max_conf"] - 0.5)


def run(inputs: list[Path], out: Path, notes: Path | None = None, fps: float = 2.0, detector=None,
        conf: float = 0.25, modality: str = "rgb", context: str = "") -> dict:
    if not 0.05 <= fps <= 30:
        raise ValueError("--fps must be between 0.05 and 30")
    if modality not in ("rgb", "ir", "gray"):
        raise ValueError("--modality must be rgb, ir or gray")
    clips = find_clips(inputs)
    if not clips:
        raise ValueError("no video files found (mp4, mov, mkv, avi, m4v)")
    segs = read_notes(notes)
    writer = Writer(out, {**DS, "modality": [modality]}, modality=modality, context=context)
    all_rows, all_queue, per_clip, seen = [], [], [], set()
    for clip in clips:
        if clip.stat().st_size > MAX_BYTES:
            raise ValueError(f"{clip.name}: larger than {MAX_BYTES // 1024 ** 3} GB")
        digest = sha256_file(clip)
        if digest in seen:                    # the same clip under two names would sit in two splits
            per_clip.append({"clip": clip.name, "sha256": digest, "skipped": "duplicate clip content"})
            continue
        seen.add(digest)
        rows, queue, st = import_clip(clip, writer, segs.get(clip.name, []), fps, detector, conf, digest=digest)
        all_rows += rows
        all_queue += queue
        per_clip.append(st)
    write_manifest(all_rows, out / "manifest.jsonl", append=(out / "manifest.jsonl").exists())
    all_queue.sort(key=_uncertainty)
    with (out / "review_queue.csv").open("w", newline="", encoding="utf-8") as fh:
        cols = ["reason", "clip", "t_s", "frame", "told", "max_conf", "image", "sequence"]
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(all_queue)
    report = {"clips": per_clip, "rows": len(all_rows), "review_queue": len(all_queue),
              "unnoted_clips": sorted(c.name for c in clips if c.name not in segs)}
    (out / "import_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def accept(out: Path) -> int:
    """Mark pre-labelled rows reviewed after a human corrected the label files in place."""
    mf = out / "manifest.jsonl"
    rows = list(read_manifest(mf))
    n = 0
    for r in rows:
        if r.extra.get("needs_review"):
            r.extra["needs_review"] = False
            r.extra["reviewed"] = True
            n += 1
    write_manifest(rows, mf)
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="*", type=Path, help="video files or folders")
    ap.add_argument("--out", type=Path, help="dataset folder (created if missing)")
    ap.add_argument("--notes", type=Path, help="CSV: clip,start_s,end_s,label")
    ap.add_argument("--fps", type=float, default=2.0, help="frames sampled per second of video (default 2)")
    ap.add_argument("--model", type=Path, help="ONNX detector for pre-labels")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--modality", default="rgb")
    ap.add_argument("--context", default="", help='free text stored on every row, e.g. "night, floodlit"')
    ap.add_argument("--accept", type=Path, metavar="DATASET", help="mark corrected pre-labels as reviewed")
    a = ap.parse_args(argv)
    try:
        if a.accept:
            print(f"marked {accept(a.accept)} rows reviewed")
            return 0
        if not a.inputs or not a.out:
            ap.error("give video inputs and --out (or --accept DATASET)")
        det = None
        if a.model:
            from .onnx_detector import OnnxYoloDetector
            det = OnnxYoloDetector(a.model, imgsz=a.imgsz, conf=a.conf)
        rep = run(a.inputs, a.out, a.notes, a.fps, det, a.conf, a.modality, a.context)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"{len(rep['clips'])} clips -> {rep['rows']} training rows, {rep['review_queue']} frames to review "
          f"(see review_queue.csv)")
    if rep["unnoted_clips"]:
        print("no notes for: " + ", ".join(rep["unnoted_clips"]) + " (all of their frames are queued for review)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
