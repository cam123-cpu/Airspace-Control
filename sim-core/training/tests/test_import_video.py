"""import_video: what enters training, what waits for a human, and how untrusted clips are handled."""
from __future__ import annotations

import csv
import json
import os

import numpy as np
import pytest

from training import import_video as iv
from training.common import read_manifest

from .conftest import draw_target, sky, write_video


class BrightBlobDetector:
    """Stand-in for the ONNX model: one 'drone' box around pixels brighter than 200."""

    def __init__(self, conf=0.9):
        self.conf = conf

    def detect(self, img):
        ys, xs = np.where(img[:, :, 0] > 200)
        if len(xs) == 0:
            return []
        return [(0, self.conf, (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)))]


def make_clip(path, seconds=6, fps=10, target_from_s=2.0, seed=0):
    frames = []
    for i in range(seconds * fps):
        f = sky(seed=seed * 1000 + i)
        if i / fps >= target_from_s:
            x = 20 + i
            draw_target(f, x, 50, x + 8, 56)
        frames.append(f)
    write_video(path, frames)


def write_notes(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "start_s", "end_s", "label"])
        w.writerows(rows)


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "in" / "Flight 03 (north fence).mp4"
    p.parent.mkdir()
    make_clip(p)
    notes = tmp_path / "notes.csv"
    write_notes(notes, [[p.name, 0, 2, "empty"], [p.name, 2, 6, "drone"]])
    return p, notes


def rows_of(out):
    return list(read_manifest(out / "manifest.jsonl"))


def queue_of(out):
    with open(out / "review_queue.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def test_negatives_trusted_prelabels_flagged(tmp_path, clip):
    p, notes = clip
    out = tmp_path / "ds"
    rep = iv.run([p.parent], out, notes, fps=2, detector=BrightBlobDetector())
    rows = rows_of(out)
    neg = [r for r in rows if r.extra["n_boxes"] == 0]
    pos = [r for r in rows if r.extra["n_boxes"] > 0]
    assert len(neg) == 4 and all(not r.extra["needs_review"] and r.extra["told"] == "empty" for r in neg)
    assert len(pos) == 8 and all(r.extra["needs_review"] for r in pos)
    lab = (out / pos[0].label).read_text().split()
    assert lab[0] == "0" and all(0 < float(v) < 1 for v in lab[1:])
    assert rep["review_queue"] == 8 and {q["reason"] for q in queue_of(out)} == {"check pre-label"}


def test_drone_frames_without_boxes_never_become_negatives(tmp_path, clip):
    p, notes = clip
    out = tmp_path / "ds"
    iv.run([p], out, notes, fps=2, detector=None)
    rows = rows_of(out)
    assert all(r.extra["told"] == "empty" for r in rows) and len(rows) == 4
    q = queue_of(out)
    assert len(q) == 8 and all(x["reason"] == "drone segment, no box" for x in q)
    assert all((out / x["image"]).is_file() and x["image"].startswith("review/") for x in q)


def test_box_inside_negative_segment_goes_first_in_queue(tmp_path, clip):
    p, _ = clip
    notes = tmp_path / "wrong_notes.csv"
    write_notes(notes, [[p.name, 0, 6, "bird"]])         # operator says birds only, model sees a drone
    out = tmp_path / "ds"
    iv.run([p], out, notes, fps=2, detector=BrightBlobDetector(conf=0.55))
    q = queue_of(out)
    assert q[0]["reason"] == "contradicts notes"
    assert all(r.extra["needs_review"] for r in rows_of(out) if r.extra["n_boxes"])


def test_build_mix_skips_unreviewed_until_accepted(tmp_path, clip):
    from training.build_mix import build
    p, notes = clip
    out = tmp_path / "ds"
    iv.run([p], out, notes, fps=2, detector=BrightBlobDetector())
    rep = build([out], None, tmp_path / "mix1")
    assert rep["skipped"].get("unreviewed_prelabel") == 8
    assert iv.accept(out) == 8
    assert all(not r.extra["needs_review"] and r.extra.get("reviewed") for r in rows_of(out) if r.extra["n_boxes"])
    rep2 = build([out], None, tmp_path / "mix2")
    assert "unreviewed_prelabel" not in rep2["skipped"]


def test_names_come_from_content_not_filenames(tmp_path, clip):
    p, notes = clip
    out = tmp_path / "ds"
    iv.run([p], out, notes, fps=2, detector=BrightBlobDetector())
    for r in rows_of(out):
        assert "Flight" not in r.image and "north" not in r.image and r.sequence.startswith("own_footage/clip-")
        data = (out / r.image).read_bytes()
        assert data[:2] == b"\xff\xd8" and b"Exif" not in data[:4096]


def test_same_clip_twice_is_imported_once(tmp_path, clip):
    p, notes = clip
    copy = p.parent / "copy.mp4"
    copy.write_bytes(p.read_bytes())
    rep = iv.run([p.parent], tmp_path / "ds", notes, fps=2)
    skipped = [c for c in rep["clips"] if c.get("skipped")]
    assert len(skipped) == 1 and len({c["sha256"] for c in rep["clips"]}) == 1


def test_static_frames_are_deduplicated(tmp_path):
    p = tmp_path / "still.mp4"
    f = sky(seed=1)
    write_video(p, [f.copy() for _ in range(40)])
    rep = iv.run([p], tmp_path / "ds", None, fps=5)
    assert rep["clips"][0]["sampled"] == 1 and rep["clips"][0]["duplicates"] >= 5


def test_untrusted_inputs(tmp_path, clip):
    p, _ = clip
    (p.parent / "notes.txt").write_text("not a video")
    (p.parent / "evil.mp4").write_bytes(b"\x00" * 1024)          # extension says video, content doesn't
    if hasattr(os, "symlink"):
        os.symlink(p, p.parent / "link.mp4")
    found = iv.find_clips([p.parent])
    assert {c.name for c in found} == {p.name, "evil.mp4"}
    with pytest.raises(ValueError, match="not a readable video"):
        iv.run([p.parent / "evil.mp4"], tmp_path / "ds", None)
    bad = tmp_path / "bad.csv"
    write_notes(bad, [[p.name, 5, 2, "drone"]])
    with pytest.raises(ValueError, match="bad segment"):
        iv.read_notes(bad)
    write_notes(bad, [[p.name, 0, 2, "helicopter"]])
    with pytest.raises(ValueError, match="bad segment"):
        iv.read_notes(bad)
    with pytest.raises(ValueError, match="fps"):
        iv.run([p], tmp_path / "ds2", None, fps=500)


def test_cli_and_report(tmp_path, clip):
    p, notes = clip
    out = tmp_path / "ds"
    assert iv.main([str(p.parent), "--out", str(out), "--notes", str(notes), "--fps", "2"]) == 0
    rep = json.loads((out / "import_report.json").read_text())
    assert rep["rows"] == 4 and rep["unnoted_clips"] == []
    assert iv.main(["--accept", str(out)]) == 0
    assert iv.main([str(tmp_path / "nothing-here"), "--out", str(out)]) == 2
