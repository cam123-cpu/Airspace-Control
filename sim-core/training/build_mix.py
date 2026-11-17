"""Build a YOLO training set that mixes converted real data with dronesim synthetic data.

    python -m training.build_mix --converted /data/conv --synthetic data/ --out /data/mix_rgb \
        --modality rgb --synthetic-ratio 0.3 --tile 1280 --augment-frac 0.25
    python -m training.build_mix ... --commercial-only           # only commercial_ok: yes sources

Guarantees
  * test and val contain REAL frames only; synthetic data is train-only
  * splits are made over whole sequences (video / flight / folder), never over frames,
    so near-identical neighbouring frames cannot leak between train and test
  * near-duplicate frames (same picture and same boxes) are dropped within a sequence
  * every row keeps its source + licence; mix_report.md lists them with attribution notes
  * unlabelled stock frames are used as negatives only when their search query was a
    pure-background query; anything that may contain drones/birds goes to to_label.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .common import CLASSES, ManifestRow, load_catalog, read_manifest, safe_join, safe_name
from .imgops import augment, dhash, hamming, same_labels, tile_image

BACKGROUND_QUERIES = {"night sky", "stadium night", "city skyline night", "airport night", "prison fence",
                      "power substation"}
SPLITS = ("train", "val", "test")


def _u(key: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:12], 16) / float(1 << 48)


def synthetic_rows(syn_dir: Path) -> list[tuple[Path, ManifestRow]]:
    """Wrap generate_dataset.py output (images/, labels/) as manifest rows."""
    syn_dir = Path(syn_dir)
    rows = []
    for ip in sorted((syn_dir / "images").glob("*.jpg")):
        lp = syn_dir / "labels" / (ip.stem + ".txt")
        rows.append((syn_dir, ManifestRow(image=str(ip.relative_to(syn_dir)),
                                          label=str(lp.relative_to(syn_dir)) if lp.exists() else None,
                                          source="synthetic", license="owned", commercial_ok="yes",
                                          sequence=f"synthetic/{int(ip.stem) // 50 if ip.stem.isdigit() else ip.stem}",
                                          modality="rgb", context="synthetic")))
    return rows


def assign_split(seq: str, source_seqs: list[str], test_frac: float, val_frac: float, seed: int) -> str:
    low = seq.lower()
    for tok, sp in (("test", "test"), ("valid", "val"), ("/val", "val"), ("train", "train")):
        if tok in low.split("/", 1)[-1]:
            return sp
    if len(source_seqs) < 3:
        return "train"  # too few sequences to hold any out without leaking; reported in mix_report
    order = sorted(source_seqs, key=lambda s: _u(s, seed))
    n_test = max(1, round(test_frac * len(order)))
    n_val = max(1, round(val_frac * len(order)))
    i = order.index(seq)
    return "test" if i < n_test else "val" if i < n_test + n_val else "train"


def modality_ok(row: ManifestRow, modality: str) -> bool:
    if modality == "all":
        return True
    if modality == "ir":
        return row.modality == "ir"
    return row.modality in ("rgb", "gray")


def build(converted: list[Path], synthetic: Path | None, out: Path, modality: str = "rgb",
          commercial_only: bool = False, synthetic_ratio: float = 0.3, test_frac: float = 0.15,
          val_frac: float = 0.1, max_per_sequence: int = 300, dedupe_hamming: int = 4, tile: int = 0,
          tile_overlap: float = 0.2, empty_tile_frac: float = 0.1, augment_frac: float = 0.0,
          link: str = "copy", seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cat = load_catalog()
    lic = {d["id"]: d for d in cat["datasets"]}
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new output folder")
    rows: list[tuple[Path, ManifestRow]] = []
    for root in converted:
        mf = Path(root) / "manifest.jsonl"
        if not mf.is_file():
            raise SystemExit(f"missing {mf}")
        rows += [(Path(root), r) for r in read_manifest(mf)]
    real = []
    to_label = []
    skipped = Counter()
    for root, r in rows:
        if r.source not in lic:
            skipped["unknown_source"] += 1
            continue
        if commercial_only and str(lic[r.source]["commercial_ok"]) != "yes":
            skipped[f"licence:{r.source}"] += 1
            continue
        if not modality_ok(r, modality):
            skipped["modality"] += 1
            continue
        if r.label is None:
            q = str(r.extra.get("query", "")).lower()
            if q in BACKGROUND_QUERIES:
                real.append((root, r))
            else:
                to_label.append(r)
            continue
        real.append((root, r))

    # ---- group by sequence, dedupe, cap
    by_seq: dict[str, list[tuple[Path, ManifestRow]]] = defaultdict(list)
    for root, r in real:
        by_seq[r.sequence].append((root, r))
    kept: list[tuple[Path, ManifestRow, list[str]]] = []
    dup = 0
    for seq, items in by_seq.items():
        items.sort(key=lambda x: (x[1].frame, x[1].image))
        last_h, last_l = None, None
        seq_kept = []
        for root, r in items:
            lab = (safe_join(root, r.label).read_text(encoding="utf-8").splitlines() if r.label else [])
            img = cv2.imread(str(safe_join(root, r.image)), cv2.IMREAD_GRAYSCALE)
            if img is None:
                skipped["unreadable"] += 1
                continue
            h = dhash(img)
            if last_h is not None and hamming(h, last_h) <= dedupe_hamming and same_labels(lab, last_l):
                dup += 1
                continue
            last_h, last_l = h, lab
            seq_kept.append((root, r, lab))
        if len(seq_kept) > max_per_sequence:  # evenly thin long videos
            idx = np.linspace(0, len(seq_kept) - 1, max_per_sequence).round().astype(int)
            seq_kept = [seq_kept[i] for i in sorted(set(idx))]
        kept += seq_kept

    # ---- split by sequence within each source
    src_seqs: dict[str, list[str]] = defaultdict(list)
    for seq in {r.sequence for _, r, _ in kept}:
        src = next(r.source for _, r, _ in kept if r.sequence == seq)
        src_seqs[src].append(seq)
    seq_split = {s: assign_split(s, src_seqs[src], test_frac, val_frac, seed)
                 for src, seqs in src_seqs.items() for s in seqs}
    single_seq_sources = [s for s, v in src_seqs.items() if len(v) < 3]

    # ---- synthetic, train only, sized to the requested ratio
    syn = synthetic_rows(synthetic) if synthetic else []
    if syn and modality == "ir":
        syn = []  # current synthetic renderer is visible-band only in the YOLO export
    n_real_train = sum(1 for _, r, _ in kept if seq_split[r.sequence] == "train")
    if synthetic_ratio <= 0:
        syn = []
    if syn and 0 < synthetic_ratio < 1:
        want = int(round(synthetic_ratio / (1 - synthetic_ratio) * max(n_real_train, 1)))
        if want < len(syn):
            syn = [syn[i] for i in sorted(rng.choice(len(syn), size=want, replace=False))]
    for root, r in syn:
        lab = safe_join(root, r.label).read_text(encoding="utf-8").splitlines() if r.label else []
        kept.append((root, r, lab))
        seq_split[r.sequence] = "train"

    # ---- write
    for sp in SPLITS:
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    stats = {sp: Counter() for sp in SPLITS}
    cls_count = {sp: Counter() for sp in SPLITS}
    split_rows = []

    def put(img_src: Path, img: np.ndarray | None, lab: list[str], sp: str, stem: str, r: ManifestRow, extra=None):
        ip = out / "images" / sp / (stem + ".jpg")
        lp = out / "labels" / sp / (stem + ".txt")
        if img is None and link != "copy" and img_src.suffix.lower() == ".jpg":
            (os.link if link == "hardlink" else os.symlink)(img_src.resolve(), ip)
        elif img is None:
            shutil.copyfile(img_src, ip) if img_src.suffix.lower() == ".jpg" else \
                cv2.imwrite(str(ip), cv2.imread(str(img_src), cv2.IMREAD_COLOR))
        else:
            cv2.imwrite(str(ip), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lp.write_text("\n".join(lab) + ("\n" if lab else ""), encoding="utf-8")
        stats[sp][r.source] += 1
        for l in lab:
            cls_count[sp][CLASSES[int(l.split()[0])]] += 1
        d = asdict(r)
        d.update(split=sp, out_image=str(ip.relative_to(out)), **(extra or {}))
        split_rows.append(d)

    for root, r, lab in kept:
        sp = seq_split[r.sequence]
        src_path = safe_join(root, r.image)
        stem0 = safe_name(f"{r.source}__{r.sequence.replace('/', '_')}__{Path(r.image).stem}", 200)
        need_tile = tile > 0
        img = None
        if need_tile:
            img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
            if img is not None and max(img.shape[:2]) > tile * 1.25:
                for k, (t_img, t_lab, (tx, ty)) in enumerate(tile_image(img, lab, tile, tile_overlap,
                                                                        keep_empty=empty_tile_frac, rng=rng)):
                    put(src_path, t_img, t_lab, sp, f"{stem0}__t{k}", r, {"tile": [tx, ty]})
                continue
        put(src_path, None if img is None else img, lab, sp, stem0, r)
        if sp == "train" and r.source != "synthetic" and augment_frac > 0 and rng.random() < augment_frac:
            base = img if img is not None else cv2.imread(str(src_path), cv2.IMREAD_COLOR)
            ops = ["ir", "noise", "motion_blur", "jpeg"] if modality == "ir" else None
            a_img, chosen = augment(base, rng, ops=ops)
            put(src_path, a_img, lab, sp, f"{stem0}__aug", r, {"augment": chosen})

    names = {i: c for i, c in enumerate(CLASSES)}
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in names.items()), encoding="utf-8")
    with open(out / "splits.jsonl", "w", encoding="utf-8") as f:
        for d in split_rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    if to_label:
        (out / "to_label.txt").write_text("\n".join(f"{r.source}\t{r.image}\t{r.extra.get('query', '')}"
                                                    for r in to_label) + "\n", encoding="utf-8")
    report = write_report(out, stats, cls_count, lic, skipped, dup, single_seq_sources, commercial_only,
                          modality, split_rows, len(to_label))
    return report


def write_report(out, stats, cls_count, lic, skipped, dup, single, commercial_only, modality, split_rows, n_to_label):
    sources = sorted({s for sp in stats.values() for s in sp})
    lines = [f"# Training mix ({modality})", "",
             f"Commercial-only filter: **{'on' if commercial_only else 'off'}**. "
             + ("Every source below is licensed for commercial use." if commercial_only else
                "Sources marked no / unclear / permission_required must NOT ship in a commercial model; "
                "rebuild with --commercial-only for that."), "",
             "| Split | Images | " + " | ".join(CLASSES) + " |", "|---|---|" + "---|" * len(CLASSES)]
    for sp in SPLITS:
        lines.append(f"| {sp} | {sum(stats[sp].values())} | " + " | ".join(str(cls_count[sp][c]) for c in CLASSES) + " |")
    lines += ["", "Test and val are real frames only; synthetic data appears in train only.", "",
              "| Source | train | val | test | commercial_ok | licence |", "|---|---|---|---|---|---|"]
    for s in sources:
        d = lic.get(s, {"commercial_ok": "yes", "license": "owned"})
        lines.append(f"| {s} | {stats['train'][s]} | {stats['val'][s]} | {stats['test'][s]} | "
                     f"{d['commercial_ok']} | {str(d['license'])[:90]} |")
    attrib = sorted({r.get("extra", {}).get("attribution") for r in split_rows if r.get("extra", {}).get("attribution")})
    lines += ["", f"Near-duplicate frames dropped: {dup}.", f"Rows skipped: {dict(skipped) or 'none'}."]
    if single:
        lines.append(f"Sources with fewer than 3 sequences went to train only (cannot be held out without leakage): "
                     f"{', '.join(sorted(single))}.")
    if n_to_label:
        lines.append(f"{n_to_label} unlabelled stock frames may contain drones/birds: listed in to_label.txt, not used.")
    if attrib:
        lines += ["", "## Attribution", ""] + [f"- {a}" for a in attrib]
    if "usc_drone" in sources:
        lines += ["", "USC Drone Dataset: keep the USC copyright notice with any redistributed copy (see datasets.yaml)."]
    (out / "mix_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"splits": {sp: sum(stats[sp].values()) for sp in SPLITS}, "dup": dup, "skipped": dict(skipped),
            "single_sequence_sources": single}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--converted", type=Path, action="append", default=[], help="folder with manifest.jsonl (repeatable)")
    ap.add_argument("--synthetic", type=Path, help="generate_dataset.py output folder")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--modality", choices=["rgb", "ir", "all"], default="rgb")
    ap.add_argument("--commercial-only", action="store_true")
    ap.add_argument("--synthetic-ratio", type=float, default=0.3, help="share of TRAIN images that are synthetic")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--max-per-sequence", type=int, default=300)
    ap.add_argument("--dedupe-hamming", type=int, default=4)
    ap.add_argument("--tile", type=int, default=0, help="tile frames larger than this (e.g. 1280); 0 = off")
    ap.add_argument("--tile-overlap", type=float, default=0.2)
    ap.add_argument("--empty-tile-frac", type=float, default=0.1)
    ap.add_argument("--augment-frac", type=float, default=0.0, help="extra night/IR/blur/noise/glare copies of real train frames")
    ap.add_argument("--link", choices=["copy", "symlink", "hardlink"], default="copy")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if not a.converted and not a.synthetic:
        ap.error("give at least one --converted folder or --synthetic")
    if not 0 <= a.synthetic_ratio < 1:
        ap.error("--synthetic-ratio must be in [0, 1)")
    rep = build(a.converted, a.synthetic, a.out, a.modality, a.commercial_only, a.synthetic_ratio, a.test_frac,
                a.val_frac, a.max_per_sequence, a.dedupe_hamming, a.tile, a.tile_overlap, a.empty_tile_frac,
                a.augment_frac, a.link, a.seed)
    print(json.dumps(rep, indent=2))
    print(f"wrote {a.out / 'data.yaml'} and {a.out / 'mix_report.md'}")


if __name__ == "__main__":
    main()
