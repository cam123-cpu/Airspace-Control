"""Generate a labelled synthetic PTZ dataset.

  python generate_dataset.py --n 2000 --out data/ --mode mixed --seed 0

Writes:
  data/images/000000.jpg ...         640x512 frames
  data/labels/000000.txt ...         YOLO: class cx cy w h (normalised)
  data/coco.json                     COCO detection annotations
  data/meta.csv                      per-object range, FOV, pixels-on-target, mode, sub-type
  data/data.yaml                     dataset file for Ultralytics-style trainers
  data/contact_sheet.png             24 samples with boxes and a zoomed inset of the main target
"""
import argparse, csv, io, json, os, sys
import numpy as np
import cv2
from render import render_frame, CLASSES, W, H
from security.safe_io import (bounded_int, bounded_float, resolve_output_dir, safe_join,
                              atomic_write_bytes, atomic_write_text, atomic_write_json, yaml_scalar,
                              UnsafePathError)

MAX_IMAGES = 200_000  # ~1 KB-100 KB each; caps disk use at roughly 20 GB


def _imencode(ext, img, params=()):
    ok, buf = cv2.imencode(ext, img, list(params))
    if not ok:
        raise RuntimeError(f"OpenCV failed to encode {ext}")
    return buf.tobytes()

COLORS = {"drone": (60, 200, 255), "drone_payload": (40, 80, 255), "bird": (90, 230, 90), "other": (230, 180, 60)}


def contact_sheet(samples, path, cols=6, tw=320, th=256):
    rows = (len(samples) + cols - 1) // cols
    sheet = np.full((rows * (th + 4) + 4, cols * (tw + 4) + 4, 3), 30, np.uint8)
    for k, (img, objs, meta) in enumerate(samples):
        t = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        sx, sy = tw / W, th / H
        for o in objs:
            x0, y0, x1, y1 = o["box"]
            cv2.rectangle(t, (int(x0 * sx) - 2, int(y0 * sy) - 2), (int(x1 * sx) + 2, int(y1 * sy) + 2), COLORS[o["cls"]], 1)
        if objs:  # zoomed inset of first object
            x0, y0, x1, y1 = objs[0]["box"]
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            half = max(12, int(max(x1 - x0, y1 - y0) * 0.8))
            crop = img[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
            if crop.size:
                crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_NEAREST)
                t[4:100, tw - 100:tw - 4] = crop
                cv2.rectangle(t, (tw - 101, 3), (tw - 3, 101), COLORS[objs[0]["cls"]], 1)
        label = f"{meta['mode']} | " + (", ".join(f"{o['cls']}:{o['px']:.0f}px" for o in objs[:2]) or "empty")
        cv2.putText(t, label, (5, th - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        r, c = divmod(k, cols)
        sheet[4 + r * (th + 4):4 + r * (th + 4) + th, 4 + c * (tw + 4):4 + c * (tw + 4) + tw] = t
    atomic_write_bytes(path, _imencode(".png", sheet))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Generate a labelled synthetic PTZ dataset.")
    ap.add_argument("--n", type=bounded_int(1, MAX_IMAGES, "--n"), default=2000)
    ap.add_argument("--out", default="data")
    ap.add_argument("--mode", choices=["mixed", "night", "thermal"], default="mixed")
    ap.add_argument("--seed", type=bounded_int(0, 2**63 - 1, "--seed"), default=0)
    ap.add_argument("--val-frac", type=bounded_float(0.0, 0.5, "--val-frac"), default=0.1,
                    help="fraction of images held out for validation (written to val.txt)")
    ap.add_argument("--overwrite", action="store_true", help="allow writing into a non-empty output directory")
    ap.add_argument("--allow-outside", action="store_true", help="allow an output directory outside the working directory")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    try:
        out = resolve_output_dir(a.out, allow_outside=a.allow_outside)
        if os.listdir(out) and not a.overwrite:
            raise UnsafePathError(f"{out} is not empty; pass --overwrite to replace its contents")
        img_dir = resolve_output_dir(safe_join(out, "images"), base=out)
        lbl_dir = resolve_output_dir(safe_join(out, "labels"), base=out)
    except UnsafePathError as e:
        sys.exit(f"error: {e}")
    rng = np.random.default_rng(a.seed)
    coco = {"images": [], "annotations": [], "categories": [{"id": i, "name": c} for i, c in enumerate(CLASSES)]}
    meta_rows, samples, ann_id = [], [], 0
    counts = {c: 0 for c in CLASSES}
    for i in range(a.n):
        img, objs, meta = render_frame(rng, None if a.mode == "mixed" else a.mode)
        name = f"{i:06d}.jpg"
        atomic_write_bytes(safe_join(img_dir, name), _imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95]), fsync=False)
        coco["images"].append({"id": i, "file_name": f"images/{name}", "width": W, "height": H, "mode": meta["mode"]})
        lines = []
        for o in objs:
            x0, y0, x1, y1 = o["box"]
            ci = CLASSES.index(o["cls"])
            lines.append(f"{ci} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n")
        atomic_write_text(safe_join(lbl_dir, f"{i:06d}.txt"), "".join(lines), fsync=False)
        for o in objs:
                x0, y0, x1, y1 = o["box"]
                ci = CLASSES.index(o["cls"])
                coco["annotations"].append({"id": ann_id, "image_id": i, "category_id": ci,
                                            "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0), "iscrowd": 0})
                ann_id += 1
                counts[o["cls"]] += 1
                meta_rows.append([name, o["cls"], o["sub"], meta["mode"], round(o["px"], 1),
                                  o["range_m"] and round(o["range_m"], 1), o["fov_deg"] and round(o["fov_deg"], 2),
                                  x1 - x0, y1 - y0, o["contrast"], meta["jpeg_q"]])
        if len(samples) < 24:
            samples.append((img, objs, meta))
    atomic_write_json(safe_join(out, "coco.json"), coco)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["image", "class", "subtype", "mode", "px_on_target", "range_m", "fov_deg", "box_w", "box_h", "contrast", "jpeg_q"])
    w.writerows(meta_rows)
    atomic_write_text(safe_join(out, "meta.csv"), buf.getvalue())
    # Train/val split (audit F-09): the old data.yaml used the same folder for train and val,
    # so validation scores measured memorisation. Split is seeded and reproducible.
    order = np.random.default_rng(a.seed + 1).permutation(a.n)
    n_val = int(round(a.n * a.val_frac))
    val_ids, train_ids = sorted(order[:n_val].tolist()), sorted(order[n_val:].tolist())
    atomic_write_text(safe_join(out, "train.txt"), "".join(f"./images/{i:06d}.jpg\n" for i in train_ids))
    atomic_write_text(safe_join(out, "val.txt"), "".join(f"./images/{i:06d}.jpg\n" for i in (val_ids or train_ids)))
    # Every scalar is quoted, so a path containing ':', '#' or a newline cannot inject YAML keys (audit F-05).
    atomic_write_text(safe_join(out, "data.yaml"),
                      f"path: {yaml_scalar(out)}\ntrain: {yaml_scalar('train.txt')}\nval: {yaml_scalar('val.txt')}\nnames:\n" +
                      "".join(f"  {i}: {yaml_scalar(c)}\n" for i, c in enumerate(CLASSES)))
    contact_sheet(samples, safe_join(out, "contact_sheet.png"))
    print(f"wrote {a.n} images ({len(train_ids)} train / {n_val} val), {ann_id} boxes to {out}: {counts}")


if __name__ == "__main__":
    main()
