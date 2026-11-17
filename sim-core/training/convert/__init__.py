"""Converters from each catalogued annotation format to YOLO. See base.py for the interface.

    python -m training.convert <dataset_id> <src_dir> <out_root> [--opt key=value ...]
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

CONVERTERS = {
    "voc_xml": "voc_xml", "coco_json": "coco_json", "yolo_passthrough": "yolo_passthrough",
    "wosdetc_txt": "wosdetc_txt", "rect_txt": "rect_txt", "anti_uav_json": "anti_uav_json",
    "aot_json": "aot_json", "halmstad_csv": "halmstad_csv",
}


def get(name: str):
    if name not in CONVERTERS:
        raise KeyError(f"unknown converter '{name}', choose from {sorted(CONVERTERS)}")
    return importlib.import_module(f".{CONVERTERS[name]}", __name__)


def run(ds_id: str, src: Path, out_root: Path, converter: str | None = None, append: bool = True, **opts):
    from ..common import load_catalog, dataset_by_id, write_manifest
    ds = dataset_by_id(load_catalog(), ds_id)
    conv = converter or ds.get("converter")
    if not conv:
        raise SystemExit(f"{ds_id} has no image converter (it is a {ds.get('modality')} dataset)")
    rows = get(conv).convert(Path(src), Path(out_root), ds, **opts)
    n = write_manifest(rows, Path(out_root) / "manifest.jsonl", append=append)
    return n


def _parse_opt(s: str):
    k, _, v = s.partition("=")
    try:
        return k, json.loads(v)
    except json.JSONDecodeError:
        return k, v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_id")
    ap.add_argument("src", type=Path)
    ap.add_argument("out_root", type=Path)
    ap.add_argument("--converter")
    ap.add_argument("--opt", action="append", default=[], help="converter option key=value (JSON values ok)")
    a = ap.parse_args(argv)
    opts = dict(_parse_opt(o) for o in a.opt)
    n = run(a.dataset_id, a.src, a.out_root, a.converter, **opts)
    print(f"{a.dataset_id}: wrote {n} manifest rows to {a.out_root / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
