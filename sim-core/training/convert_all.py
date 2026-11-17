"""Convert every raw dataset folder that is present, using the catalog's converter and
per-dataset defaults. Missing datasets are skipped with a note.

    python -m training.convert_all --raw /data/raw --out /data/converted [--commercial-only]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .common import load_catalog, write_manifest
from .convert import get
from .convert.base import SchemaError

# Per-dataset converter options. Frame steps keep near-duplicate video frames down before
# build_mix de-duplicates further.
DEFAULT_OPTS = {
    "aot": {"frame_step": 10},
    "usc_drone": {"fmt": "cxcywh", "frame_col": True, "gt_name": "*.txt"},
    "dut_anti_uav": {},
    "anti_uav": {"frame_step": 5},
    "drone_vs_bird": {"frame_step": 5},
    "halmstad_multisensor": {"frame_step": 3},
    "kios_uav": {},
    "det_fly": {},
    "mav_vid": {"frame_step": 3},
    "cst_anti_uav": {"frame_step": 5},
    "rgbt_tiny": {"frame_step": 3},
    "mmaud": {},
    "fbd_sv_2024": {"frame_step": 2},
    "aod4": {},
    "amateur_drone_mendeley": {},
    "own_footage": {},
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--commercial-only", action="store_true")
    ap.add_argument("--only", action="append", default=[])
    a = ap.parse_args(argv)
    cat = load_catalog()
    rc = 0
    for ds in cat["datasets"]:
        i, conv = ds["id"], ds.get("converter")
        if not conv or i == "synthetic" or (a.only and i not in a.only):
            continue
        if a.commercial_only and str(ds["commercial_ok"]) != "yes":
            continue
        src = a.raw / i
        if not src.is_dir():
            print(f"skip {i}: {src} not present")
            continue
        out = a.out / i
        try:
            rows = get(conv).convert(src, out, ds, **DEFAULT_OPTS.get(i, {}))
        except SchemaError as e:
            print(f"error {i}: {e}\n  -> inspect the files and pass the right converter options "
                  f"(python -m training.convert {i} {src} {out} --opt key=value)", file=sys.stderr)
            rc = 1
            continue
        write_manifest(rows, out / "manifest.jsonl")
        print(f"{i}: {len(rows)} frames -> {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
