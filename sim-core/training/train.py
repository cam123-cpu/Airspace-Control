"""Train, evaluate on the real-only test split, and export to ONNX.

    python -m training.train --config training/configs/rgb.yaml
    python -m training.train --config training/configs/ir.yaml --set epochs=80 --set model=yolo11m.pt
    python -m training.train --config training/configs/rgb.yaml --dry-run    # print the plan only

Outputs under runs/<name>/: weights/best.pt, Ultralytics plots, eval_test/metrics.md (our
size-binned real-only scoring), <name>.onnx, and model_card.md recording the data mix and
whether the model is commercial-safe (only if the mix was built with --commercial-only).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

ULTRALYTICS_VERSION = "8.3.172"   # pinned; verified to exist on PyPI (release 8.3.172)

TRAIN_KEYS = {"data", "model", "epochs", "patience", "imgsz", "batch", "optimizer", "cos_lr", "close_mosaic",
              "mosaic", "mixup", "copy_paste", "scale", "translate", "degrees", "flipud", "fliplr", "hsv_h",
              "hsv_s", "hsv_v", "workers", "seed", "cache", "lr0", "lrf", "weight_decay", "device", "fraction",
              "rect", "multi_scale", "amp", "freeze", "dropout", "single_cls"}
META_KEYS = {"name", "pretrained", "export", "eval_tile"}


def load_config(path: Path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise SystemExit(f"{path}: config must be a mapping")
    for o in overrides:
        k, _, v = o.partition("=")
        if not k or not _:
            raise SystemExit(f"--set expects key=value, got '{o}'")
        cfg[k] = yaml.safe_load(v)
    unknown = set(cfg) - TRAIN_KEYS - META_KEYS
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)} (allowed: {sorted(TRAIN_KEYS | META_KEYS)})")
    for k in ("name", "data", "model"):
        if k not in cfg:
            raise SystemExit(f"config missing '{k}'")
    return cfg


def plan(cfg: dict, project: Path) -> list[str]:
    name = cfg["name"]
    train_args = {k: v for k, v in cfg.items() if k in TRAIN_KEYS}
    mix = Path(cfg["data"]).parent
    return [
        f"ultralytics=={ULTRALYTICS_VERSION} YOLO({cfg['model']!r})"
        + (f".load({cfg['pretrained']!r})" if str(cfg['model']).endswith('.yaml') and cfg.get('pretrained') else ""),
        f"train({json.dumps(train_args)}, project={str(project)!r}, name={name!r})",
        f"val(split='test') on {cfg['data']}",
        f"evaluate_real --mix {mix} --weights {project / name / 'weights' / 'best.pt'} --tile {cfg.get('eval_tile', 0)}",
        f"export({json.dumps(cfg.get('export', {}))}) -> {project / name / (name + '.onnx')}",
        f"model_card.md <- {mix / 'mix_report.md'}",
    ]


def model_card(cfg: dict, run_dir: Path, metrics_md: Path | None, onnx: Path | None) -> Path:
    mix = Path(cfg["data"]).parent
    rep = (mix / "mix_report.md").read_text(encoding="utf-8") if (mix / "mix_report.md").is_file() else ""
    commercial = "Commercial-only filter: **on**" in rep
    L = [f"# Model card: {cfg['name']}", "",
         f"- Architecture: {cfg['model']} (Ultralytics {ULTRALYTICS_VERSION}), input {cfg.get('imgsz')} px",
         f"- ONNX: {onnx.name if onnx else 'not exported'}",
         f"- Commercial use of the weights: **{'allowed by the data licences used' if commercial else 'NOT cleared'}**"
         + ("" if commercial else " - the mix includes research-only or unclear-licence data; rebuild with "
                                  "build_mix.py --commercial-only and retrain before shipping."),
         "- Classes: drone, drone_payload, bird, other", "",
         "## Real-only test results", "", metrics_md.read_text(encoding="utf-8") if metrics_md and metrics_md.is_file()
         else "(not run)", "", "## Training data", "", rep or "(mix_report.md not found)", "",
         "## Known limits", "",
         "- drone_payload examples come almost entirely from synthetic data and your own flights; treat that class "
         "as unvalidated until you have real labelled payload footage.",
         "- Night visible-band performance depends on how much real night footage is in the mix; check the by-source table.",
         "- Scores are per frame. The fusion engine combines them over time with radar/RF/acoustic evidence."]
    p = run_dir / "model_card.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


def run(cfg: dict, project: Path) -> None:
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(f"pip install ultralytics=={ULTRALYTICS_VERSION} (see training/requirements.txt)")
    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        print(f"warning: ultralytics {ultralytics.__version__} installed, pipeline tested against {ULTRALYTICS_VERSION}",
              file=sys.stderr)
    name = cfg["name"]
    model = YOLO(cfg["model"])
    if str(cfg["model"]).endswith(".yaml") and cfg.get("pretrained"):
        model = model.load(cfg["pretrained"])  # transfer matching layers into the custom architecture
    train_args = {k: v for k, v in cfg.items() if k in TRAIN_KEYS}
    model.train(**train_args, project=str(project), name=name, exist_ok=True, plots=True)
    run_dir = project / name
    best = run_dir / "weights" / "best.pt"
    best_model = YOLO(str(best))
    best_model.val(data=cfg["data"], split="test", imgsz=cfg.get("imgsz", 1280), project=str(project),
                   name=f"{name}_test", exist_ok=True)
    from .evaluate_real import evaluate, load_split, predict_ultralytics, to_markdown
    mix = Path(cfg["data"]).parent
    items = load_split(mix, "test")
    metrics_md = None
    if items:
        res = evaluate(items, predict_ultralytics(best, items, cfg.get("imgsz", 1280), cfg.get("eval_tile", 0)))
        ev = run_dir / "eval_test"
        ev.mkdir(parents=True, exist_ok=True)
        (ev / "metrics.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
        metrics_md = ev / "metrics.md"
        metrics_md.write_text(to_markdown(res, f"{name}: real-only test"), encoding="utf-8")
        print(metrics_md.read_text(encoding="utf-8"))
    exp = dict(cfg.get("export") or {"format": "onnx"})
    exp.setdefault("imgsz", cfg.get("imgsz", 1280))
    onnx = Path(best_model.export(**exp))
    final = run_dir / f"{name}.onnx"
    if onnx.resolve() != final.resolve():
        shutil.copyfile(onnx, final)
    print(f"model card: {model_card(cfg, run_dir, metrics_md, final)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--set", action="append", default=[], help="override a config key: key=value")
    ap.add_argument("--project", type=Path, default=Path("runs"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    if a.dry_run:
        print("\n".join(plan(cfg, a.project)))
        return
    run(cfg, a.project)


if __name__ == "__main__":
    main()
