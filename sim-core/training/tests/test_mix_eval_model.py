import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from training import build_mix, evaluate_real, imgops, onnx_detector, temporal, train
from training.common import ManifestRow, write_manifest

from .conftest import draw_target, sky


def _make_source(root: Path, source: str, commercial: str, n_seq: int, n_frames: int, dup: bool = False, big=False):
    rows = []
    for s in range(n_seq):
        for f in range(n_frames):
            W, H = (2600, 1600) if big else (160, 120)
            img = sky(W, H, seed=0 if dup else s * 100 + f)
            x = 20 + (0 if dup else f * 3)
            draw_target(img, x, 30, x + 12, 38)
            ip = root / "images" / source / f"s{s}" / f"{f:04d}.jpg"
            lp = root / "labels" / source / f"s{s}" / f"{f:04d}.txt"
            ip.parent.mkdir(parents=True, exist_ok=True)
            lp.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ip), img)
            lp.write_text(f"0 {(x + 6) / W:.6f} {34 / H:.6f} {12 / W:.6f} {8 / H:.6f}\n")
            rows.append(ManifestRow(image=str(ip.relative_to(root)), label=str(lp.relative_to(root)), source=source,
                                    license="x", commercial_ok=commercial, sequence=f"{source}/s{s}", frame=f))
    write_manifest(rows, root / "manifest.jsonl", append=True)


def _synthetic(root: Path, n=10):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for i in range(n):
        cv2.imwrite(str(root / "images" / f"{i:06d}.jpg"), draw_target(sky(seed=500 + i), 10, 10, 20, 18))
        (root / "labels" / f"{i:06d}.txt").write_text("0 0.1 0.1 0.06 0.06\n")
    return root


def _splits(out):
    return [json.loads(l) for l in (out / "splits.jsonl").read_text().splitlines()]


def test_build_mix_splits_by_sequence_real_only_test(tmp_path):
    conv = tmp_path / "conv"
    _make_source(conv, "usc_drone", "yes", n_seq=10, n_frames=4)
    _make_source(conv, "drone_vs_bird", "no", n_seq=6, n_frames=3)
    syn = _synthetic(tmp_path / "syn", 20)
    out = tmp_path / "mix"
    rep = build_mix.build([conv], syn, out, synthetic_ratio=0.3, seed=1)
    rows = _splits(out)
    seq_to_split = {}
    for r in rows:
        assert seq_to_split.setdefault(r["sequence"], r["split"]) == r["split"], "sequence leaked across splits"
    assert all(r["source"] != "synthetic" for r in rows if r["split"] in ("val", "test"))
    assert rep["splits"]["test"] > 0 and rep["splits"]["val"] > 0
    n_real_train = sum(1 for r in rows if r["split"] == "train" and r["source"] != "synthetic")
    n_syn = sum(1 for r in rows if r["source"] == "synthetic")
    assert abs(n_syn / (n_syn + n_real_train) - 0.3) < 0.08
    assert "NOT ship" in (out / "mix_report.md").read_text()
    assert "test: images/test" in (out / "data.yaml").read_text()


def test_build_mix_commercial_only(tmp_path):
    conv = tmp_path / "conv"
    _make_source(conv, "usc_drone", "yes", 4, 2)
    _make_source(conv, "drone_vs_bird", "no", 4, 2)
    out = tmp_path / "mix"
    build_mix.build([conv], None, out, commercial_only=True)
    srcs = {r["source"] for r in _splits(out)}
    assert srcs == {"usc_drone"}
    assert "Every source below is licensed for commercial use" in (out / "mix_report.md").read_text()


def test_build_mix_dedupes_static_frames(tmp_path):
    conv = tmp_path / "conv"
    _make_source(conv, "usc_drone", "yes", 3, 6, dup=True)
    rep = build_mix.build([conv], None, tmp_path / "mix")
    assert rep["dup"] == 3 * 5


def test_build_mix_tiles_large_frames(tmp_path):
    conv = tmp_path / "conv"
    _make_source(conv, "aot", "yes", 3, 1, big=True)
    out = tmp_path / "mix"
    build_mix.build([conv], None, out, tile=1024, empty_tile_frac=0.0)
    labs = [p for p in (out / "labels").rglob("*.txt")]
    imgs = [cv2.imread(str(p)) for p in (out / "images").rglob("*.jpg")]
    assert all(max(i.shape[:2]) <= 1024 for i in imgs)
    assert labs and all(p.read_text().strip() for p in labs)  # only tiles containing the target


def test_build_mix_refuses_nonempty_out(tmp_path):
    (tmp_path / "mix").mkdir()
    (tmp_path / "mix" / "x").write_text("x")
    with pytest.raises(SystemExit):
        build_mix.build([], _synthetic(tmp_path / "s", 2), tmp_path / "mix")


def test_unlabelled_stock_frames_only_background_queries(tmp_path):
    conv = tmp_path / "conv"
    _make_source(conv, "usc_drone", "yes", 3, 2)
    rows = []
    for q in ("night sky", "drone flying"):
        ip = conv / "images" / "pixabay_video" / q.replace(" ", "_") / "00001.jpg"
        ip.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ip), sky())
        rows.append(ManifestRow(image=str(ip.relative_to(conv)), label=None, source="pixabay_video", license="p",
                                commercial_ok="unclear", sequence=f"pixabay_video/{q}", extra={"query": q}))
    write_manifest(rows, conv / "manifest.jsonl", append=True)
    out = tmp_path / "mix"
    build_mix.build([conv], None, out)
    used = [r for r in _splits(out) if r["source"] == "pixabay_video"]
    assert len(used) == 1 and used[0]["extra"]["query"] == "night sky"
    assert "drone flying" in (out / "to_label.txt").read_text()


def test_tile_and_augment():
    img = sky(3000, 2000)
    lab = ["0 0.5 0.5 0.01 0.01"]
    tiles = list(imgops.tile_image(img, lab, tile=1280, keep_empty=0.0))
    assert tiles and all(t[1] for t in tiles)
    for t_img, t_lab, (tx, ty) in tiles:
        c, cx, cy, w, h = t_lab[0].split()
        X = tx + float(cx) * t_img.shape[1]
        assert abs(X - 1500) < 2
    rng = np.random.default_rng(0)
    for name, fn in imgops.AUGS.items():
        out = fn(sky(), rng)
        assert out.shape == (120, 160, 3) and out.dtype == np.uint8, name


def test_evaluate_ap(tmp_path):
    items = [{"stem": "a", "W": 100, "H": 100, "gts": [(0, 0.1, 0.1, 0.2, 0.2)], "source": "s", "context": ""},
             {"stem": "b", "W": 100, "H": 100, "gts": [(2, 0.5, 0.5, 0.6, 0.6)], "source": "s", "context": ""}]
    perfect = {"a": [(0, 0.9, 0.1, 0.1, 0.2, 0.2)], "b": [(2, 0.8, 0.5, 0.5, 0.6, 0.6)]}
    r = evaluate_real.evaluate(items, perfect)
    assert r["any_drone"]["AP50"] == pytest.approx(1.0) and r["per_class"]["bird"]["AP50"] == pytest.approx(1.0)
    assert r["by_size_px"]["8-16"]["n_gt"] == 1 and r["false_drone_on_no_drone_frames"]["rate"] == 0
    bad = {"a": [(0, 0.9, 0.6, 0.6, 0.7, 0.7)], "b": [(0, 0.95, 0.5, 0.5, 0.6, 0.6)]}
    r = evaluate_real.evaluate(items, bad)
    assert r["any_drone"]["AP50"] == 0.0 and r["false_drone_on_no_drone_frames"]["bird_frame_rate"] == 1.0
    md = evaluate_real.to_markdown(r, "t")
    assert "any drone" in md


def test_evaluate_cli_with_predictions(tmp_path):
    conv = tmp_path / "conv"
    _make_source(conv, "usc_drone", "yes", 8, 2)
    mix = tmp_path / "mix"
    build_mix.build([conv], None, mix)
    preds = []
    for lp in (mix / "labels" / "test").glob("*.txt"):
        c, cx, cy, w, h = map(float, lp.read_text().split())
        preds.append({"stem": lp.stem, "cls": 0, "conf": 0.9, "xyxyn": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]})
    pf = tmp_path / "p.jsonl"
    pf.write_text("\n".join(json.dumps(p) for p in preds))
    evaluate_real.main(["--mix", str(mix), "--predictions", str(pf)])
    m = json.loads((mix / "eval_test" / "metrics.json").read_text())
    assert m["any_drone"]["AP50"] == pytest.approx(1.0)


class _FakeSess:
    """Mimics onnxruntime: returns one confident drone at the image centre (input coords)."""
    def __init__(self, size, cls=0, score=0.9):
        self.size, self.cls, self.score = size, cls, score

    def get_inputs(self):
        return [type("I", (), {"name": "images"})()]

    def run(self, _, feeds):
        x = feeds["images"]
        assert x.shape == (1, 3, self.size, self.size) and x.dtype == np.float32 and x.max() <= 1.0
        out = np.zeros((1, 8, 3), np.float32)
        out[0, :4, 0] = [self.size / 2, self.size / 2, 20, 10]
        out[0, 4 + self.cls, 0] = self.score
        out[0, :4, 1] = [self.size / 2 + 1, self.size / 2, 20, 10]   # duplicate -> removed by NMS
        out[0, 4 + self.cls, 1] = self.score - 0.1
        out[0, 4 + 2, 2] = 0.001                                      # below conf
        return [out]


def test_onnx_decode_and_detect():
    det = onnx_detector.OnnxYoloDetector("unused.onnx", imgsz=320, conf=0.25, session=_FakeSess(320))
    img = sky(640, 480)
    d = det.detect(img)
    assert len(d) == 1 and d[0][0] == 0
    x0, y0, x1, y1 = d[0][2]
    assert abs((x0 + x1) / 2 - 320) < 2 and abs((y0 + y1) / 2 - 240) < 2 and abs((x1 - x0) - 40) < 2
    assert det.p_drone(img) == pytest.approx(0.9)
    det_bird = onnx_detector.OnnxYoloDetector("u.onnx", imgsz=320, session=_FakeSess(320, cls=2))
    assert det_bird.p_drone(img) == onnx_detector.FLOOR


def test_onnx_tiled_detect_merges():
    det = onnx_detector.OnnxYoloDetector("u.onnx", imgsz=256, conf=0.25, tile=512, session=_FakeSess(256))
    d = det.detect(sky(1400, 900))
    assert len(d) >= 1


def test_onnx_missing_file():
    with pytest.raises(FileNotFoundError):
        onnx_detector.OnnxYoloDetector("does_not_exist.onnx")


def test_sim_classifier_adapter(monkeypatch):
    real_init = onnx_detector.OnnxYoloDetector.__init__

    def init(self, model_path, **kw):
        real_init(self, model_path, session=_FakeSess(kw.get("imgsz", 640)), **{k: v for k, v in kw.items()})
    monkeypatch.setattr(onnx_detector.OnnxYoloDetector, "__init__", init)
    clf = onnx_detector.make_sim_classifier("m.onnx", imgsz=640)
    p = clf.score(np.random.default_rng(0), "drone", 20.0)
    assert 0.0 <= p <= 1.0 and p == pytest.approx(0.9)


def test_train_config_and_dry_run(tmp_path, capsys):
    cfg = train.load_config(Path(train.__file__).parent / "configs" / "rgb.yaml", ["epochs=3", "model=yolov8s-p2.yaml"])
    assert cfg["epochs"] == 3
    train.main(["--config", str(Path(train.__file__).parent / "configs" / "ir.yaml"), "--dry-run"])
    out = capsys.readouterr().out
    assert "8.3.172" in out and "export" in out
    with pytest.raises(SystemExit):
        train.load_config(Path(train.__file__).parent / "configs" / "rgb.yaml", ["evil_key=1"])


def test_model_card_flags_non_commercial(tmp_path):
    mix = tmp_path / "mix"
    mix.mkdir()
    (mix / "mix_report.md").write_text("Commercial-only filter: **off**.")
    cfg = {"name": "rgb", "data": str(mix / "data.yaml"), "model": "yolo11s.pt", "imgsz": 1280}
    p = train.model_card(cfg, tmp_path, None, None)
    assert "NOT cleared" in p.read_text()


def test_temporal_separates_flapping_birds(tmp_path):
    rng = np.random.default_rng(0)
    lines = []
    for k in range(60):
        fps, n = 25, 50
        t = np.arange(n) / fps
        if k % 2:  # bird: flapping size + wandering path
            s = 10 * (1 + 0.35 * np.sin(2 * np.pi * rng.uniform(3, 6) * t))
            x = np.cumsum(rng.normal(2, 3, n)); y = np.cumsum(rng.normal(0, 3, n))
            lab = "bird"
        else:      # drone: steady size, smooth path
            s = 10 * (1 + 0.03 * rng.normal(size=n))
            x = 3 * t * fps / 10; y = 0.5 * t
            lab = "drone"
        boxes = [[xi, yi, xi + si, yi + si] for xi, yi, si in zip(x, y, s)]
        lines.append(json.dumps({"label": lab, "fps": fps, "boxes": boxes}))
    f = tmp_path / "t.jsonl"
    f.write_text("\n".join(lines))
    X, y = temporal._load(f)
    m = temporal.fit(X[:40], y[:40])
    acc = ((temporal.predict(m, X[40:]) > 0.5) == y[40:]).mean()
    assert acc >= 0.9
