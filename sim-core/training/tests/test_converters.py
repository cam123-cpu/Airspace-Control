import json

import cv2
import numpy as np
import pytest

from training.convert import get
from training.convert.base import SchemaError
from training.common import read_manifest, write_manifest

from .conftest import draw_target, sky, write_video


def _labels(root, rows):
    return [(root / r.label).read_text().split() for r in rows]


def _check_box(line, W, H, x0, y0, x1, y1, cls=0, tol=0.01):
    c, cx, cy, w, h = int(line[0]), *map(float, line[1:5])
    assert c == cls
    assert abs(cx - (x0 + x1) / 2 / W) < tol and abs(cy - (y0 + y1) / 2 / H) < tol
    assert abs(w - (x1 - x0) / W) < tol and abs(h - (y1 - y0) / H) < tol


def test_voc_xml(tmp_path, ds_yes):
    src = tmp_path / "src" / "seq1"
    src.mkdir(parents=True)
    cv2.imwrite(str(src / "f001.jpg"), sky())
    (src / "f001.xml").write_text("""<annotation><filename>f001.jpg</filename>
      <object><name>UAV</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax></bndbox></object>
      <object><name>bird</name><bndbox><xmin>100</xmin><ymin>50</ymin><xmax>120</xmax><ymax>60</ymax></bndbox></object>
      </annotation>""")
    out = tmp_path / "out"
    rows = get("voc_xml").convert(tmp_path / "src", out, ds_yes)
    assert len(rows) == 1 and rows[0].sequence == "usc_drone/seq1" and rows[0].commercial_ok == "yes"
    lines = (out / rows[0].label).read_text().splitlines()
    _check_box(lines[0].split(), 160, 120, 10, 20, 30, 40, cls=0)
    _check_box(lines[1].split(), 160, 120, 100, 50, 120, 60, cls=2)


def test_voc_xml_missing_image_is_loud(tmp_path, ds_yes):
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "a.xml").write_text("<annotation><filename>nope.jpg</filename></annotation>")
    with pytest.raises(SchemaError, match="not found"):
        get("voc_xml").convert(tmp_path / "s", tmp_path / "o", ds_yes)


def test_coco_json(tmp_path, ds_yes):
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    cv2.imwrite(str(src / "images" / "a.jpg"), sky())
    cv2.imwrite(str(src / "images" / "b.jpg"), sky(seed=1))
    coco = {"images": [{"id": 1, "file_name": "a.jpg", "width": 160, "height": 120},
                       {"id": 2, "file_name": "b.jpg", "width": 160, "height": 120}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 10, 20, 10]}],
            "categories": [{"id": 7, "name": "drone"}]}
    (src / "ann.json").write_text(json.dumps(coco))
    rows = get("coco_json").convert(src, tmp_path / "out", ds_yes)
    assert len(rows) == 2
    labs = {r.image.split("/")[-1]: (tmp_path / "out" / r.label).read_text() for r in rows}
    assert labs["b.jpg"] == ""  # empty frame kept as a negative
    _check_box(labs["a.jpg"].split(), 160, 120, 10, 10, 30, 20)


def test_yolo_passthrough_remaps_classes(tmp_path, ds_yes):
    src = tmp_path / "src"
    (src / "images" / "v1").mkdir(parents=True)
    (src / "labels" / "v1").mkdir(parents=True)
    cv2.imwrite(str(src / "images" / "v1" / "x.jpg"), sky())
    (src / "labels" / "v1" / "x.txt").write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n")
    (src / "data.yaml").write_text("names: ['Helicopter', 'Drone']\n")
    rows = get("yolo_passthrough").convert(src, tmp_path / "out", ds_yes)
    lines = (tmp_path / "out" / rows[0].label).read_text().splitlines()
    assert [l.split()[0] for l in lines] == ["3", "0"]
    assert rows[0].sequence == "usc_drone/v1"


def test_yolo_passthrough_bad_class(tmp_path, ds_yes):
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir(parents=True)
    cv2.imwrite(str(src / "images" / "x.jpg"), sky())
    (src / "labels" / "x.txt").write_text("5 0.5 0.5 0.1 0.1\n")
    (src / "classes.txt").write_text("drone\n")
    with pytest.raises(SchemaError, match="outside names"):
        get("yolo_passthrough").convert(src, tmp_path / "out", ds_yes)


def test_wosdetc_video(tmp_path, ds_no):
    src = tmp_path / "src"
    src.mkdir()
    frames = [draw_target(sky(seed=i), 40 + i * 4, 30, 60 + i * 4, 45) for i in range(6)]
    write_video(src / "gopro_001.mp4", frames)
    (src / "gopro_001.txt").write_text("0 1 40 30 20 15 drone\n2 2 48 30 20 15 drone 100 80 10 6 bird\n4 0\n")
    rows = get("wosdetc_txt").convert(src, tmp_path / "out", ds_no, frame_step=1)
    assert sorted(r.frame for r in rows) == [0, 2, 4]
    by = {r.frame: (tmp_path / "out" / r.label).read_text().splitlines() for r in rows}
    assert len(by[2]) == 2 and by[2][1].split()[0] == "2" and by[4] == []
    assert all(r.commercial_ok == "no" for r in rows)


def test_wosdetc_bad_line(tmp_path, ds_no):
    src = tmp_path / "s"
    src.mkdir()
    write_video(src / "v.mp4", [sky()] * 2)
    (src / "v.txt").write_text("0 2 1 2 3 4 drone\n")
    with pytest.raises(SchemaError, match="2 objects declared"):
        get("wosdetc_txt").convert(src, tmp_path / "o", ds_no)


def test_rect_txt_otb_frames(tmp_path, ds_yes):
    seq = tmp_path / "src" / "video07"
    (seq / "img").mkdir(parents=True)
    for i in range(4):
        cv2.imwrite(str(seq / "img" / f"{i + 1:04d}.jpg"), sky(seed=i))
    (seq / "groundtruth_rect.txt").write_text("10,10,20,10\n0,0,0,0\nNaN,NaN,NaN,NaN\n12\t11\t20\t10\n")
    rows = get("rect_txt").convert(tmp_path / "src", tmp_path / "out", ds_yes, gt_name="groundtruth*.txt")
    assert len(rows) == 4
    labs = [(tmp_path / "out" / r.label).read_text() for r in sorted(rows, key=lambda r: r.frame)]
    assert labs[1] == "" and labs[2] == "" and labs[0] and labs[3]


def test_rect_txt_usc_center_with_frame_col(tmp_path, ds_yes):
    seq = tmp_path / "src" / "clip"
    seq.mkdir(parents=True)
    for n in (1, 16, 31):
        cv2.imwrite(str(seq / f"{n:04d}.jpg"), sky(seed=n))
    (seq / "groundtruth.txt").write_text("1 50 40 20 10\n16 60 40 20 10\n")
    rows = get("rect_txt").convert(tmp_path / "src", tmp_path / "out", ds_yes, fmt="cxcywh", frame_col=True)
    assert sorted(r.frame for r in rows) == [1, 16]  # frame 31 unlabelled -> skipped
    r1 = next(r for r in rows if r.frame == 1)
    _check_box((tmp_path / "out" / r1.label).read_text().split(), 160, 120, 40, 35, 60, 45)


def test_anti_uav_frames_and_schema_check(tmp_path, ds_yes):
    seq = tmp_path / "src" / "20190925_101846_1_1"
    (seq / "IR").mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(seq / "IR" / f"{i:05d}.jpg"), sky(seed=i))
    (seq / "IR_label.json").write_text(json.dumps({"exist": [1, 0, 1], "gt_rect": [[10, 10, 8, 6], [], [12, 10, 8, 6]]}))
    rows = get("anti_uav_json").convert(tmp_path / "src", tmp_path / "out", ds_yes, frame_step=1)
    assert len(rows) == 3 and all(r.modality == "ir" for r in rows)
    (seq / "IR_label.json").write_text(json.dumps({"exist": [1], "gt_rect": [[1, 1, 2, 2], [1, 1, 2, 2]]}))
    with pytest.raises(SchemaError, match="lengths differ"):
        get("anti_uav_json").convert(tmp_path / "src", tmp_path / "out2", ds_yes)


def test_aot_json(tmp_path, ds_yes):
    part = tmp_path / "src" / "part1"
    (part / "ImageSets").mkdir(parents=True)
    (part / "Images" / "flightA").mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(part / "Images" / "flightA" / f"img{i}.png"), sky(seed=i)[..., 0])
    gt = {"samples": {"flightA": {"metadata": {}, "entities": [
        {"img_name": "img0.png", "bb": [10, 10, 6, 4], "id": "Drone1", "blob": {"frame": 0}},
        {"img_name": "img1.png", "bb": [20, 10, 6, 4], "id": "Airplane2", "blob": {"frame": 1}},
        {"img_name": "img1.png", "bb": [50, 50, 6, 4], "id": "Bird3", "blob": {"frame": 1}},
        {"img_name": "img2.png", "id": "Drone1", "blob": {"frame": 2}}]}}}
    (part / "ImageSets" / "groundtruth.json").write_text(json.dumps(gt))
    rows = get("aot_json").convert(tmp_path / "src", tmp_path / "out", ds_yes, frame_step=1)
    assert len(rows) == 2  # img2 has no bb and keep_empty=False by default
    labs = {r.image.split("/")[-1]: (tmp_path / "out" / r.label).read_text().split("\n") for r in rows}
    assert labs["img1.jpg"][0].split()[0] == "3" and labs["img1.jpg"][1].split()[0] == "2"


def test_halmstad_csv(tmp_path, ds_yes):
    src = tmp_path / "src"
    src.mkdir()
    write_video(src / "IR_DRONE_001.mp4", [sky(seed=i) for i in range(4)])
    (src / "labels.csv").write_text("video,frame,class,x,y,w,h\nIR_DRONE_001,1,drone,10,10,8,6\n"
                                    "IR_DRONE_001,2,,0,0,0,0\nIR_DRONE_001,4,bird,20,20,8,6\n")
    rows = get("halmstad_csv").convert(src, tmp_path / "out", ds_yes, frame_step=1)
    assert sorted(r.frame for r in rows) == [1, 2, 4] and all(r.modality == "ir" for r in rows)


def test_cli_appends_manifest(tmp_path):
    from training.convert import run
    src = tmp_path / "s" / "seq"
    src.mkdir(parents=True)
    cv2.imwrite(str(src / "a.jpg"), sky())
    (src / "a.xml").write_text("<annotation><filename>a.jpg</filename></annotation>")
    n1 = run("det_fly", tmp_path / "s", tmp_path / "o")
    n2 = run("det_fly", tmp_path / "s", tmp_path / "o")
    rows = list(read_manifest(tmp_path / "o" / "manifest.jsonl"))
    assert n1 == n2 == 1 and len(rows) == 2 and rows[0].commercial_ok == "unclear"
