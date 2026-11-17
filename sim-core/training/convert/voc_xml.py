"""Pascal VOC XML (labelImg and most Chinese-hosted UAV datasets).

Sequence = folder holding the XML. The image is found by <filename>, or by the XML stem."""
from __future__ import annotations

from pathlib import Path

try:  # hardened parser if available; stdlib expat >= 2.4 also blocks entity-expansion bombs
    from defusedxml import ElementTree as ET  # type: ignore
except ImportError:  # pragma: no cover
    import xml.etree.ElementTree as ET  # noqa: S405

from .base import Box, SchemaError, Writer, cls_of, index_images, read_image


def convert(src: Path, out_root: Path, ds: dict, class_map: dict | None = None, frame_step: int = 1,
            keep_empty: bool = True, **_) -> list:
    src = Path(src)
    xmls = sorted(src.rglob("*.xml"))
    if not xmls:
        raise SchemaError(f"no .xml annotation files under {src}")
    imgs = index_images(src)
    w = Writer(out_root, ds, keep_empty=keep_empty)
    for k, xp in enumerate(xmls):
        if k % max(1, frame_step):
            continue
        try:
            root = ET.parse(str(xp)).getroot()
        except ET.ParseError as e:
            raise SchemaError(f"{xp}: bad XML: {e}") from None
        if root.tag != "annotation":
            continue  # not a VOC file (e.g. a config xml); skip
        fn = (root.findtext("filename") or "").strip()
        ip = imgs.get(Path(fn).name) or imgs.get(xp.stem)
        if ip is None:
            raise SchemaError(f"{xp}: image '{fn or xp.stem}' not found under {src}")
        boxes = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "other").strip()
            bb = obj.find("bndbox")
            if bb is None:
                raise SchemaError(f"{xp}: <object> without <bndbox>")
            try:
                x0, y0, x1, y1 = (float(bb.findtext(t)) for t in ("xmin", "ymin", "xmax", "ymax"))
            except (TypeError, ValueError):
                raise SchemaError(f"{xp}: non-numeric bndbox") from None
            boxes.append(Box(cls_of(name, class_map), x0, y0, x1, y1))
        seq = xp.parent.relative_to(src).as_posix() or "root"
        w.add(read_image(ip), boxes, seq, k, name=ip.stem)
    return w.rows
