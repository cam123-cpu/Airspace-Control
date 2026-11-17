"""Download the datasets in datasets.yaml that can be fetched unattended, and print exact
instructions for the ones that need a human (registration, agreements, Baidu/Drive links).

    python -m training.fetch --list
    python -m training.fetch --dataset halmstad_multisensor --out /data/raw
    python -m training.fetch --all-direct --out /data/raw --commercial-only
    python -m training.fetch --dataset aot --out /data/raw --max-sequences 200
    PIXABAY_API_KEY=... python -m training.fetch --dataset pixabay_video --out /data/raw --accept-pixabay-terms

Safety: HTTPS only to an allowlist of hosts, size caps, resumable downloads with published
checksums verified, rate limiting, archives extracted without path traversal, API keys read
from environment variables only (never argv, never logged). Every fetched file gets a row in
<out>/fetch_manifest.jsonl with its source and licence.
"""
from __future__ import annotations

import re
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from .common import (CatalogError, ManifestRow, dataset_by_id, file_md5, load_catalog, safe_extract,
                     safe_join, safe_name, write_manifest)

ALLOWED_HOSTS = ("zenodo.org", "figshare.com", "figsharecdn.com", "amazonaws.com", "pixabay.com",
                 "pexels.com", "huggingface.co", "hf.co")
MAX_FILE_BYTES = 200 * 1024 ** 3      # refuse single files over 200 GB
USER_AGENT = "dronesim-training-fetch/1.0 (+research; contact via repo)"
MIN_INTERVAL_S = 1.0                  # polite default between API calls

STOCK_QUERIES = ["night sky", "drone flying", "birds flying sky", "flock of birds", "stadium night",
                 "city skyline night", "airport night", "seagulls", "prison fence", "power substation"]


class FetchError(RuntimeError):
    pass


# ------------------------------------------------------------------ http

_last_call = [0.0]


def _throttle(min_interval: float = MIN_INTERVAL_S):
    dt = time.monotonic() - _last_call[0]
    if dt < min_interval:
        time.sleep(min_interval - dt)
    _last_call[0] = time.monotonic()


def check_url(url: str) -> str:
    u = urlparse(url)
    if u.scheme != "https":
        raise FetchError(f"refusing non-HTTPS URL: {url}")
    host = (u.hostname or "").lower()
    if not any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS):
        raise FetchError(f"host not on allowlist: {host}")
    return url


def _session():
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def get_json(url: str, params: dict | None = None, session=None) -> dict:
    check_url(url)
    s = session or _session()
    for attempt in range(5):
        _throttle()
        r = s.get(url, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(min(60, 2 ** attempt * 5))
            continue
        r.raise_for_status()
        return r.json()
    raise FetchError(f"rate limited repeatedly: {url}")


def download(url: str, dest: Path, size: int | None = None, md5: str | None = None, session=None,
             retries: int = 5) -> Path:
    """Resumable streaming download with size cap and optional md5 verification."""
    check_url(url)
    if size and size > MAX_FILE_BYTES:
        raise FetchError(f"{dest.name}: {size} bytes exceeds MAX_FILE_BYTES")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and md5 and file_md5(dest) == md5:
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    s = session or _session()
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            _throttle(0.2)
            with s.get(url, headers=headers, stream=True, timeout=120, allow_redirects=True) as r:
                check_url(r.url)  # the redirect target must be allowlisted too
                if r.status_code == 416:  # already complete
                    break
                r.raise_for_status()
                mode = "ab" if (have and r.status_code == 206) else "wb"
                written = have if mode == "ab" else 0
                with open(part, mode) as f:
                    for chunk in r.iter_content(1 << 20):
                        written += len(chunk)
                        if written > MAX_FILE_BYTES:
                            raise FetchError(f"{dest.name}: exceeded MAX_FILE_BYTES while streaming")
                        f.write(chunk)
            break
        except FetchError:
            raise
        except Exception as e:  # network hiccup: back off and resume
            if attempt == retries - 1:
                raise FetchError(f"{url}: {e}") from None
            time.sleep(2 ** attempt)
    if md5 and file_md5(part) != md5:
        part.unlink(missing_ok=True)
        raise FetchError(f"{dest.name}: md5 mismatch (corrupt or tampered download)")
    part.replace(dest)
    return dest


# ------------------------------------------------------------------ sources

def _row(ds, path: Path, root: Path, sequence: str, label=None, extra=None) -> ManifestRow:
    return ManifestRow(image=str(path.relative_to(root)), label=label, source=ds["id"], license=str(ds["license"]),
                       commercial_ok=str(ds["commercial_ok"]), sequence=sequence,
                       modality=(ds.get("modality") or ["rgb"])[0], extra=extra or {})


def fetch_zenodo(ds: dict, out: Path, extract: bool = True, **_) -> list[ManifestRow]:
    rec = str(ds["fetch"]["record"])
    meta = get_json(f"https://zenodo.org/api/records/{rec}")
    files = meta.get("files") or []
    if not files:
        raise FetchError(f"zenodo record {rec} lists no files (restricted access?)")
    dest_dir = safe_join(out, safe_name(ds["id"]))
    rows = []
    for f in files:
        key = safe_name(f.get("key") or f.get("filename"))
        url = (f.get("links") or {}).get("content") or (f.get("links") or {}).get("self")
        md5 = str(f.get("checksum", "")).removeprefix("md5:") or None
        p = download(url, dest_dir / key, size=f.get("size"), md5=md5)
        rows.append(_row(ds, p, out, f"{ds['id']}/archive", extra={"md5": md5, "zenodo_record": rec}))
        if extract and p.suffix.lower() in (".zip", ".tar", ".gz", ".tgz"):
            safe_extract(p, dest_dir / p.stem)
    return rows


def fetch_figshare(ds: dict, out: Path, extract: bool = True, **_) -> list[ManifestRow]:
    art = str(ds["fetch"]["article"])
    meta = get_json(f"https://api.figshare.com/v2/articles/{art}")
    dest_dir = safe_join(out, safe_name(ds["id"]))
    rows = []
    for f in meta.get("files", []):
        md5 = f.get("supplied_md5") or f.get("computed_md5") or None
        p = download(f["download_url"], dest_dir / safe_name(f["name"]), size=f.get("size"), md5=md5)
        rows.append(_row(ds, p, out, f"{ds['id']}/archive", extra={"md5": md5, "figshare_article": art}))
        if extract and p.suffix.lower() in (".zip", ".tar", ".gz", ".tgz"):
            safe_extract(p, dest_dir / p.stem)
    return rows


def _aws() -> str:
    exe = shutil.which("aws")
    if not exe:
        raise FetchError("AWS CLI not found: pip install awscli (no AWS account needed, --no-sign-request)")
    return exe


def fetch_s3(ds: dict, out: Path, max_sequences: int | None = 50, parts: list[str] | None = None,
             **_) -> list[ManifestRow]:
    """AOT: groundtruth.json for each part, then the first N flight folders of Images/."""
    cfg = ds["fetch"]
    bucket = cfg["bucket"].rstrip("/") + "/"
    base = [_aws(), "--region", cfg.get("region", "us-east-1")] + (["--no-sign-request"] if cfg.get("no_sign_request") else [])
    dest = safe_join(out, safe_name(ds["id"]))
    if parts is None:
        ls = subprocess.run(base + ["s3", "ls", bucket], capture_output=True, text=True, check=True)
        parts = [l.split()[-1].rstrip("/") for l in ls.stdout.splitlines() if l.strip().startswith("PRE")]
    rows, taken = [], 0
    for part in parts:
        part = safe_name(part)
        subprocess.run(base + ["s3", "cp", f"{bucket}{part}/ImageSets/groundtruth.json",
                               str(dest / part / "ImageSets" / "groundtruth.json")], check=True)
        ls = subprocess.run(base + ["s3", "ls", f"{bucket}{part}/Images/"], capture_output=True, text=True, check=True)
        flights = [l.split()[-1].rstrip("/") for l in ls.stdout.splitlines() if l.strip().startswith("PRE")]
        for fl in flights:
            if max_sequences is not None and taken >= max_sequences:
                return rows
            fl = safe_name(fl)
            target = dest / part / "Images" / fl
            subprocess.run(base + ["s3", "sync", f"{bucket}{part}/Images/{fl}/", str(target), "--only-show-errors"],
                           check=True)
            rows.append(_row(ds, target, out, f"{ds['id']}/{fl}", extra={"part": part}))
            taken += 1
    return rows


def fetch_huggingface(ds: dict, out: Path, **_) -> list[ManifestRow]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise FetchError("pip install huggingface_hub, or git lfs clone "
                         f"https://huggingface.co/datasets/{ds['fetch']['repo']}") from None
    rev = ds["fetch"].get("revision")
    if not (isinstance(rev, str) and re.fullmatch(r"[0-9a-f]{40}", rev)):
        raise FetchError(f"{ds['id']}: pin fetch.revision to a 40-character commit hash before downloading (supply-chain safety)")
    dest = safe_join(out, safe_name(ds["id"]))
    p = Path(snapshot_download(repo_id=ds["fetch"]["repo"], repo_type="dataset", revision=rev, local_dir=str(dest)))
    return [_row(ds, p, out, f"{ds['id']}/snapshot")]


def extract_frames(video: Path, out_dir: Path, fps: float = 1.0, max_frames: int = 60) -> list[Path]:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FetchError("ffmpeg not found")
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vf", f"fps={fps}",
                    "-frames:v", str(int(max_frames)), "-q:v", "2", str(out_dir / "%05d.jpg")], check=True)
    return sorted(out_dir.glob("*.jpg"))


def fetch_pixabay(ds: dict, out: Path, accept_pixabay_terms: bool = False, per_query: int = 5,
                  max_total: int = 50, queries: list[str] | None = None, fps: float = 1.0, **_) -> list[ManifestRow]:
    if not accept_pixabay_terms:
        raise FetchError("Pixabay's API rules forbid systematic mass downloads and require showing the source. "
                         "Read https://pixabay.com/api/docs/ and the content licence, then pass "
                         "--accept-pixabay-terms. Volume is capped (--max-total, default 50).")
    key = os.environ.get(ds["fetch"]["env_key"])
    if not key:
        raise FetchError(f"set {ds['fetch']['env_key']} in the environment (never pass keys on the command line)")
    max_total = min(int(max_total), 200)
    dest = safe_join(out, safe_name(ds["id"]))
    cache = dest / "api_cache"
    cache.mkdir(parents=True, exist_ok=True)
    rows, n = [], 0
    for q in queries or STOCK_QUERIES:
        cf = cache / (safe_name(q) + ".json")
        if cf.exists() and time.time() - cf.stat().st_mtime < 24 * 3600:  # API rule: cache 24 h
            res = json.loads(cf.read_text(encoding="utf-8"))
        else:
            res = get_json("https://pixabay.com/api/videos/", {"key": key, "q": q, "per_page": max(3, per_query),
                                                               "safesearch": "true"})
            cf.write_text(json.dumps(res), encoding="utf-8")
        for hit in res.get("hits", [])[:per_query]:
            if n >= max_total:
                return rows
            vids = hit.get("videos", {})
            v = vids.get("medium") or vids.get("small") or vids.get("large")
            if not v or not v.get("url"):
                continue
            vid_id = safe_name(str(hit["id"]))
            mp4 = download(v["url"], dest / "videos" / f"{vid_id}.mp4")
            for fp in extract_frames(mp4, dest / "frames" / vid_id, fps=fps):
                rows.append(_row(ds, fp, out, f"{ds['id']}/{vid_id}", extra={
                    "query": q, "page_url": hit.get("pageURL"), "user": hit.get("user"),
                    "attribution": f"Video by {hit.get('user')} on Pixabay ({hit.get('pageURL')})"}))
            n += 1
    return rows


def fetch_pexels(ds: dict, out: Path, pexels_permission_ref: str | None = None, **_) -> list[ManifestRow]:
    if not pexels_permission_ref:
        raise FetchError("Pexels Terms of Service s.8 prohibit automated collection 'for machine learning "
                         "purposes' and bulk copying without permission. Get written permission from Pexels, "
                         "then pass --pexels-permission-ref <ticket/email id>. Not fetching.")
    raise FetchError("Pexels fetching is deliberately not implemented until the scope of your permission is "
                     "known; add it here to match what Pexels granted (reference: %s)" % safe_name(pexels_permission_ref))


METHODS = {"zenodo": fetch_zenodo, "figshare": fetch_figshare, "s3": fetch_s3, "huggingface": fetch_huggingface,
           "pixabay": fetch_pixabay, "pexels": fetch_pexels}


def instructions(ds: dict) -> str:
    lines = [f"[{ds['id']}] {ds['name']}", f"  licence: {ds['license']}  (commercial_ok: {ds['commercial_ok']})",
             f"  access: {ds['access']}  url: {ds.get('url')}"]
    if ds.get("request"):
        lines.append(f"  how: {ds['request']}")
    for k, v in (ds.get("links") or {}).items():
        lines.append(f"  {k}: {v}")
    if ds.get("converter"):
        lines.append(f"  then: python -m training.convert {ds['id']} <extracted_dir> <converted_root>")
    if ds.get("notes"):
        lines.append(f"  note: {ds['notes']}")
    return "\n".join(lines)


def fetch(ds: dict, out: Path, **opts) -> list[ManifestRow]:
    method = (ds.get("fetch") or {}).get("method")
    if ds["access"] != "direct" or method not in METHODS:
        print(instructions(ds))
        return []
    rows = METHODS[method](ds, out, **opts)
    write_manifest(rows, out / "fetch_manifest.jsonl", append=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the catalog with licences and access")
    ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--all-direct", action="store_true", help="every dataset with access: direct")
    ap.add_argument("--commercial-only", action="store_true", help="skip anything not commercial_ok: yes")
    ap.add_argument("--out", type=Path, default=Path("raw"))
    ap.add_argument("--max-sequences", type=int, default=50, help="AOT flights to fetch")
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--accept-pixabay-terms", action="store_true")
    ap.add_argument("--max-total", type=int, default=50, help="Pixabay videos in total (hard cap 200)")
    ap.add_argument("--pexels-permission-ref")
    a = ap.parse_args(argv)
    cat = load_catalog()
    if a.list:
        for d in cat["datasets"]:
            print(f"{d['id']:<24} {d['access']:<11} commercial_ok={d['commercial_ok']:<20} {d['license'][:70]}")
        return 0
    ids = list(a.dataset)
    if a.all_direct:
        ids += [d["id"] for d in cat["datasets"] if d["access"] == "direct" and d.get("fetch")
                and d["fetch"]["method"] not in ("pixabay", "pexels")]
    if not ids:
        ap.error("give --dataset ID (repeatable), --all-direct or --list")
    a.out.mkdir(parents=True, exist_ok=True)
    rc = 0
    for i in dict.fromkeys(ids):
        try:
            ds = dataset_by_id(cat, i)
        except CatalogError as e:
            print(f"error: {e}", file=sys.stderr)
            rc = 2
            continue
        if a.commercial_only and str(ds["commercial_ok"]) != "yes":
            print(f"skip {i}: commercial_ok={ds['commercial_ok']}")
            continue
        try:
            rows = fetch(ds, a.out, max_sequences=a.max_sequences, extract=not a.no_extract,
                         accept_pixabay_terms=a.accept_pixabay_terms, max_total=a.max_total,
                         pexels_permission_ref=a.pexels_permission_ref)
            if rows:
                print(f"{i}: {len(rows)} items -> {a.out}")
        except (FetchError, subprocess.CalledProcessError) as e:
            print(f"error fetching {i}: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
