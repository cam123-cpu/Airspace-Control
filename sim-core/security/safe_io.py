"""Hardened file-output helpers shared by the CLI tools.

Threats addressed (see SECURITY_AUDIT.md):
  * writing outside the intended output directory (path traversal via CLI/job config)
  * symlink-following writes (a planted symlink redirecting output onto another file)
  * partially written / torn files when a run is killed (atomic replace)
  * world-readable outputs (files are created 0o640, dirs 0o750)
  * unbounded numeric arguments that exhaust CPU, RAM or disk
"""
import json
import math
import os
import stat
import tempfile

FILE_MODE = 0o640
DIR_MODE = 0o750


class UnsafePathError(ValueError):
    pass


def bounded_int(lo, hi, name="value"):
    """argparse type: integer in [lo, hi]."""
    def parse(s):
        try:
            v = int(s, 10)
        except (TypeError, ValueError):
            raise _arg_error(f"{name} must be an integer, got {s!r}")
        if not lo <= v <= hi:
            raise _arg_error(f"{name} must be between {lo} and {hi}, got {v}")
        return v
    parse.__name__ = name
    return parse


def bounded_float(lo, hi, name="value"):
    """argparse type: finite float in [lo, hi] (rejects nan/inf)."""
    def parse(s):
        try:
            v = float(s)
        except (TypeError, ValueError):
            raise _arg_error(f"{name} must be a number, got {s!r}")
        if not math.isfinite(v) or not lo <= v <= hi:
            raise _arg_error(f"{name} must be a finite number between {lo} and {hi}, got {s!r}")
        return v
    parse.__name__ = name
    return parse


def _arg_error(msg):
    import argparse
    return argparse.ArgumentTypeError(msg)


def _has_symlink_component(path):
    """True if any existing component of `path` (absolute, normalised) is a symlink."""
    cur = os.path.sep
    for part in path.strip(os.path.sep).split(os.path.sep):
        cur = os.path.join(cur, part)
        if os.path.lexists(cur) and os.path.islink(cur):
            return True
        if not os.path.lexists(cur):
            break
    return False


def resolve_output_dir(path, base=None, allow_outside=False):
    """Validate and create an output directory.

    - The path is made absolute and normalised (no `..` games).
    - It must lie inside `base` (default: current working directory) unless allow_outside.
    - No existing component may be a symlink (prevents redirected writes).
    - Created with mode 0o750.
    """
    if not isinstance(path, str) or not path or "\x00" in path:
        raise UnsafePathError("output path is empty or contains a NUL byte")
    base = os.path.abspath(base or os.getcwd())
    target = os.path.abspath(os.path.normpath(path))
    if not allow_outside and os.path.commonpath([base, target]) != base:
        raise UnsafePathError(f"refusing to write outside {base}: {target} (pass --allow-outside to override)")
    if _has_symlink_component(target):
        raise UnsafePathError(f"refusing to write through a symlink: {target}")
    os.makedirs(target, mode=DIR_MODE, exist_ok=True)
    if not stat.S_ISDIR(os.lstat(target).st_mode):
        raise UnsafePathError(f"not a directory: {target}")
    return target


def safe_join(root, *parts):
    """Join under root; reject absolute parts, `..` escapes and symlinked targets."""
    p = os.path.abspath(os.path.join(root, *parts))
    if os.path.commonpath([os.path.abspath(root), p]) != os.path.abspath(root):
        raise UnsafePathError(f"path escapes output directory: {p}")
    if os.path.lexists(p) and os.path.islink(p):
        raise UnsafePathError(f"refusing to overwrite a symlink: {p}")
    return p


def atomic_write_bytes(path, data, fsync=True):
    """Write to a private temp file in the same directory, fsync, then os.replace.
    os.replace swaps the directory entry, so a symlink planted at `path` is replaced,
    never followed."""
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            if fsync:
                os.fsync(f.fileno())
        os.chmod(tmp, FILE_MODE)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path, text, fsync=True):
    atomic_write_bytes(path, text.encode("utf-8"), fsync=fsync)


def atomic_write_json(path, obj, **kw):
    kw.setdefault("allow_nan", False)  # NaN/Infinity are not valid JSON and break strict parsers
    atomic_write_text(path, json.dumps(obj, **kw))


def yaml_scalar(s):
    """Quote a string for YAML. A JSON string is a valid YAML double-quoted scalar,
    so colons, '#', newlines etc. in a path cannot inject keys."""
    return json.dumps(str(s))
