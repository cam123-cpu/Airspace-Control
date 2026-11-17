"""Regression tests for the security audit fixes (SECURITY_AUDIT.md).

Run from the package root:
    python -m unittest discover -s security/tests -t . -v
"""
import json
import os
import shutil
import stat
import sys
import tempfile
import tracemalloc
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from security import safe_io  # noqa: E402
from security.safe_classifier import SafeClassifier  # noqa: E402


class _InTempDir(unittest.TestCase):
    def setUp(self):
        self.old = os.getcwd()
        self.tmp = tempfile.mkdtemp(prefix="dronesim-sec-")
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self.old)
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestSafeIO(_InTempDir):
    def test_rejects_path_outside_base(self):  # F-04
        with self.assertRaises(safe_io.UnsafePathError):
            safe_io.resolve_output_dir("../escape")
        with self.assertRaises(safe_io.UnsafePathError):
            safe_io.resolve_output_dir("/etc/dronesim")

    def test_allow_outside_is_explicit(self):
        other = tempfile.mkdtemp(prefix="dronesim-out-")
        try:
            self.assertEqual(safe_io.resolve_output_dir(other, allow_outside=True), other)
        finally:
            shutil.rmtree(other)

    def test_rejects_symlinked_output_dir(self):  # F-04
        victim = tempfile.mkdtemp(prefix="dronesim-victim-")
        try:
            os.symlink(victim, "out")
            with self.assertRaises(safe_io.UnsafePathError):
                safe_io.resolve_output_dir("out")
            os.makedirs("real/sub")
            os.symlink(victim, "real/link")
            with self.assertRaises(safe_io.UnsafePathError):
                safe_io.resolve_output_dir("real/link/deeper")
        finally:
            shutil.rmtree(victim)

    def test_safe_join_blocks_escape_and_symlink_targets(self):
        d = safe_io.resolve_output_dir("o")
        with self.assertRaises(safe_io.UnsafePathError):
            safe_io.safe_join(d, "../x")
        with self.assertRaises(safe_io.UnsafePathError):
            safe_io.safe_join(d, "/etc/passwd")
        os.symlink("/etc/hostname", os.path.join(d, "planted.json"))
        with self.assertRaises(safe_io.UnsafePathError):
            safe_io.safe_join(d, "planted.json")

    def test_atomic_write_never_follows_symlink(self):
        d = safe_io.resolve_output_dir("o")
        target = os.path.join(self.tmp, "victim.txt")
        with open(target, "w") as f:
            f.write("original")
        link = os.path.join(d, "report.md")
        os.symlink(target, link)
        safe_io.atomic_write_text(link, "new")  # replaces the link itself
        with open(target) as f:
            self.assertEqual(f.read(), "original")
        self.assertFalse(os.path.islink(link))

    def test_file_and_dir_permissions(self):  # F-11
        old = os.umask(0)
        try:
            d = safe_io.resolve_output_dir("perm")
            p = os.path.join(d, "x.json")
            safe_io.atomic_write_json(p, {"a": 1})
        finally:
            os.umask(old)
        self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o640)
        self.assertEqual(stat.S_IMODE(os.stat(d).st_mode) & 0o007, 0)

    def test_json_rejects_nan(self):  # F-06
        with self.assertRaises(ValueError):
            safe_io.atomic_write_json(os.path.join(self.tmp, "n.json"), {"x": float("nan")})

    def test_yaml_scalar_cannot_inject(self):  # F-05
        evil = "data\ntrain: /etc\n#"
        line = f"path: {safe_io.yaml_scalar(evil)}"
        self.assertEqual(line.count("\n"), 0)

    def test_bounded_args(self):  # F-02
        import argparse
        p = safe_io.bounded_int(1, 10, "--n")
        self.assertEqual(p("5"), 5)
        for bad in ("0", "11", "-1", "1e9", "abc", ""):
            with self.assertRaises(argparse.ArgumentTypeError):
                p(bad)
        f = safe_io.bounded_float(0, 1, "--x")
        for bad in ("nan", "inf", "-inf", "2"):
            with self.assertRaises(argparse.ArgumentTypeError):
                f(bad)


class TestRenderLimits(unittest.TestCase):  # F-01
    def test_huge_px_is_bounded_in_memory(self):
        from render import render_frame
        rng = np.random.default_rng(0)
        tracemalloc.start()
        render_frame(rng, "night", "drone_payload", px=5000)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        self.assertLess(peak, 600e6, f"render used {peak / 1e6:.0f} MB")

    def test_rejects_bad_inputs(self):
        from render import render_frame
        rng = np.random.default_rng(0)
        for args in [("bogus", "drone", 10), ("night", "drone", float("nan")), ("night", "drone", -1),
                     ("night", "tank", 10), ("night", "drone", float("inf"))]:
            with self.assertRaises(ValueError, msg=str(args)):
                render_frame(rng, *args)


class TestGenerateDatasetCLI(_InTempDir):
    def _gen(self, *argv):
        import generate_dataset
        return generate_dataset.main(list(argv))

    def test_end_to_end_and_split(self):  # F-09
        self._gen("--n", "12", "--out", "d", "--val-frac", "0.25")
        with open("d/train.txt") as f:
            train = set(f.read().split())
        with open("d/val.txt") as f:
            val = set(f.read().split())
        self.assertEqual(len(train), 9)
        self.assertEqual(len(val), 3)
        self.assertFalse(train & val)
        with open("d/data.yaml") as f:
            y = f.read()
        self.assertIn('val: "val.txt"', y)
        json.load(open("d/coco.json"))

    def test_refuses_non_empty_dir_without_overwrite(self):  # F-12
        os.makedirs("d")
        open("d/keep.txt", "w").close()
        with self.assertRaises(SystemExit):
            self._gen("--n", "1", "--out", "d")

    def test_refuses_outside_and_bad_n(self):
        with self.assertRaises(SystemExit):
            self._gen("--n", "1", "--out", "/etc/dronesim-test")
        with self.assertRaises(SystemExit):
            self._gen("--n", "-5")
        with self.assertRaises(SystemExit):
            self._gen("--n", "999999999")

    def test_deterministic(self):  # F-08
        self._gen("--n", "4", "--out", "a", "--seed", "3")
        self._gen("--n", "4", "--out", "b", "--seed", "3")
        for name in ("labels/000000.txt", "labels/000003.txt", "train.txt", "val.txt"):
            with open(os.path.join("a", name)) as f1, open(os.path.join("b", name)) as f2:
                self.assertEqual(f1.read(), f2.read(), name)


class TestEvaluateCLI(_InTempDir):
    def _args(self, *argv):
        import evaluate
        return evaluate.parse_args(list(argv))

    def test_rejects_resource_exhaustion_and_garbage(self):  # F-02
        for argv in (["--nights", "0"], ["--nights", "100000000"], ["--procs", "0"], ["--procs", "100000"],
                     ["--threshold", "0.77"], ["--sweep", "nan"], ["--sweep", ",".join(["1"] * 11)],
                     ["--classifier-strength", "inf"]):
            with self.assertRaises(SystemExit, msg=str(argv)):
                self._args(*argv)

    def test_export_sample_path_checks(self):  # F-04
        import evaluate
        with self.assertRaises(SystemExit):
            evaluate.main(["--export-sample", "/etc/x.json"])
        with self.assertRaises(SystemExit):
            evaluate.main(["--export-sample", "x.txt"])

    def test_clean_makes_strict_json(self):  # F-06
        import evaluate
        out = evaluate._clean({"a": float("nan"), "b": np.float64(2.5), "c": np.array([1, np.inf]), 3: (np.int64(4),)})
        self.assertEqual(out, {"a": None, "b": 2.5, "c": [1.0, None], "3": [4]})
        json.dumps(out, allow_nan=False)


class TestClassifierFailSafe(unittest.TestCase):  # F-03
    class NaNModel:
        def score(self, *a, **k):
            return float("nan")

    class Boom:
        def score(self, *a, **k):
            raise RuntimeError("cuda error")

    def test_wrapper_neutralises_bad_outputs_and_reports_health(self):
        faults = []
        c = SafeClassifier(self.NaNModel(), max_consecutive_faults=3, on_fault=lambda w, n: faults.append(n))
        self.assertEqual(c.score(None, "drone", 10), 0.5)
        self.assertTrue(c.healthy)
        c.score(None, "drone", 10); c.score(None, "drone", 10)
        self.assertFalse(c.healthy)
        self.assertEqual(faults, [1, 2, 3])
        self.assertEqual(SafeClassifier(self.Boom()).score(None, "drone", 10), 0.5)
        for v in (-0.1, 1.5, "x", None, float("inf")):
            class M:
                def score(self, *a, _v=v, **k):
                    return _v
            self.assertEqual(SafeClassifier(M()).score(), 0.5, v)

    def test_nan_model_without_wrapper_suppresses_alerts(self):
        """Documents the fail-open bug in fusion.update_camera (pending patch P-1)
        and proves the wrapper restores alerting for an RF-emitting drone."""
        from scenario import make_episode
        from sim import run_episode
        from sensors import StandInClassifier

        def first_alert(params, seed=5):
            ep = make_episode(np.random.default_rng(seed), "drone")
            return run_episode(ep, seed, params)["drones"][0]["first_alert"][0.9]

        self.assertIsNotNone(first_alert({"classifier": StandInClassifier()}))
        self.assertIsNotNone(first_alert({"classifier": SafeClassifier(self.NaNModel())}))
        fusion_patched = "np.isfinite" in open(os.path.join(ROOT, "fusion.py")).read()
        if not fusion_patched:
            self.assertIsNone(first_alert({"classifier": self.NaNModel()}),
                              "unwrapped NaN model no longer suppresses alerts: update this test and P-1")


if __name__ == "__main__":
    unittest.main()
