"""Tests for the H2/H3 fusion additions: micro-Doppler, bird-gated RF, overridable ground class, remote RF mast."""
import os, sys, unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import microdoppler as md                      # noqa: E402
from fusion import Fusion, Track               # noqa: E402
from sensors import rf_nodes, SensorSuite      # noqa: E402
from scenario import SITE                      # noqa: E402


class MicroDoppler(unittest.TestCase):
    def test_snr_falls_with_range_and_matches_r50(self):
        self.assertAlmostEqual(md.body_snr_db(0.01, 1500.0), md.SNR_AT_R50_DB, places=6)
        self.assertGreater(md.body_snr_db(0.01, 400.0), md.body_snr_db(0.01, 1200.0))

    def test_reliability_bounds(self):
        self.assertLess(md.p_correct(5.0), 0.52)               # far / weak: uninformative
        self.assertGreater(md.p_correct(45.0), 0.94)           # close / strong: near the 0.95 ceiling
        self.assertLessEqual(md.p_correct(80.0), md.PC_MAX + 1e-9)

    def test_sign_follows_truth_on_average(self):
        rng = np.random.default_rng(0)
        d = [md.observe(True, 40.0, rng) for _ in range(400)]
        b = [md.observe(False, 40.0, rng) for _ in range(400)]
        self.assertGreater(np.mean(d), 2.0)
        self.assertLess(np.mean(b), -2.0)

    def test_weak_returns_give_no_evidence(self):
        self.assertIsNone(md.observe(True, 0.0, np.random.default_rng(1)))


class RfNodes(unittest.TestCase):
    def test_default_three_fence_corners(self):
        self.assertEqual(len(rf_nodes()), len(SITE["sensors"]["rf"]))

    def test_remote_mast_option(self):
        n = rf_nodes({"rf_remote_mast": True})
        self.assertEqual(len(n), len(SITE["sensors"]["rf"]) + 1)
        self.assertGreater(np.hypot(*n[-1]["pos"][:2]), 900)
        self.assertEqual(len(Fusion({"rf_remote_mast": True}).rf_pos), len(n))

    def test_site_dict_not_mutated(self):
        before = len(SITE["sensors"]["rf"]); rf_nodes({"rf_remote_mast": True}); rf_nodes({"rf_remote_mast": True})
        self.assertEqual(len(SITE["sensors"]["rf"]), before)


class Gating(unittest.TestCase):
    def _track(self, pos, md_val=0.0, cam=0.0):
        tr = Track(0.0, np.array(pos, float), np.eye(3)); tr.status = "confirmed"; tr.md = md_val; tr.parts = {"camera": cam}
        return tr

    def test_rf_not_attached_to_bird_like_track(self):
        f = Fusion({"hardened": True})
        bird = self._track([300.0, -600.0, 40.0], md_val=-3.0)
        f.tracks = [bird]
        az = [float(np.degrees(np.arctan2(*(bird.pos[:2] - p[:2])))) % 360 for p in f.rf_pos]
        reps = [{"node": k, "az_drone": az[k], "az_ctrl": None, "model": "ocusync", "emitter": "E"} for k in range(len(az))]
        f._update_rf_hardened(10.0, reps, [])
        self.assertEqual(bird.rf_hits, 0)
        self.assertFalse(f.spoof_alerts)          # withheld, but not mistaken for a decoy either

    def test_rf_attached_to_neutral_track(self):
        f = Fusion({"hardened": True})
        tr = self._track([300.0, -600.0, 40.0])
        f.tracks = [tr]
        az = [float(np.degrees(np.arctan2(*(tr.pos[:2] - p[:2])))) % 360 for p in f.rf_pos]
        reps = [{"node": k, "az_drone": az[k], "az_ctrl": None, "model": "ocusync", "emitter": "E"} for k in range(len(az))]
        f._update_rf_hardened(10.0, reps, [])
        self.assertGreater(tr.rf_hits, 0)

    def test_ground_class_overridden_by_blade_flash(self):
        f = Fusion({"hardened": True})
        R = SITE["perimeter_road_radius"]
        tr = self._track([R, 0.0, 4.0])
        tr.x[3] = 10.0
        tr.hist = [(i * 0.5, R, 0.0, 4.0, 10.0, 0.0, 0.0) for i in range(20)]
        f._kinematics(tr); self.assertTrue(tr.ground)
        tr.md = 3.0
        f._kinematics(tr); self.assertFalse(tr.ground)


class Determinism(unittest.TestCase):
    def test_md_rng_consumes_no_draw(self):
        a = np.random.default_rng(7); b = np.random.default_rng(7)
        SensorSuite(a, np.zeros(2), {"hardened": True}); SensorSuite(b, np.zeros(2), {"hardened": False})
        self.assertEqual(a.random(), b.random())


if __name__ == "__main__":
    unittest.main()
