"""Regression tests for the red-team defences. Run: python -m pytest -q redteam/tests"""
import os, sys, unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from redteam.attacks import SCENARIOS                      # noqa: E402
from redteam.run_redteam import PairedStandIn               # noqa: E402
from sim import run_episode                                 # noqa: E402
from fusion import Fusion, _two_clusters, _intersect        # noqa: E402
from sensors import los_blocked                             # noqa: E402

TH = 0.9


def run(name, seed, hardened):
    ep = SCENARIOS[name](np.random.default_rng(seed))
    return run_episode(ep, seed + 7, {"hardened": hardened, "threshold": TH, "classifier": PairedStandIn(seed)})


class Geometry(unittest.TestCase):
    def test_building_shadow(self):
        radar = np.array([185.0, -140.0, 20.0])
        self.assertTrue(los_blocked(radar, np.array([-130.0, 135.0, 10.0])))     # low, behind Housing A
        self.assertFalse(los_blocked(radar, np.array([-130.0, 135.0, 40.0])))    # above the roofline

    def test_triangulation_unbiased_on_mean_lines(self):
        nodes = [np.array([-195.0, 145.0]), np.array([195.0, 145.0]), np.array([-195.0, -145.0])]
        tgt = np.array([-430.0, -1158.0])
        az = [np.degrees(np.arctan2(*(tgt - n))) % 360 for n in nodes]
        est, res = _intersect(list(zip(nodes, az)))
        self.assertLess(np.linalg.norm(est - tgt), 1.0)

    def test_two_cluster_split(self):
        rng = np.random.default_rng(0)
        a = list(rng.normal(30, 3, 20)) + list(rng.normal(60, 3, 20))
        means, ns = _two_clusters(a)
        self.assertEqual(len(means), 2)
        self.assertEqual(len(_two_clusters(list(rng.normal(30, 3, 40)))[0]), 1)


class Defences(unittest.TestCase):
    def test_ghost_rid_no_dispatch_and_flagged(self):
        r1, r2 = run("rid_ghost", 1001, False), run("rid_ghost", 1001, True)
        self.assertGreater(r1["false_alarms"][TH], 0)                 # v1 is fooled
        self.assertEqual(r2["false_alarms"][TH], 0)                   # hardened is not
        self.assertTrue(any(a["type"] in ("rid_spoof", "rid_no_radar_return") for a in r2["spoof_alerts"]))

    def test_operator_spoof_rejected(self):
        r = run("rid_op_misdirect", 1002, True)
        d = r["drones"][0]
        self.assertTrue(any(a["type"] == "rid_operator_mismatch" for a in r["spoof_alerts"]))
        self.assertNotEqual(d["pilot_method"], "remote_id")

    def test_camera_dazzle_flagged_and_drone_still_alerted(self):
        ok = 0
        for s in (1001, 1002, 1003, 1004):
            r = run("camera_dazzle", s, True)
            self.assertTrue(any(a["sensor"] == "camera" and a["state"] == "degraded" for a in r["sensor_alerts"]))
            d = r["drones"][0]
            ok += d["first_alert"][TH] is not None and d["first_alert"][TH] <= d["drop_t"]
        self.assertGreaterEqual(ok, 3)

    def test_acoustic_flood_flagged(self):
        r = run("acoustic_flood", 1003, True)
        self.assertTrue(any(a["sensor"] == "acoustic" and a["state"] == "degraded" for a in r["sensor_alerts"]))

    def test_radar_surge_flagged(self):
        r = run("radar_surge", 1004, True)
        self.assertTrue(any(a["sensor"] == "radar" and a["state"] == "degraded" for a in r["sensor_alerts"]))

    def test_clean_run_raises_no_indicators(self):
        for s in (1001, 1002, 1003):
            r = run("baseline_rf", s, True)
            self.assertEqual([a for a in r["sensor_alerts"] if a["state"] == "degraded"], [])
            self.assertEqual(r["spoof_alerts"], [])

    def test_legacy_mode_still_available(self):
        f = Fusion({"hardened": False})
        self.assertFalse(f.hardened)
        self.assertTrue(Fusion().hardened)


if __name__ == "__main__":
    unittest.main()
