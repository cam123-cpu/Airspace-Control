"""Sensor noise models: radar, RF direction finding (+ Remote ID), acoustic, PTZ camera.

Each sensor returns detections with a hidden `tid` (truth id, -1 for false
alarms). Fusion never reads `tid`; only the scorer does.
"""
import numpy as np
from scenario import SITE, DT, compass_az

D2R = np.pi / 180

# Axis-aligned building boxes (x0, x1, y0, y1, h) for radar / RF line-of-sight checks.
BOXES = [(min(p[0] for p in b["poly"]), max(p[0] for p in b["poly"]),
          min(p[1] for p in b["poly"]), max(p[1] for p in b["poly"]), b["h"]) for b in SITE["buildings"]]


def los_blocked(a, b, boxes=BOXES):
    """True if the straight segment a->b passes through any building box (slab test)."""
    d = b - a
    for x0, x1, y0, y1, h in boxes:
        t0, t1 = 0.0, 1.0
        ok = True
        for k, lo, hi in ((0, x0, x1), (1, y0, y1), (2, 0.0, h)):
            if abs(d[k]) < 1e-9:
                if a[k] < lo or a[k] > hi:
                    ok = False; break
            else:
                u0, u1 = (lo - a[k]) / d[k], (hi - a[k]) / d[k]
                if u0 > u1: u0, u1 = u1, u0
                t0, t1 = max(t0, u0), min(t1, u1)
                if t0 > t1:
                    ok = False; break
        if ok and t1 > 1e-6 and t0 < 1 - 1e-6:
            return True
    return False


def _in(win, t):
    return win is not None and t is not None and win[0] <= t <= win[1]


def _interp_path(path, t):
    """path: list of [t, x, y, z] waypoints -> position at t (None outside)."""
    P = np.asarray(path, float)
    if t < P[0, 0] or t > P[-1, 0]:
        return None
    return np.array([np.interp(t, P[:, 0], P[:, k]) for k in (1, 2, 3)])


def sph_to_enu(R, az, el):
    return np.array([R * np.cos(el) * np.sin(az), R * np.cos(el) * np.cos(az), R * np.sin(el)])


def sph_cov(R, az, el, sR, saz, sel):
    J = np.array([
        [np.cos(el) * np.sin(az), R * np.cos(el) * np.cos(az), -R * np.sin(el) * np.sin(az)],
        [np.cos(el) * np.cos(az), -R * np.cos(el) * np.sin(az), -R * np.sin(el) * np.cos(az)],
        [np.sin(el), 0.0, R * np.cos(el)]])
    return J @ np.diag([sR ** 2, saz ** 2, sel ** 2]) @ J.T


def ang_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


class StandInClassifier:
    """Placeholder for the trained camera model. Returns P(drone) for whatever is
    in the frame. Quality rises with pixels on target (px50 = px at which the
    model is half as good as its best). Swap for a real detector: see README."""

    def __init__(self, strength=4.0, noise=1.2, px50=10.0, flip_rate=0.02, hard_rate=0.05):
        self.strength, self.noise, self.px50, self.flip_rate = strength, noise, px50, flip_rate
        self.hard_rate = hard_rate   # fraction of objects the model is consistently wrong about
        self._hard = {}

    def score(self, rng, kind, px, has_payload=False, obj_id=None):
        if kind is None:  # empty frame: weak evidence against
            logit = -0.6 + rng.normal(0, 0.5)
        else:
            q = 1.0 / (1.0 + np.exp(-(px - self.px50) / 3.0))
            sign = 1.0 if kind == "drone" else -1.0
            if obj_id is not None and obj_id not in self._hard:
                self._hard[obj_id] = rng.random() < self.hard_rate
            hard = self._hard.get(obj_id, False)
            if hard or rng.random() < self.flip_rate:  # confident mistakes: folded-wing crow, backlit drone...
                sign = -sign * (0.7 if hard else 1.0)
            logit = sign * self.strength * q + rng.normal(0, self.noise)
        return 1.0 / (1.0 + np.exp(-logit))


class RenderedModelClassifier:
    """Closes the loop with a real detector: renders a synthetic PTZ frame of the
    object actually in view (at the simulated pixels-on-target) and runs your
    model on it. predict_fn(bgr_image) -> P(drone) in [0,1], e.g. max confidence
    of drone/drone_payload boxes from a YOLO model trained on generate_dataset.py output."""

    def __init__(self, predict_fn, mode=None):
        self.predict_fn, self.mode = predict_fn, mode

    def score(self, rng, kind, px, has_payload=False, obj_id=None):
        from render import render_frame
        cls = None if kind is None else {"drone": "drone", "bird": "bird"}.get(kind, "other")
        img, _, _ = render_frame(rng, self.mode, main_cls=cls, px=max(px, 3.0) if cls else None)
        return float(self.predict_fn(img))


REMOTE_RF_MAST = {"pos": [780.0, -760.0, 30.0], "note": "remote DF mast (e.g. water tower) ~1.1 km from the fence corners"}


def rf_nodes(params=None):
    """RF direction-finding nodes: the three fence corners, plus an optional remote mast.
    Fence-corner baselines (~400-500 m) triangulate a pilot 1+ km away poorly; a remote mast fixes the geometry."""
    nodes = list(SITE["sensors"]["rf"])
    if (params or {}).get("rf_remote_mast", False):
        nodes.append(REMOTE_RF_MAST)
    return nodes


class SensorSuite:
    def __init__(self, rng, wind, params=None):
        p = params or {}
        s = SITE["sensors"]
        self.rng = rng
        self.wind = float(np.linalg.norm(wind))
        self.radar_pos = np.array(s["radar"]["pos"])
        self.radar_rmax = s["radar"]["range_max"]
        self.rf_pos = [np.array(n["pos"]) for n in rf_nodes(p)]
        self.ac_pos = [np.array(n["pos"]) for n in s["acoustic"]]
        self.ptz_pos = np.array(s["ptz"]["pos"])
        # radar micro-Doppler (blade-flash test); its own RNG stream so enabling it leaves every other draw unchanged
        self.micro_doppler = p.get("micro_doppler", p.get("hardened", True))
        self.md_rng = np.random.default_rng([int(rng.bit_generator.state["state"]["state"]) % (1 << 63), 4242]) if self.micro_doppler else None  # derived, consumes no draw
        self.rf_bias = rng.normal(0, 1.5, len(self.rf_pos))  # per-node calibration / multipath bias (deg)
        self.clutter_rate = p.get("clutter_rate", 2.0)       # false radar plots per scan
        self.sR, self.saz, self.sel = 3.0, 0.5 * D2R, 1.0 * D2R
        self.rf_sigma = p.get("rf_sigma_deg", 3.5)
        from security.safe_classifier import SafeClassifier   # P-1: a broken model must fail safe, not open
        inner = p.get("classifier", StandInClassifier(p.get("classifier_strength", 4.0)))
        self.classifier = inner if isinstance(inner, SafeClassifier) else SafeClassifier(inner)
        # PTZ state
        self.ptz_az, self.ptz_el, self.ptz_fov = 0.0, 5.0 * D2R, 30.0 * D2R
        self.slew = 60.0 * D2R
        # video lock: once the object is acquired in a wide frame, the video tracker centres it and
        # zooms to ~10x the object size instead of staying as wide as the radar uncertainty
        self.video_lock = p.get("video_lock", p.get("hardened", True))
        self._lock = None   # (track id, object size m, range m)
        self.min_fov, self.max_fov = 1.5 * D2R, 30.0 * D2R
        # Red-team / environment hooks (all optional). See redteam/attacks.py.
        self.attack = p.get("attack") or {}
        self.rain = float(self.attack.get("weather", {}).get("rain", 0.0))   # 0..1
        self.fog = float(self.attack.get("weather", {}).get("fog", 0.0))     # 0..1
        self.clutter_rate += 6.0 * self.rain

    # ---------------------------------------------------------------- radar
    def pd(self, rcs, R):
        R50 = 1500.0 * (rcs / 0.01) ** 0.25
        return 0.95 / (1.0 + np.exp((R - R50) / (0.12 * R50)))

    def radar(self, present, t=None):
        rng, dets = self.rng, []
        for e, p, v in present:
            rel = p - self.radar_pos
            R = np.linalg.norm(rel)
            if R < 30 or R > self.radar_rmax:
                continue
            el = np.arcsin(rel[2] / R)
            pd = self.pd(e.rcs, R) * (1 - 0.3 * self.rain)
            if el < 0.3 * D2R and p[2] < 20 and R > 300:
                pd *= 0.15                      # terrain / tree-line masking of low targets
            if el < -6 * D2R or rng.random() > pd:
                continue
            if los_blocked(self.radar_pos, p):  # radar shadow behind buildings
                continue
            az = np.arctan2(rel[0], rel[1])
            md = None
            if self.micro_doppler:
                from microdoppler import body_snr_db, observe
                md = observe(e.kind == "drone", body_snr_db(e.rcs, R) - 3.0 * self.rain, self.md_rng)
            self._plot(dets, R + rng.normal(0, self.sR), az + rng.normal(0, self.saz),
                       el + rng.normal(0, self.sel), e.id, md)
        rate = self.clutter_rate
        surge = self.attack.get("radar_surge")
        if surge and _in(surge["win"], t):
            rate += surge["rate"]
        self.last_clutter_rate = rate
        for _ in range(rng.poisson(rate)):
            self._plot(dets, self.radar_rmax * np.sqrt(rng.uniform(0.01, 1)), rng.uniform(-np.pi, np.pi),
                       rng.uniform(-3, 10) * D2R, -1)
        return dets

    def _plot(self, dets, R, az, el, tid, md=None):
        dets.append({"pos": self.radar_pos + sph_to_enu(R, az, el),
                     "cov": sph_cov(R, az, el, self.sR, self.saz, self.sel), "tid": tid, "md": md})

    # ---------------------------------------------------------------- RF
    def _bearing(self, k, target):
        n = self.rf_pos[k]
        return (compass_az(*(np.asarray(target)[:2] - n[:2])) + self.rf_bias[k] + self.rng.normal(0, self.rf_sigma)) % 360

    def rf(self, present, t=None):
        """Call at 1 Hz. Returns (bearing reports, remote-id messages).
        Each report carries an `emitter` fingerprint (protocol / hop-pattern / MAC signature) as
        real DF receivers do. Remote ID messages carry `src_az`: bearings to the broadcast's
        actual transmitter, measured by the DF nodes (the key to spotting spoofed broadcasts)."""
        rng, out, rid = self.rng, [], []
        for e, p, v in present:
            if not e.is_drone:
                continue
            pilot = e.attrs["pilot"]
            sig = e.attrs.get("rf_sig", f"sig-{e.id}")
            if e.emitting:
                for k, n in enumerate(self.rf_pos):
                    R = np.linalg.norm(p[:2] - n[:2])
                    if R > 3000 or rng.random() > 0.9 * (1 - (R / 3000) ** 4):
                        continue
                    az_d = self._bearing(k, p)
                    az_c = None
                    if np.linalg.norm(pilot[:2] - n[:2]) < 3000 and rng.random() < 0.7:
                        az_c = self._bearing(k, pilot)
                    model = e.attrs["model"] if rng.random() < 0.8 else "unknown"
                    out.append({"node": k, "az_drone": az_d, "az_ctrl": az_c,
                                "model": model, "emitter": sig, "tid": e.id})
            if e.remote_id and min(np.linalg.norm(p[:2] - n[:2]) for n in self.rf_pos) < 1500 \
                    and rng.random() < 0.9:
                sp = e.attrs.get("rid_spoof") or {}
                claim = p + np.r_[sp.get("pos_offset", [0, 0, 0])] + rng.normal(0, 4, 3)
                op = (np.r_[sp["op"], 1.5] if "op" in sp else pilot) + np.r_[rng.normal(0, 8, 2), 0]
                rid.append({"pos": claim, "op": op, "serial": e.attrs.get("rid_serial", f"RID-{e.id:04d}"),
                            "src_az": self._src_bearings(p), "tid": e.id})
        # ---- attacker transmitters -------------------------------------------------
        for g in self.attack.get("rid_spoofs", []):        # ghost Remote ID broadcasts
            if not _in(g["win"], t) or rng.random() > 0.9:
                continue
            claim = _interp_path(g["path"], t)
            if claim is None:
                continue
            rid.append({"pos": claim + rng.normal(0, 4, 3), "op": np.r_[g["op"], 1.5] + np.r_[rng.normal(0, 8, 2), 0],
                        "serial": g["serial"], "src_az": self._src_bearings(np.asarray(g["tx"], float)), "tid": -2})
        for d in self.attack.get("rf_decoys", []):         # decoy / cloned emitters
            if not _in(d["win"], t):
                continue
            tx = np.asarray(d["pos"], float)
            for k, n in enumerate(self.rf_pos):
                if np.linalg.norm(tx[:2] - n[:2]) > 3000 or rng.random() > 0.85:
                    continue
                if d["mimic"] == "drone":   # drone-like downlink from a ground / mast transmitter
                    out.append({"node": k, "az_drone": self._bearing(k, tx), "az_ctrl": None,
                                "model": d.get("model", "quad-A (OcuSync-like)"), "emitter": d["sig"], "tid": -3})
                else:                       # controller-like uplink (possibly cloned signature)
                    out.append({"node": k, "az_drone": None, "az_ctrl": self._bearing(k, tx),
                                "model": d.get("model", "quad-A (OcuSync-like)"), "emitter": d["sig"], "tid": -3})
        return out, rid

    def _src_bearings(self, tx):
        return [(k, self._bearing(k, tx)) for k, n in enumerate(self.rf_pos)
                if np.linalg.norm(np.asarray(tx)[:2] - n[:2]) < 2500 and self.rng.random() < 0.85]

    # ---------------------------------------------------------------- acoustic
    def acoustic(self, present, t=None):
        """Call at 1 Hz. Rotor-noise detections with confidence 0..1."""
        rng, out = self.rng, []
        R00 = 250.0 * max(0.35, 1 - self.wind / 20.0) * (1 - 0.5 * self.rain)
        flood = self.attack.get("acoustic_flood")
        for k, n in enumerate(self.ac_pos):
            R0 = R00
            if flood and _in(flood["win"], t) and k in flood["nodes"]:
                R0 *= 0.15                      # loud broadband source masks rotor noise...
                for _ in range(rng.poisson(flood.get("rate", 3.0))):   # ...and trips the detector at random
                    out.append({"node": k, "conf": rng.uniform(0.3, 0.95), "az": rng.uniform(0, 360), "tid": -1})
            for e, p, v in present:
                R = np.linalg.norm(p - n)
                if e.is_drone and R < R0:
                    conf = np.clip(1.2 * (1 - R / R0) + rng.normal(0, 0.1), 0, 1)
                    if conf > 0.25:
                        out.append({"node": k, "conf": conf,
                                    "az": (compass_az(*(p[:2] - n[:2])) + rng.normal(0, 12)) % 360, "tid": e.id})
                elif e.kind == "truck" and R < 200 and rng.random() < 0.3:
                    out.append({"node": k, "conf": rng.uniform(0.25, 0.6),
                                "az": (compass_az(*(p[:2] - n[:2])) + rng.normal(0, 15)) % 360, "tid": e.id})
            if rng.random() < 0.002 * (self.wind / 5.0):
                out.append({"node": k, "conf": rng.uniform(0.25, 0.5), "az": rng.uniform(0, 360), "tid": -1})
        return out

    # ---------------------------------------------------------------- PTZ camera
    def ptz_step(self, cue, present, t=None):
        """cue = (track_id, est_pos, pos_sigma) or None. Slews toward the cue; once on
        target, grabs a frame and runs the classifier on whatever is nearest boresight."""
        if cue is None:
            return None
        tid_track, est, sig = cue
        if self.video_lock and self._lock is not None and self._lock[0] == tid_track:
            le = next((p for e, p, v in present if e.id == self._lock[3]), None)
            if le is not None:
                est = le                      # visual servo: the video tracker follows the object it locked on
            else:
                self._lock = None
        rel = est - self.ptz_pos
        R = np.linalg.norm(rel)
        az_d, el_d = np.arctan2(rel[0], rel[1]), np.arcsin(rel[2] / max(R, 1))
        daz = (az_d - self.ptz_az + np.pi) % (2 * np.pi) - np.pi
        step = self.slew * DT
        self.ptz_az += np.clip(daz, -step, step)
        self.ptz_el += np.clip(el_d - self.ptz_el, -step, step)
        self.ptz_fov = np.clip(max(0.5 / max(R, 1) * 16, 8 * sig / max(R, 1)), self.min_fov, self.max_fov)
        if self.video_lock and self._lock is not None and self._lock[0] == tid_track:
            self.ptz_fov = np.clip(self._lock[1] * 10 / max(self._lock[2], 1), self.min_fov, self.max_fov)
        settled = abs(daz) < self.ptz_fov / 4 and abs(el_d - self.ptz_el) < self.ptz_fov / 4
        obs = {"track": tid_track, "az": np.degrees(self.ptz_az) % 360, "el": np.degrees(self.ptz_el),
               "fov": np.degrees(self.ptz_fov), "settled": bool(settled)}
        if not settled:
            return obs
        dz = self.attack.get("camera_dazzle")
        if dz and _in(dz["win"], t) and self.rng.random() < dz.get("duty", 0.85):
            # laser dazzle / glare: sensor saturates. The frame-health metric (fraction of
            # saturated pixels) is something any real camera pipeline can measure.
            obs.update(kind=None, px=0.0, tid=-1, sat=float(self.rng.uniform(0.35, 1.0)))
            obs["score"] = float(self.classifier.score(self.rng, None, 0.0, obj_id=-1))
            return obs
        obs["sat"] = float(self.rng.uniform(0, 0.04))
        best, best_off = None, None
        hf, vf = self.ptz_fov / 2, self.ptz_fov * 512 / 640 / 2
        for e, p, v in present:
            if los_blocked(self.ptz_pos, p):   # camera shares the radar mast: same building shadow
                continue
            r = p - self.ptz_pos
            Re = np.linalg.norm(r)
            ea = np.arctan2(r[0], r[1]); ee = np.arcsin(r[2] / Re)
            da = ((ea - self.ptz_az + np.pi) % (2 * np.pi) - np.pi) * np.cos(ee)
            de = ee - self.ptz_el
            if abs(da) < hf and abs(de) < vf:
                off = np.hypot(da, de)
                if best_off is None or off < best_off:
                    best, best_off = (e, Re), off
        if best is None:
            self._lock = None
            obs.update(kind=None, px=0.0, tid=-1)
        else:
            e, Re = best
            px = float(e.size / Re / self.ptz_fov * 640) * (1 - 0.6 * max(self.fog, self.rain * 0.5))
            obs.update(kind=e.kind, px=px, tid=e.id)
            self._lock = (tid_track, e.size, Re, e.id)
        obs["score"] = float(self.classifier.score(self.rng, obs["kind"], obs["px"], obj_id=obs["tid"]))
        return obs
