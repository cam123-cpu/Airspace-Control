"""Multi-sensor fusion: radar tracking (CV Kalman filter + nearest-neighbour gating),
log-odds evidence combination, alert rule, drop-zone prediction, pilot triangulation.

`hardened=True` (default) adds the red-team defences documented in REDTEAM.md:
Remote ID verification against DF bearings of the broadcast and radar, emitter-level RF
association, controller-clone detection, sensor-health monitoring (radar clutter surge,
acoustic flooding, camera saturation), urgency-based camera scheduling, and kinematic
cues for tether / dive / formation attacks. `hardened=False` reproduces the v1 logic."""
import numpy as np
from scenario import SITE, DT, inside_fence, compass_az, FENCE_X, FENCE_Y
from sensors import ang_diff

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
PRIOR_LOGIT = np.log(0.01 / 0.99)
GATE = 16.0          # chi-square, 3 dof
ALERT_RADIUS = 1500  # only alert on tracks within this distance of site centre
ROAD_R = SITE["perimeter_road_radius"]


def _circ_mean(a_deg):
    a = np.radians(a_deg)
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360)


def _intersect(lines):
    """lines: [(node_xy, az_deg)] -> least-squares intersection (xy), rms perpendicular residual."""
    A, b = [], []
    for n, az in lines:
        a = np.radians(az); nrm = np.array([np.cos(a), -np.sin(a)])
        A.append(nrm); b.append(nrm @ n)
    A, b = np.array(A), np.array(b)
    est, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = float(np.sqrt(np.mean((A @ est - b) ** 2)))
    return est, res


def _two_clusters(az_list, min_sep=10.0, min_n=4, min_frac=0.2):
    """Split bearings into two clusters if clearly bimodal; returns [mean] or [meanA, meanB]."""
    m = _circ_mean(az_list)
    u = np.sort((np.asarray(az_list) - m + 180) % 360 - 180)
    if len(u) < 2 * min_n:
        return [m], [len(u)]
    gaps = np.diff(u); i = int(np.argmax(gaps))
    nA, nB = i + 1, len(u) - i - 1
    if gaps[i] > min_sep and min(nA, nB) >= min_n and min(nA, nB) / len(u) >= min_frac:
        return [(m + u[:nA].mean()) % 360, (m + u[nA:].mean()) % 360], [nA, nB]
    return [m], [len(u)]


def _F(dt):
    F = np.eye(6); F[0, 3] = F[1, 4] = F[2, 5] = dt
    return F


def _Q(dt, sa=3.0):
    q = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * sa ** 2
    Q = np.zeros((6, 6))
    for k in range(3):
        Q[np.ix_([k, k + 3], [k, k + 3])] = q
    return Q


H = np.hstack([np.eye(3), np.zeros((3, 3))])
F0, Q0 = _F(DT), _Q(DT)


class Track:
    _next = 1

    def __init__(self, t, z, R):
        self.id = Track._next; Track._next += 1
        self.x = np.r_[z, 0, 0, 0]
        self.P = np.zeros((6, 6)); self.P[:3, :3] = R; self.P[3:, 3:] = np.eye(3) * 100.0
        self.born, self.last = t, t
        self.hits = [True]; self.misses = 0
        self.status = "tentative"
        self.hist = []               # (t, x, y, z, vx, vy, vz)
        self.tid_counts = {}
        self.rf_hits = 0; self.rid = False; self.rid_ops = []; self.ctrl = []  # (node_pos, az)
        self.models = {}
        self.ac = 0.0
        self.cam = []                # list of llr
        self.cam_last = -1e9
        self.hover_time = 0.0; self.hover_seen = False; self.inside_seen = False
        self.hover_pts = []
        self.kin = 0.0
        self.parts = {}
        self.p = 1 / (1 + np.exp(-PRIOR_LOGIT))
        self.cross = {}              # threshold -> first time crossed (with alert gating)
        self.cross_pos = {}          # threshold -> track position when it crossed (scoring: was the alert in the right place?)
        self.drop_pred = None
        self.pilot_est = None; self.pilot_method = None
        self.pilot_cands = None; self.pilot_conf = None
        self.emitters = set(); self.rid_serials = set(); self.rid_op_rejected = False
        self.dive_seen = False; self.form_time = 0.0; self.ground = False
        self.md = 0.0                # running micro-Doppler evidence (blade flash present/absent)

    def predict(self):
        self.x = F0 @ self.x
        self.P = F0 @ self.P @ F0.T + Q0

    def update(self, z, R):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(6) - K @ H) @ self.P

    def maha(self, z, R):
        S = H @ self.P @ H.T + R
        d = z - H @ self.x
        return float(d @ np.linalg.solve(S, d))

    @property
    def pos(self):
        return self.x[:3]

    @property
    def sigma(self):
        return float(np.sqrt(np.trace(self.P[:3, :3]) / 3))

    def truth(self):
        if not self.tid_counts:
            return -1
        return max(self.tid_counts, key=self.tid_counts.get)


class Fusion:
    def __init__(self, params=None):
        p = params or {}
        s = SITE["sensors"]
        from sensors import rf_nodes
        self.rf_pos = [np.array(n["pos"]) for n in rf_nodes(p)]
        self.use_md = p.get("micro_doppler", p.get("hardened", True))
        self.ac_pos = [np.array(n["pos"]) for n in s["acoustic"]]
        self.ptz_pos = np.array(s["ptz"]["pos"])
        self.threshold = p.get("threshold", 0.9)
        self.tracks = []
        self.dead = []
        self.alerts = []
        self.cue = None; self.cue_t0 = 0.0
        self.last_looked = {}
        self.hardened = p.get("hardened", True)
        self.spoof_alerts = []      # {"t", "type", ...}: spoofing / decoy indicators (not a drone dispatch)
        self.sensor_alerts = []     # {"t", "sensor", "node", "state", "reason"}
        self.degraded = {}          # (sensor, node) -> since t
        self.emitter_db = {}        # rf signature -> {"ctrl": [(node, az, t)], "fixes": [(t, xy)], "last_assoc": t}
        self.rid_db = {}            # serial -> {"good", "bad", "flagged"}
        self.radar_counts = []
        self.ac_hist = {k: [] for k in range(len(self.ac_pos))}
        self.cam_health = []
        self._spoof_keys = set()

    # ------------------------------------------------------------ alerts helpers
    def _spoof(self, t, typ, key, **kw):
        if (typ, key) in self._spoof_keys:
            return
        self._spoof_keys.add((typ, key))
        self.spoof_alerts.append({"t": t, "type": typ, "key": key, **kw})

    def _health(self, t, sensor, node, bad, reason):
        k = (sensor, node)
        if bad and k not in self.degraded:
            self.degraded[k] = t
            self.sensor_alerts.append({"t": t, "sensor": sensor, "node": node, "state": "degraded", "reason": reason})
        elif not bad and k in self.degraded:
            del self.degraded[k]
            self.sensor_alerts.append({"t": t, "sensor": sensor, "node": node, "state": "recovered", "reason": reason})

    def degraded_any(self):
        return bool(self.degraded)

    # ------------------------------------------------------------ radar / tracking
    def predict(self):
        for tr in self.tracks:
            tr.predict()

    def _associate(self, t, meas):
        """meas: list of (z, R, tid). Greedy nearest-neighbour within gate."""
        pairs = []
        for j, mm in enumerate(meas):
            z, R = mm[0], mm[1]
            for i, tr in enumerate(self.tracks):
                if np.linalg.norm(tr.pos - z) > 200:
                    continue
                d = tr.maha(z, R)
                if d < GATE:
                    pairs.append((d, i, j))
        pairs.sort()
        used_t, used_m = set(), set()
        for d, i, j in pairs:
            if i in used_t or j in used_m:
                continue
            used_t.add(i); used_m.add(j)
            tr = self.tracks[i]; z, R, tid = meas[j][:3]
            tr.update(z, R); tr.last = t
            tr.tid_counts[tid] = tr.tid_counts.get(tid, 0) + 1
            md = meas[j][3] if len(meas[j]) > 3 else None
            if md is not None:
                tr.md = float(np.clip(0.8 * tr.md + 0.5 * md, -4.0, 4.0))
        return used_t, used_m

    def update_radar(self, t, dets):
        meas = [(d["pos"], d["cov"], d["tid"], d.get("md") if self.use_md else None) for d in dets]
        need = 3
        if self.hardened:
            self.radar_counts.append(len(dets))
            hist = self.radar_counts[-130:-10]
            if len(hist) >= 30:
                base = float(np.median(hist))
                lim = base + 4 * np.sqrt(base + 1) + 5
                recent = self.radar_counts[-6:]
                surge = sum(c > lim for c in recent) >= 4
                if surge or ("radar", 0) in self.degraded:
                    self._health(t, "radar", 0, surge or sum(c > lim for c in recent) >= 2, "plot-rate surge (chaff / rain / clutter injection)")
                    if surge:
                        self.radar_counts[-1] = int(base)  # keep the baseline clean
            if ("radar", 0) in self.degraded:
                need = 4
        used_t, used_m = self._associate(t, meas)
        for i, tr in enumerate(self.tracks):
            hit = i in used_t or tr.last == t
            tr.hits = (tr.hits + [hit])[-5:]
            tr.misses = 0 if hit else tr.misses + 1
            if tr.status == "tentative" and sum(tr.hits) >= need:
                tr.status = "confirmed"
        for j, mm in enumerate(meas):
            z, R, tid = mm[:3]
            if j not in used_m:
                tr = Track(t, z, R); tr.tid_counts[tid] = 1; tr.last = t
                if len(mm) > 3 and mm[3] is not None:
                    tr.md = float(np.clip(0.5 * mm[3], -4.0, 4.0))
                self.tracks.append(tr)
        keep = []
        for tr in self.tracks:
            dead = (tr.status == "tentative" and len(tr.hits) >= 4 and sum(tr.hits) < 2) or \
                   (tr.status == "tentative" and tr.misses >= 3) or tr.misses >= 10
            (self.dead if dead else keep).append(tr)
        self.tracks = keep

    # ------------------------------------------------------------ RF / Remote ID
    def update_rf(self, t, reports, rid):
        if self.hardened:
            return self._update_rf_hardened(t, reports, rid)
        for m in rid:
            z, R = m["pos"], np.eye(3) * 16.0
            best = min(self.tracks, key=lambda tr: np.linalg.norm(tr.pos - z), default=None)
            if best is None or np.linalg.norm(best.pos - z) > 40:
                best = Track(t, z, R); self.tracks.append(best)
            else:
                best.update(z, R)
            best.last = t; best.rid = True; best.status = "confirmed"
            best.tid_counts[m["tid"]] = best.tid_counts.get(m["tid"], 0) + 1
            best.rid_ops.append(m["op"][:2]); del best.rid_ops[:-200]
        for r in reports:
            n = self.rf_pos[r["node"]]
            if r.get("az_drone") is None:
                # v1 had no emitter IDs: a controller-only report is grouped with the strongest
                # RF track of the same protocol (naive signature grouping).
                if r["az_ctrl"] is None:
                    continue
                cands = [tr for tr in self.tracks if tr.rf_hits and r["model"] in tr.models]
                if cands:
                    b_ = max(cands, key=lambda tr: tr.rf_hits); b_.ctrl.append((n[:2], r["az_ctrl"])); del b_.ctrl[:-400]
                continue
            best, bd = None, 6.0
            for tr in self.tracks:
                if tr.status != "confirmed":
                    continue
                d = abs(ang_diff(compass_az(*(tr.pos[:2] - n[:2])), r["az_drone"]))
                if d < bd:
                    best, bd = tr, d
            if best is not None:
                best.rf_hits += 1
                best.models[r["model"]] = best.models.get(r["model"], 0) + 1
                if r["az_ctrl"] is not None:
                    best.ctrl.append((n[:2], r["az_ctrl"])); del best.ctrl[:-400]

    def _rid_check(self, t, m):
        """Is the Remote ID broadcast physically coming from where it claims to be?"""
        claim = m["pos"]
        st = self.rid_db.setdefault(m["serial"], {"good": 0, "bad": 0, "flagged": False, "tx": []})
        diffs = [abs(ang_diff(compass_az(*(claim[:2] - self.rf_pos[k][:2])), az)) for k, az in m.get("src_az", [])]
        consistent = None
        if len(diffs) >= 2:
            consistent = float(np.median(diffs)) < 8.0
            st["good" if consistent else "bad"] += 1
            st["tx"].append([(self.rf_pos[k][:2], az) for k, az in m["src_az"]])
        if not st["flagged"] and st["bad"] >= 3 and st["bad"] > st["good"]:
            st["flagged"] = True
            lines = [l for msg in st["tx"][-20:] for l in msg]
            tx, _ = _intersect(lines) if len(lines) >= 2 else (None, None)
            self._spoof(t, "rid_spoof", m["serial"], claim=[float(v) for v in claim[:2]],
                        transmitter_est=None if tx is None else [float(v) for v in tx],
                        note="Remote ID position does not match where the broadcast is coming from")
        return consistent, st["flagged"]

    def _rid_radar_check(self, t, m, radar_ok):
        """P-2: a Remote ID claim with no radar return for > 5 s, where radar should see it, is a spoof indicator."""
        from sensors import los_blocked
        st = self.rid_db.setdefault(m["serial"], {"good": 0, "bad": 0, "flagged": False, "tx": []})
        z = m["pos"]
        radar = np.array(SITE["sensors"]["radar"]["pos"])
        visible = z[2] > 20 and np.linalg.norm(z - radar) < 2500 and not los_blocked(radar, z)
        far = all(np.linalg.norm(tr.pos - z) > 100 for tr in self.tracks)
        if visible and far and not radar_ok:
            st.setdefault("no_radar_since", t)
            if t - st["no_radar_since"] > 5:
                self._spoof(t, "rid_no_radar_return", m["serial"], claim=[float(v) for v in z[:2]],
                            note="Remote ID claims a drone where radar sees nothing")
        else:
            st.pop("no_radar_since", None)

    def _update_rf_hardened(self, t, reports, rid):
        for m in rid:
            consistent, flagged = self._rid_check(t, m)
            if flagged or consistent is False:
                continue
            z, R = m["pos"], np.eye(3) * 16.0
            near = min(self.tracks, key=lambda tr: np.linalg.norm(tr.pos - z), default=None)
            radar_ok = (near is not None and np.linalg.norm(near.pos - z) <= 40 and near.status == "confirmed"
                        and sum(near.hits[-6:]) >= 3)
            self._rid_radar_check(t, m, radar_ok)
            if near is not None and np.linalg.norm(near.pos - z) <= 40:
                near.update(z, R)
            elif consistent:   # DF bearings show the broadcast really comes from the claimed spot (radar shadow?)
                near = Track(t, z, R); self.tracks.append(near)   # tentative until radar or DF keeps corroborating
            else:
                continue       # unverifiable and uncorroborated: RID alone never creates or confirms a track
            near.last = t; near.rid_serials.add(m["serial"])
            near.rid_claims = getattr(near, "rid_claims", 0) + 1
            if not (radar_ok or consistent):
                continue       # keep for the evidence log only: no RID evidence, no pilot dispatch
            near.rid = True
            if consistent:
                near.rid_verified = True
                if near.status != "confirmed" and getattr(near, "rid_claims", 0) >= 3:
                    near.status = "confirmed"      # three DF-verified claims from the same airborne spot
            near.tid_counts[m["tid"]] = near.tid_counts.get(m["tid"], 0) + 1
            near.rid_ops.append(m["op"][:2]); del near.rid_ops[:-200]
        by_em = {}
        for r in reports:
            by_em.setdefault(r.get("emitter", f"model:{r['model']}"), []).append(r)
        # never hand a drone's RF to an object that radar micro-Doppler or the camera already calls a bird
        airborne = [tr for tr in self.tracks if tr.status == "confirmed" and not tr.ground]
        conf = [tr for tr in airborne if tr.md > -1.0 and tr.parts.get("camera", 0.0) > -1.5]
        for sig, reps in by_em.items():
            em = self.emitter_db.setdefault(sig, {"ctrl": [], "fixes": [], "last_assoc": -1e9, "track": None})
            for r in reps:
                if r["az_ctrl"] is not None:
                    em["ctrl"].append((r["node"], r["az_ctrl"], t))
            em["ctrl"] = em["ctrl"][-400:]
            dr = [(r["node"], r["az_drone"]) for r in reps if r.get("az_drone") is not None]
            if not dr:
                if em["track"] is not None:
                    tr = next((x for x in self.tracks if x.id == em["track"]), None)
                    if tr is not None:
                        tr.emitters.add(sig)
                continue
            best, bres = None, None
            for tr in conf:
                res = max(abs(ang_diff(compass_az(*(tr.pos[:2] - self.rf_pos[k][:2])), az)) for k, az in dr)
                lim = 7.0 if len(dr) >= 2 else 4.0
                if res < lim and (bres is None or res < bres):
                    best, bres = tr, res
            if best is not None:
                best.rf_hits += 1.0 if len(dr) >= 2 else 0.5
                for r in reps:
                    best.models[r["model"]] = best.models.get(r["model"], 0) + 1
                best.emitters.add(sig); em["track"] = best.id; em["last_assoc"] = t
                best.emitter_counts = getattr(best, "emitter_counts", {})
                best.emitter_counts[sig] = best.emitter_counts.get(sig, 0) + 1
            elif any(max(abs(ang_diff(compass_az(*(tr.pos[:2] - self.rf_pos[k][:2])), az)) for k, az in dr)
                     < (7.0 if len(dr) >= 2 else 4.0) for tr in airborne):
                continue   # an airborne track is on these bearings (withheld as bird-like): not a decoy indicator
            elif len(dr) >= 2:
                em["fixes"] = (em["fixes"] + [(t, dr)])[-40:]
                rec = [d_ for tt, d_ in em["fixes"] if t - tt <= 20]
                if len(rec) >= 10 and t - em["last_assoc"] > 20:
                    by = {}
                    for d_ in rec:
                        for k, az in d_:
                            by.setdefault(k, []).append(az)
                    fix, _ = _intersect([(self.rf_pos[k][:2], _circ_mean(v)) for k, v in by.items()])
                    self._spoof(t, "rf_emitter_no_airborne_target", sig, fix=[float(v) for v in fix],
                                note="drone-type RF emitter with no matching airborne track: decoy transmitter or a drone still on the ground")

    # ------------------------------------------------------------ acoustic
    def update_acoustic(self, t, reports):
        for tr in self.tracks:
            tr.ac *= 0.97
        if self.hardened:
            for r in reports:
                node = self.ac_pos[r["node"]][:2]
                expl = any(np.linalg.norm(tr.pos[:2] - node) < 400 and
                           abs(ang_diff(compass_az(*(tr.pos[:2] - node)), r["az"])) <= 30 for tr in self.tracks)
                self.ac_hist[r["node"]].append((t, r["az"], expl))
            for k, h in self.ac_hist.items():
                h[:] = [x for x in h if t - x[0] <= 10]
                n = len(h)
                # A flooded node reports bearings that no radar track explains; real rotor sources line up
                # with a track at the time of the report (tentative tracks count).
                unexpl = sum(not x[2] for x in h)
                bad = n >= 12 and unexpl / n > 0.7
                if bad or (("acoustic", k) in self.degraded and n >= 4 and unexpl / max(n, 1) > 0.4):
                    self._health(t, "acoustic", k, True, "broadband noise flooding (detections from all directions)")
                else:
                    self._health(t, "acoustic", k, False, "noise floor back to normal")
            reports = [r for r in reports if ("acoustic", r["node"]) not in self.degraded]
            self._blind_sector(t, reports)
        for r in reports:
            n = self.ac_pos[r["node"]]
            for tr in self.tracks:
                if tr.status != "confirmed" or np.linalg.norm(tr.pos[:2] - n[:2]) > 350:
                    continue
                if abs(ang_diff(compass_az(*(tr.pos[:2] - n[:2])), r["az"])) < 25:
                    tr.ac = max(tr.ac, r["conf"])

    def _blind_sector(self, t, reports):
        """Two or more microphones hear rotors that no radar track explains, and their bearings cross
        inside the fence in a spot radar cannot see (building shadow): raise an advisory so a guard
        looks. Radar-independent coverage of the radar's blind sectors."""
        from sensors import los_blocked
        un = [r for r in reports if r["conf"] > 0.4 and not any(
            np.linalg.norm(tr.pos[:2] - self.ac_pos[r["node"]][:2]) < 400 and
            abs(ang_diff(compass_az(*(tr.pos[:2] - self.ac_pos[r["node"]][:2])), r["az"])) <= 30 for tr in self.tracks)]
        nodes = {r["node"] for r in un}
        hit = None
        if len(nodes) >= 2:
            fix, res = _intersect([(self.ac_pos[r["node"]][:2], r["az"]) for r in un])
            radar = np.array(SITE["sensors"]["radar"]["pos"])
            if inside_fence(fix[0], fix[1], 50) and los_blocked(radar, np.r_[fix, 10.0]) and res < 40:
                hit = fix
        self._bs = [(tt, f_) for tt, f_ in getattr(self, "_bs", []) if t - tt <= 6] + ([(t, hit)] if hit is not None else [])
        if len(self._bs) >= 4:
            pts = np.array([f_ for _, f_ in self._bs])
            if np.ptp(pts, axis=0).max() < 80:
                self._spoof(t, "blind_sector_acoustic_contact", tuple(np.round(pts.mean(0), -1)), fix=[float(v) for v in pts.mean(0)],
                            note="rotor noise from a spot radar and camera cannot see: send a guard to look")

    # ------------------------------------------------------------ camera
    def _urgency(self, tr):
        x = tr.x
        d_f = max(0.0, max(abs(x[0]) - FENCE_X, abs(x[1]) - FENCE_Y))
        r = np.linalg.norm(x[:2]); vr = -(x[:2] @ x[3:5]) / max(r, 1.0)
        ttf = d_f / max(vr, 0.5) if d_f > 0 else 0.0
        u = 1.0 + 2.0 / (1.0 + ttf / 30.0)
        if inside_fence(x[0], x[1], 60) or tr.hover_seen:
            u += 1.0
        return u

    def _choose_cue_hardened(self, t):
        cands = [tr for tr in self.tracks if tr.status == "confirmed" and not tr.ground
                 and np.linalg.norm(tr.pos - self.ptz_pos) <= 2000]
        urg = {tr.id: self._urgency(tr) for tr in cands}
        saturated = sum(u > 2.0 for u in urg.values()) >= 3
        if self.cue is not None:
            tr = next((x for x in cands if x.id == self.cue), None)
            dwell = 2.0 if saturated else 3.0
            resolved = tr is not None and len(tr.cam) >= 3 and (tr.p > 0.97 or tr.p < 0.03)
            if tr is not None and t - self.cue_t0 < dwell and not (resolved and t - self.cue_t0 >= 1.5):
                return (tr.id, tr.pos.copy(), tr.sigma)
            self.last_looked[self.cue] = t
            self.cue = None
        best, bs = None, 0.0
        for tr in cands:
            u = urg[tr.id]
            if t - self.last_looked.get(tr.id, -1e9) < (3.0 if u > 2.0 else 6.0):
                continue
            unc = 1.0 if 0.03 < tr.p < 0.97 else 0.3
            s = unc * u / (1 + 0.3 * len(tr.cam))
            s += 1e-3 * (t - self.last_looked.get(tr.id, t - 60))  # round-robin tie-break: longest unseen first
            if s > bs:
                best, bs = tr, s
        if best is None:
            return None
        self.cue, self.cue_t0 = best.id, t
        return (best.id, best.pos.copy(), best.sigma)

    def choose_cue(self, t):
        if self.hardened:
            return self._choose_cue_hardened(t)
        if self.cue is not None:
            tr = next((x for x in self.tracks if x.id == self.cue), None)
            if tr is not None and t - self.cue_t0 < 3.0:
                return (tr.id, tr.pos.copy(), tr.sigma)
            self.last_looked[self.cue] = t
            self.cue = None
        best, bs = None, 0.0
        for tr in self.tracks:
            if tr.status != "confirmed" or np.linalg.norm(tr.pos - self.ptz_pos) > 2000:
                continue
            if t - self.last_looked.get(tr.id, -1e9) < 6.0:
                continue
            s = (1.0 if 0.03 < tr.p < 0.97 else 0.3) / (1 + len(tr.cam))
            if inside_fence(*tr.pos[:2], 100):
                s *= 1.5
            if s > bs:
                best, bs = tr, s
        if best is None:
            return None
        self.cue, self.cue_t0 = best.id, t
        return (best.id, best.pos.copy(), best.sigma)

    def update_camera(self, t, obs):
        if not obs.get("settled") or "score" not in obs:
            return
        if self.hardened:
            sat = obs.get("sat", 0.0) > 0.3
            self.cam_health = (self.cam_health + [sat])[-6:]
            n_bad = sum(self.cam_health)
            if n_bad >= 3:
                self._health(t, "camera", 0, True, "image saturation (glare / laser dazzle)")
            elif n_bad == 0:
                self._health(t, "camera", 0, False, "image quality normal")
            if sat:
                return   # a blinded frame is no evidence either way
        tr = next((x for x in self.tracks if x.id == obs["track"]), None)
        if tr is None:
            return
        score = float(obs["score"])
        if not np.isfinite(score):          # P-1: a broken model must not poison evidence
            self.camera_faults = getattr(self, "camera_faults", 0) + 1
            return
        s = np.clip(score, 1e-4, 1 - 1e-4)
        tr.cam.append(float(np.clip(np.log(s / (1 - s)), -3, 3))); del tr.cam[:-200]
        tr.cam_last = t

    # ------------------------------------------------------------ evidence + alerts
    def _kinematics(self, tr):
        h = np.array(tr.hist[-20:])
        if len(h) < 6:
            return 0.0
        z = h[:, 3].mean(); spd = np.hypot(h[:, 4], h[:, 5])
        path = np.sum(np.linalg.norm(np.diff(h[:, 1:3], axis=0), axis=1))
        disp = np.linalg.norm(h[-1, 1:3] - h[0, 1:3])
        tort = path / max(disp, 1.0)
        k = 0.0
        if tr.hover_seen: k += 2.2
        if tr.inside_seen: k += 0.8
        if not self.hardened:
            if z < 6: k -= 2.5
            if z > 400: k -= 4.0
            if spd.max() > 28: k -= 3.0
            if tort > 1.8 and spd.mean() > 3: k -= 0.8
            if tort < 1.15 and 6 < spd.mean() < 20 and 15 < z < 200: k += 0.4
            return k
        x = tr.x
        r = np.linalg.norm(h[:, 1:3], axis=1).mean()
        near_road = abs(r - ROAD_R) < 30
        near_fence = inside_fence(x[0], x[1], 100)
        # road-gated, recomputed every second (revocable), and overridden by drone signatures: radar elevation
        # noise makes a low drone look like a car, but cars have no blade flash and the camera sees a rotorcraft
        tr.ground = near_road and z < 8 and spd.mean() > 4 and not (tr.md > 2.0 or tr.parts.get("camera", 0.0) > 2.0)
        if tr.ground: k -= 2.5           # vehicle on the perimeter road (map-aware, not just "low")
        elif z < 3: k -= 1.0
        if z > 400: k -= 4.0
        if spd.max() > 32: k -= 3.0
        if tort > 1.8 and spd.mean() > 3 and not near_fence: k -= 0.8   # erratic paths near the fence are not excused
        if tort < 1.15 and 6 < spd.mean() < 20 and 15 < z < 200: k += 0.4
        if tr.hover_seen and inside_fence(x[0], x[1], 80) and not inside_fence(x[0], x[1]): k += 0.6   # tether / line drop
        vz = h[-4:, 6]
        if vz.mean() < -5 and z < 100 and inside_fence(x[0], x[1], 150):
            tr.dive_seen = True
        if tr.dive_seen: k += 1.5
        return k

    def evidence(self, t, every_second):
        for tr in self.tracks:
            if tr.status != "confirmed":
                continue
            x = tr.x
            tr.hist.append((t, *x)); del tr.hist[:-600]
            # hover test on the smoothed track: net horizontal drift over the last 6 s
            w = np.array([h[1:4] for h in tr.hist[-12:]])
            hovering = False
            if len(w) == 12 and x[2] > 8 and tr.sigma < 25:
                drift = np.linalg.norm(w[-3:, :2].mean(0) - w[:3, :2].mean(0))
                spread = np.ptp(w[:, :2], axis=0).max()
                hovering = drift < 5.0 and spread < 12.0
            if hovering:
                tr.hover_time += DT
                tr.hover_pts.append(x[:2].copy()); del tr.hover_pts[:-600]
                if tr.hover_time >= 3: tr.hover_seen = True
            else:
                tr.hover_time = 0.0; tr.hover_pts = []
            if inside_fence(x[0], x[1]) and x[2] > 8:
                tr.inside_seen = True
            if every_second:
                tr.kin = self._kinematics(tr)
                self._drop(tr)
                self._pilot(tr, t)
            cam = 0.0
            if tr.cam:
                c = np.array(tr.cam[-6:])
                cam = float(np.clip(c.mean() * min(len(tr.cam), 3) / 3 * 1.2, -4, 4))
            tr.parts = {"kinematics": float(tr.kin), "rf": min(3.5, 0.6 * tr.rf_hits), "remote_id": 6.0 if tr.rid else 0.0,
                        "acoustic": float(2.0 * tr.ac), "camera": cam}
            if self.hardened and self.use_md:
                tr.parts["micro_doppler"] = float(np.clip(tr.md, -4.0, 4.0))
            if self.hardened:
                # no single sensor can veto: hover over/at the fence (radar) + rotor noise (acoustic) are two
                # independent physical rotorcraft cues, so a confident "not a drone" from the camera model
                # (model blind spot, camouflage, bird-shaped shell) is capped instead of overriding them
                if tr.hover_seen and tr.ac > 0.25 and inside_fence(x[0], x[1], 60):
                    tr.parts["camera"] = max(tr.parts["camera"], -1.0)
                if tr.rid and all(self.rid_db.get(s_, {}).get("flagged") for s_ in tr.rid_serials):
                    tr.parts["remote_id"] = 0.0
                # formation: moving in lock-step with an alerted drone (swarm members, escorts)
                buddy = False
                if np.linalg.norm(x[3:5]) > 4:
                    for o in self.tracks:
                        if o is tr or o.status != "confirmed" or o.p < self.threshold:
                            continue
                        if np.linalg.norm(o.pos - x[:3]) < 200 and np.linalg.norm(o.x[3:6] - x[3:6]) < 3:
                            buddy = True; break
                tr.form_time = tr.form_time + DT if buddy else 0.0
                tr.parts["formation"] = 1.5 if tr.form_time >= 5 else 0.0
                # degraded-mode policy: while a sensor is blinded or flooded, a hover over / at the
                # fence is treated as more suspicious (documented operator policy, not evidence)
                # (camera blinded) + hover at/over the fence + rotor noise heard there = treat as a drone
                tr.parts["degraded_policy"] = 2.5 if (("camera", 0) in self.degraded and tr.hover_seen
                                                      and inside_fence(x[0], x[1], 60) and tr.ac > 0.25) else 0.0
            L = PRIOR_LOGIT + sum(tr.parts.values())
            tr.p = float(1 / (1 + np.exp(-L)))
            if np.linalg.norm(x[:2]) < ALERT_RADIUS:
                for th in THRESHOLDS:
                    if th not in tr.cross and tr.p >= th:
                        tr.cross[th] = t; tr.cross_pos[th] = x[:3].copy()
                        if th == self.threshold:
                            self.alerts.append({"t": t, "track": tr.id, "p": tr.p, "pos": x[:3].copy()})

    def _drop(self, tr):
        x = tr.x
        if self.hardened and tr.hover_time >= 2 and inside_fence(x[0], x[1], 60) and not inside_fence(x[0], x[1], 5):
            hp = np.mean(tr.hover_pts, axis=0)   # hovering just outside: line / tether drop lands just inside
            tr.drop_pred = np.array([np.clip(hp[0], -FENCE_X + 8, FENCE_X - 8), np.clip(hp[1], -FENCE_Y + 8, FENCE_Y - 8)])
            return
        if tr.hover_time >= 2 and inside_fence(x[0], x[1], 20):
            tr.drop_pred = np.mean(tr.hover_pts, axis=0); return
        v = x[3:5]; sp = np.linalg.norm(v)
        if sp < 3 or x[2] < 8:
            return
        pts = x[:2][None, :] + v[None, :] * np.arange(0, 120, 1.0)[:, None]
        ins = (np.abs(pts[:, 0]) <= FENCE_X) & (np.abs(pts[:, 1]) <= FENCE_Y)
        if ins.any() and not inside_fence(x[0], x[1]):
            tr.drop_pred = pts[ins].mean(axis=0)
        elif inside_fence(x[0], x[1]) and tr.drop_pred is None:
            tr.drop_pred = x[:2].copy()

    def _df_fix(self, by_node, n_ctrl):
        """F-15: circular mean per node first, then intersect (raw least squares is biased toward the nodes).
        Returns (xy, radius_m, basis) or None."""
        if len(by_node) < 2:
            return None
        lines = [(self.rf_pos[k][:2], _circ_mean(v)) for k, v in by_node.items()]
        est, _ = _intersect(lines)
        if np.linalg.norm(est) >= 6000:
            return None
        se = [np.radians(max(1.0, np.std(((np.array(v) - _circ_mean(v) + 180) % 360) - 180))) / np.sqrt(len(v)) for v in by_node.values()]
        rng_ = np.mean([np.linalg.norm(est - self.rf_pos[k][:2]) for k in by_node])
        ang = [np.radians(a) for _, a in lines]
        gam = max(0.05, max(abs(np.sin(a - b)) for a in ang for b in ang))
        rad = float(np.clip(2 * rng_ * np.sqrt(np.sum(np.square(se))) / gam + 30, 30, 1500))
        return est, rad, f"RF direction finding, {len(by_node)} nodes, {n_ctrl} bearings"

    def _pilot_hardened(self, t, tr):
        ec = getattr(tr, "emitter_counts", {})
        sigs = [max(ec, key=ec.get)] if ec else list(tr.emitters)   # dominant emitter only: never mix pilots
        ctrl = [(k, az) for sig in sigs for k, az, tt in self.emitter_db.get(sig, {}).get("ctrl", [])]
        by_node = {}
        for k, az in ctrl:
            by_node.setdefault(k, []).append(az)
        if tr.rid_ops and not tr.rid_op_rejected and not all(self.rid_db.get(s_, {}).get("flagged") for s_ in tr.rid_serials):
            op = np.mean(tr.rid_ops, axis=0)
            if len(ctrl) >= 8 and len(by_node) >= 2:
                d = [abs(ang_diff(compass_az(*(op - self.rf_pos[k][:2])), az)) for k, az in ctrl]
                df = self._df_fix(by_node, len(ctrl)) if len(ctrl) >= 20 else None
                # angle test catches sideways misdirection; range test catches a claim pushed out (or in)
                # along the same line of bearing, which a short DF baseline cannot see in angle
                far = df is not None and np.linalg.norm(op - df[0]) > max(300.0, 2.5 * df[1])
                if np.median(d) > 9.0 or far:
                    tr.rid_op_rejected = True
                    self._spoof(t, "rid_operator_mismatch", tr.id, claimed_op=[float(v) for v in op],
                                note="controller signal does not come from the Remote ID operator location")
            if not tr.rid_op_rejected:
                verified = getattr(tr, "rid_verified", False) and len(ctrl) >= 8
                tr.pilot_est = op; tr.pilot_cands = None
                tr.pilot_method = "remote_id" if getattr(tr, "rid_verified", False) else "remote_id_unverified"
                tr.pilot_conf = {"radius_m": 15.0 if verified else None,
                                 "basis": "RID operator field, DF bearings agree" if verified else
                                          "RID operator field (unauthenticated; not yet cross-checked)"}
                return
        if len(ctrl) < 4 or len(by_node) < 2:
            return
        clus = {k: _two_clusters(v) for k, v in by_node.items()}
        nodes = sorted(clus)
        if sum(len(c[0]) == 2 for c in clus.values()) >= 2:
            best = None
            for mask in range(2 ** len(nodes)):
                la, lb = [], []
                for i, k in enumerate(nodes):
                    ms = clus[k][0]
                    a = ms[(mask >> i) & 1] if len(ms) == 2 else ms[0]
                    b = ms[1 - ((mask >> i) & 1)] if len(ms) == 2 else ms[0]
                    la.append((self.rf_pos[k][:2], a)); lb.append((self.rf_pos[k][:2], b))
                (pa, ra), (pb, rb) = _intersect(la), _intersect(lb)
                if best is None or ra + rb < best[0]:
                    best = (ra + rb, pa, pb)
            _, pa, pb = best
            if np.linalg.norm(pa - pb) > 150:
                tr.pilot_cands = [pa, pb]; tr.pilot_est = pa; tr.pilot_method = "rf_df_ambiguous"
                tr.pilot_conf = {"radius_m": None, "basis": "two candidate locations: send units to both"}
                self._spoof(t, "controller_signal_two_sources", tr.id, candidates=[[float(v) for v in pa], [float(v) for v in pb]],
                            note="controller-type signal arrives from two places: cloned / decoy transmitter")
                return
        df = self._df_fix(by_node, len(ctrl))
        if df is not None:
            tr.pilot_est, tr.pilot_method, tr.pilot_cands = df[0], "rf_df", None
            tr.pilot_conf = {"radius_m": df[1], "basis": df[2]}

    def _pilot(self, tr, t=None):
        if self.hardened:
            return self._pilot_hardened(t, tr)
        if tr.rid_ops:
            tr.pilot_est = np.mean(tr.rid_ops, axis=0); tr.pilot_method = "remote_id"; return
        if len(tr.ctrl) < 4 or len({tuple(n) for n, _ in tr.ctrl}) < 2:
            return
        A, b = [], []
        for n, az in tr.ctrl[-200:]:
            a = np.radians(az)
            nrm = np.array([np.cos(a), -np.sin(a)])  # normal to bearing line (dir = sin, cos)
            A.append(nrm); b.append(nrm @ n)
        est, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
        if np.linalg.norm(est) < 6000:
            tr.pilot_est = est; tr.pilot_method = "rf_df"

    def all_tracks(self):
        return self.tracks + self.dead
