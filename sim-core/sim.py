"""Episode runner: steps truth -> sensors -> fusion, scores the result, optionally records a replay."""
import numpy as np
from scenario import DT, SITE
from sensors import SensorSuite
from fusion import Fusion, THRESHOLDS


def _r(a, n=1):
    return [round(float(v), n) for v in a]


def run_episode(ep, seed, params=None, record=False):
    params = dict(params or {})
    if ep.get("attack"):
        params["attack"] = ep["attack"]
    rng = np.random.default_rng(seed)
    sens = SensorSuite(rng, ep["wind"], params)   # params["hardened"] also switches on the PTZ video lock
    fus = Fusion(params)
    ts, ents = ep["ts"], ep["entities"]
    drones = [e for e in ents if e.is_drone]
    drop_snap = {}
    alert_drop = {}
    n_alerts = 0
    frames = []
    for i, t in enumerate(ts):
        present = [(e, e.pos[i], e.vel[i]) for e in ents if not np.isnan(e.pos[i, 0])]
        fus.predict()
        rdets = sens.radar(present, t)
        fus.update_radar(t, rdets)
        rf, rid, ac = [], [], []
        sec = i % 2 == 0
        if sec:
            rf, rid = sens.rf(present, t)
            fus.update_rf(t, rf, rid)
            ac = sens.acoustic(present, t)
            fus.update_acoustic(t, ac)
        cue = fus.choose_cue(t)
        obs = sens.ptz_step(cue, present, t)
        if obs:
            fus.update_camera(t, obs)
        fus.evidence(t, sec)
        for a in fus.alerts[n_alerts:]:
            tr = next((x for x in fus.tracks if x.id == a["track"]), None)
            for d in drones:
                if tr is not None and tr.truth() == d.id and d.id not in alert_drop and d.attrs["drop_pos"] is not None:
                    alert_drop[d.id] = None if tr.drop_pred is None else \
                        float(np.linalg.norm(tr.drop_pred - d.attrs["drop_pos"][:2]))
        n_alerts = len(fus.alerts)
        for d in drones:
            if d.attrs["drop_t"] is not None and abs(t - d.attrs["drop_t"]) < DT / 2:
                own = [tr for tr in fus.tracks if tr.truth() == d.id and tr.drop_pred is not None]
                if own:
                    best = max(own, key=lambda tr: tr.p)
                    drop_snap[d.id] = float(np.linalg.norm(best.drop_pred - d.attrs["drop_pos"][:2]))
        if record:
            frames.append({
                "t": round(float(t), 1),
                "truth": [{"id": e.id, "kind": e.kind, "p": _r(p),
                           "payload": bool(e.is_drone and (e.attrs["drop_t"] is None or t < e.attrs["drop_t"]))}
                          for e, p, v in present],
                "radar": [_r(d["pos"]) for d in rdets],
                "rf": [{"node": r["node"], "az_drone": round(r["az_drone"], 1),
                        "az_ctrl": None if r["az_ctrl"] is None else round(r["az_ctrl"], 1)} for r in rf],
                "remote_id": [{"pos": _r(m["pos"]), "op": _r(m["op"][:2]), "serial": m["serial"]} for m in rid],
                "acoustic": [{"node": a["node"], "conf": round(float(a["conf"]), 2), "az": round(a["az"], 1)} for a in ac],
                "camera": None if obs is None else {k: (round(v, 2) if isinstance(v, float) else v)
                                                    for k, v in obs.items() if k != "tid"},
                "tracks": [{"id": tr.id, "pos": _r(tr.pos), "vel": _r(tr.x[3:]), "p": round(tr.p, 3),
                            "parts": {k: round(v, 2) for k, v in tr.parts.items()},
                            "hover": tr.hover_seen,
                            "drop_pred": None if tr.drop_pred is None else _r(tr.drop_pred),
                            "pilot_est": None if tr.pilot_est is None else _r(tr.pilot_est),
                            "pilot_method": tr.pilot_method}
                           for tr in fus.tracks if tr.status == "confirmed"],
                "alerts": [a["track"] for a in fus.alerts if abs(a["t"] - t) < DT / 2],
                "spoof_alerts": [a["type"] for a in fus.spoof_alerts if abs(a["t"] - t) < DT / 2],
                "sensor_alerts": [f'{a["sensor"]}{a["node"]}:{a["state"]}' for a in fus.sensor_alerts if abs(a["t"] - t) < DT / 2],
            })
    res = score(ep, fus, drop_snap, alert_drop)
    if record:
        res["frames"] = frames
    return res


def score(ep, fus, drop_snap, alert_drop=None):
    alert_drop = alert_drop or {}
    tracks = [tr for tr in fus.all_tracks() if tr.status == "confirmed"]
    ents = {e.id: e for e in ep["entities"]}
    out = {"kind": ep["kind"], "drones": [], "false_alarms": {th: 0 for th in THRESHOLDS},
           "fa_kinds": {th: {} for th in THRESHOLDS},
           "spoof_alerts": [{k: v for k, v in a.items() if k in ("t", "type")} for a in getattr(fus, "spoof_alerts", [])],
           "sensor_alerts": [{k: v for k, v in a.items() if k in ("t", "sensor", "node", "state")}
                             for a in getattr(fus, "sensor_alerts", [])]}
    t0 = ep["ts"][0]

    def placed_ok(tr, th, e):
        """An alert counts for drone e only if it was raised where e actually was (within 150 m)."""
        cp = getattr(tr, "cross_pos", {}).get(th)
        if cp is None:
            return True
        i = int(round((tr.cross[th] - t0) / DT))
        i = min(max(i, 0), len(e.pos) - 1)
        return not np.isnan(e.pos[i, 0]) and np.linalg.norm(e.pos[i] - cp) <= 150

    for tr in tracks:
        tid = tr.truth()
        e = ents.get(tid)
        if e is not None and e.is_drone:
            for th in tr.cross:
                if not placed_ok(tr, th, e):   # e.g. a spoofed Remote ID position: dispatch to the wrong place
                    out["false_alarms"][th] += 1
                    out["fa_kinds"][th]["misplaced_alert"] = out["fa_kinds"][th].get("misplaced_alert", 0) + 1
            continue
        kind = e.kind if e is not None else ("rid_ghost" if tid == -2 else "clutter")
        for th in tr.cross:
            out["false_alarms"][th] += 1
            out["fa_kinds"][th][kind] = out["fa_kinds"][th].get(kind, 0) + 1
    for e in ents.values():
        if not e.is_drone:
            continue
        own = [tr for tr in tracks if tr.truth() == e.id]
        first = {}
        for th in THRESHOLDS:
            times = [tr.cross[th] for tr in own if th in tr.cross and placed_ok(tr, th, e)]
            first[th] = min(times) if times else None
        pil = [tr for tr in own if tr.pilot_est is not None]
        perr, pmethod = None, None
        if pil:
            b = max(pil, key=lambda tr: (tr.pilot_method == "remote_id", len(tr.ctrl), tr.last))
            cands = b.pilot_cands if getattr(b, "pilot_cands", None) else [b.pilot_est]
            perr = float(min(np.linalg.norm(c - e.attrs["pilot"][:2]) for c in cands)); pmethod = b.pilot_method
        out["drones"].append({
            "emitting": e.emitting, "remote_id": e.remote_id, "takeoff_t": e.attrs["takeoff_t"],
            "decoy": bool(e.attrs.get("decoy", False)), "role": e.attrs.get("role", "payload"),
            "drop_t": e.attrs["drop_t"], "first_alert": first, "tracked": bool(own),
            "drop_err": drop_snap.get(e.id), "drop_err_at_alert": alert_drop.get(e.id), "pilot_err": perr, "pilot_method": pmethod,
            "dup_tracks": {th: max(0, sum(th in tr.cross and placed_ok(tr, th, e) for tr in own) - 1) for th in THRESHOLDS}})
    return out


def site_geometry():
    return SITE
