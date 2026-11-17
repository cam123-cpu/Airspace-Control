"""Red-team scenario builders: adversarial drone runs and attacks on the sensors.

Every builder takes (rng) and returns an episode dict compatible with sim.run_episode:
    {"kind": "drone", "name", "ts", "entities", "wind", "attack"}
`attack` is read by SensorSuite (sensors.py): rid_spoofs, rf_decoys, acoustic_flood,
camera_dazzle, radar_surge, weather. Geometry is randomised per seed.

Scope: these model what an attacker does to *evade or confuse detection*. Nothing here
models jamming or defeating a drone; the product is detect-and-report only.
"""
import numpy as np
from scenario import DT, Entity, make_bird, make_truck, FENCE_X, FENCE_Y, SITE

G = 9.81
MODELS = ["quad-A (OcuSync-like)", "quad-B (WiFi FPV)", "quad-C (ELRS)"]


def _limit(v, vmax):
    n = np.linalg.norm(v)
    return v if n <= vmax or n == 0 else v * (vmax / n)


def polar(az_deg, dist, z=0.0):
    a = np.radians(az_deg)
    return np.array([dist * np.sin(a), dist * np.cos(a), z])


def fence_target(rng, margin=40):
    return np.array([rng.uniform(-FENCE_X + margin, FENCE_X - margin), rng.uniform(-FENCE_Y + margin, FENCE_Y - margin)])


def plan_drone(rng, eid, ts, pilot, t_start, legs, speed=12.0, cruise=80.0, emitting=False, remote_id=False,
               erratic=False, accel=4.0, **attrs):
    """Fly a list of legs, then return to the pilot and land.
    legs: ("goto", xyz[, speed]) | ("hover", seconds) | ("drop", xy_or_None) | ("loiter", centre_xy, radius, alt, seconds)
          | ("dive", z_target, vz)"""
    N = len(ts)
    pos = np.full((N, 3), np.nan)
    p = np.array([pilot[0], pilot[1], 0.0]); v = np.zeros(3)
    legs = list(legs) + [("goto", np.array([pilot[0], pilot[1], cruise]), speed), ("land",)]
    li, leg_t0 = 0, None
    drop_t = drop_pos = hover_t0 = None
    hdg_noise = 0.0
    i0 = int(np.searchsorted(ts, t_start))
    for i in range(i0, N):
        t = ts[i]
        if li >= len(legs):
            break
        leg = legs[li]
        if leg_t0 is None:
            leg_t0 = t
        kind = leg[0]
        vdes = np.zeros(3); done = False
        if kind == "goto":
            tgt = np.asarray(leg[1], float); spd = leg[2] if len(leg) > 2 else speed
            d = tgt - p; dh = np.linalg.norm(d[:2])
            vh = d[:2] / max(dh, 1e-6) * min(spd, 0.4 * dh + 0.5)
            if erratic:
                hdg_noise = np.clip(hdg_noise + rng.normal(0, 0.5) * np.sqrt(DT), -1.1, 1.1)
                c, s_ = np.cos(hdg_noise), np.sin(hdg_noise)
                vh = np.array([c * vh[0] + s_ * vh[1], -s_ * vh[0] + c * vh[1]])
            vz = np.clip(d[2], -3, 4) + (rng.normal(0, 1.2) if erratic else 0.0)
            vdes = np.r_[vh, vz]
            done = dh < 6 and abs(d[2]) < 4
        elif kind == "hover":
            if hover_t0 is None:
                hover_t0 = t
            vdes = np.r_[rng.normal(0, 0.3, 2) - v[:2] * 0.3, rng.normal(0, 0.2)]
            done = t - leg_t0 >= leg[1]
        elif kind == "drop":
            drop_t = t
            if leg[1] is not None:
                drop_pos = np.r_[leg[1], 0.0]
            else:   # released while moving: ballistic fall
                tf = np.sqrt(2 * max(p[2], 1.0) / G)
                drop_pos = np.r_[p[:2] + v[:2] * tf, 0.0]
            done = True
        elif kind == "loiter":
            c, r, alt, dur = np.asarray(leg[1], float), leg[2], leg[3], leg[4]
            ang = np.arctan2(p[0] - c[0], p[1] - c[1]) + speed / max(r, 1) * DT * 1.5
            tgt = np.r_[c + r * np.array([np.sin(ang), np.cos(ang)]), alt]
            vdes = _limit((tgt - p) / DT * 0.5, speed)
            done = t - leg_t0 >= dur
        elif kind == "dive":
            ztgt, vzd = leg[1], leg[2]
            vdes = np.r_[-v[:2] * 0.5, -vzd]
            done = p[2] <= ztgt + 2
        elif kind == "land":
            vdes = np.r_[-v[:2] * 0.5, -2.5]
            if p[2] < 0.8:
                break
        if done and kind != "land":
            li += 1; leg_t0 = None
            continue_ = True
        a_lim = (accel * 3 if kind == "dive" else accel) * DT
        v = v + _limit(vdes - v, a_lim) + np.r_[rng.normal(0, 0.15, 2), 0]
        p = p + v * DT
        p[2] = max(p[2], 0.0)
        pos[i] = p
    model = rng.choice(MODELS)
    return Entity(eid, "drone", pos, rcs=attrs.pop("rcs", rng.uniform(0.005, 0.05)), size=attrs.pop("size", rng.uniform(0.25, 0.9)),
                  emitting=emitting, remote_id=remote_id, pilot=np.array([pilot[0], pilot[1], 1.5]),
                  takeoff_t=ts[i0], drop_t=drop_t, drop_pos=drop_pos, hover_t0=hover_t0,
                  target=attrs.pop("target", None), model=model, **attrs)


def birds(rng, ts, n_site=3, n_far=3, first=100):
    out = []
    for k in range(n_site):
        out.append(make_bird(rng, first + k, ts, behaviour=rng.choice(["circle", "forage"])))
    for k in range(n_far):
        out.append(make_bird(rng, first + n_site + k, ts, behaviour="transit"))
    return out


def _ep(name, ents, wind, attack=None, T=480):
    return {"kind": "drone", "name": name, "ts": np.arange(0, T, DT), "entities": ents, "wind": wind, "attack": attack or {}}


def standard_legs(rng, target, cruise, hover_alt, hover_dur, drop_frac=0.7):
    return [("goto", np.r_[target, cruise]), ("goto", np.r_[target, hover_alt], 5.0),
            ("hover", hover_dur * drop_frac), ("drop", None if False else target), ("hover", hover_dur * (1 - drop_frac))]


def _standard(rng, ts, eid=1, emitting=True, remote_id=False, **kw):
    az = rng.uniform(0, 360); pilot = polar(az, rng.uniform(1000, 1500))
    tgt = fence_target(rng)
    cruise, hov = rng.uniform(60, 120), rng.uniform(20, 40)
    legs = standard_legs(rng, tgt, cruise, hov, rng.uniform(10, 40))
    return plan_drone(rng, eid, ts, pilot, rng.uniform(5, 20), legs, speed=rng.uniform(8, 15), cruise=cruise,
                      emitting=emitting, remote_id=remote_id, target=tgt, **kw)


# ----------------------------------------------------------------------------------------- scenarios
def baseline_rf(rng):
    ts = np.arange(0, 480, DT); w = rng.normal(0, 3, 2)
    return _ep("baseline_rf", [_standard(rng, ts, emitting=True)] + birds(rng, ts), w)


def baseline_dark(rng):
    ts = np.arange(0, 480, DT); w = rng.normal(0, 3, 2)
    return _ep("baseline_dark", [_standard(rng, ts, emitting=False)] + birds(rng, ts), w)


def rid_ghost(rng):
    """No drone at all: a ground transmitter broadcasts a fake Remote ID drone that 'flies' into the yard."""
    ts = np.arange(0, 300, DT); w = rng.normal(0, 3, 2)
    az = rng.uniform(0, 360); start = polar(az, 900, 70); tgt = fence_target(rng)
    path = [[10, *start], [80, tgt[0], tgt[1], 60], [85, tgt[0], tgt[1], 35], [125, tgt[0], tgt[1], 35],
            [200, *(polar(az + 30, 900, 70))]]
    tx = polar(az + rng.uniform(-60, 60), rng.uniform(500, 900), 1.5)
    attack = {"rid_spoofs": [{"win": (10, 200), "path": path, "op": polar(az + 120, 1100)[:2], "serial": "RID-SPOOF-1", "tx": tx}]}
    return _ep("rid_ghost", birds(rng, ts, 4, 4), w, attack, T=300)


def rid_op_misdirect(rng):
    """Real drone with Remote ID that broadcasts a false operator location (sends police to the wrong place)."""
    ts = np.arange(0, 480, DT); w = rng.normal(0, 3, 2)
    d = _standard(rng, ts, emitting=True, remote_id=True)
    fake = d.attrs["pilot"][:2] + polar(rng.uniform(0, 360), rng.uniform(600, 900))[:2]
    d.attrs["rid_spoof"] = {"op": fake}
    return _ep("rid_op_misdirect", [d] + birds(rng, ts), w)


def rid_pos_offset(rng):
    """Real drone whose Remote ID reports a position 200-300 m away plus a fake operator."""
    ts = np.arange(0, 480, DT); w = rng.normal(0, 3, 2)
    d = _standard(rng, ts, emitting=True, remote_id=True)
    off = polar(rng.uniform(0, 360), rng.uniform(200, 300))
    d.attrs["rid_spoof"] = {"pos_offset": [off[0], off[1], 0.0], "op": d.attrs["pilot"][:2] + polar(rng.uniform(0, 360), 800)[:2]}
    return _ep("rid_pos_offset", [d] + birds(rng, ts), w)


def rf_decoy_downlink(rng):
    """Dark (RF-silent) drone plus a ground transmitter mimicking a drone video downlink elsewhere."""
    ts = np.arange(0, 480, DT); w = rng.normal(0, 3, 2)
    d = _standard(rng, ts, emitting=False)
    dec = polar(rng.uniform(0, 360), rng.uniform(300, 700), 3.0)
    attack = {"rf_decoys": [{"win": (0, 480), "pos": dec, "mimic": "drone", "sig": "sig-decoy-1"}]}
    bs = birds(rng, ts, 3, 3)
    for k in range(4):   # birds working the area around the decoy (they share its bearings)
        bs.append(make_bird(rng, 200 + k, ts, start=np.r_[dec[:2] * rng.uniform(0.3, 0.9) + rng.normal(0, 40, 2), rng.uniform(20, 80)],
                            behaviour="circle"))
    return _ep("rf_decoy_downlink", [d] + bs, w, attack)


def ctrl_clone(rng):
    """RF drone; a second transmitter clones the controller signature from another road to misdirect the pilot search."""
    ts = np.arange(0, 480, DT); w = rng.normal(0, 3, 2)
    d = _standard(rng, ts, emitting=True)
    dec = d.attrs["pilot"] + polar(rng.uniform(0, 360), rng.uniform(500, 1000), 0)
    attack = {"rf_decoys": [{"win": (0, 480), "pos": dec, "mimic": "ctrl", "sig": f"sig-{d.id}", "model": d.attrs["model"]}]}
    return _ep("ctrl_clone", [d] + birds(rng, ts), w, attack)


def decoy_saturation(rng):
    """Three RF-emitting decoy drones loiter outside the fence to soak up camera time; the real
    (dark) payload drone comes in from the opposite side 20-40 s later."""
    ts = np.arange(0, 480, DT); w = rng.normal(0, 3, 2)
    base = rng.uniform(0, 360); ents = []
    for k in range(3):
        az = base + rng.uniform(-50, 50)
        pilot = polar(az, rng.uniform(900, 1300))
        c = polar(az + rng.uniform(-20, 20), rng.uniform(260, 420))[:2]
        legs = [("goto", np.r_[c, 60]), ("loiter", c, rng.uniform(40, 90), rng.uniform(40, 80), 220)]
        ents.append(plan_drone(rng, 10 + k, ts, pilot, rng.uniform(0, 10), legs, speed=10, cruise=70, emitting=True,
                               decoy=True, role="decoy"))
    az = base + 180 + rng.uniform(-40, 40); pilot = polar(az, rng.uniform(1000, 1400)); tgt = fence_target(rng)
    legs = standard_legs(rng, tgt, rng.uniform(60, 100), rng.uniform(20, 35), rng.uniform(10, 25))
    ents.append(plan_drone(rng, 1, ts, pilot, rng.uniform(25, 45), legs, speed=rng.uniform(10, 15), cruise=80, emitting=False, target=tgt))
    return _ep("decoy_saturation", ents + birds(rng, ts), w)


def swarm(rng):
    """Four dark payload drones from one launch area, staggered by 0-15 s, different drop points."""
    ts = np.arange(0, 480, DT); w = rng.normal(0, 3, 2)
    az = rng.uniform(0, 360); base = polar(az, rng.uniform(1000, 1400)); ents = []
    for k in range(4):
        pilot = base + np.r_[rng.normal(0, 30, 2), 0]
        tgt = fence_target(rng)
        legs = standard_legs(rng, tgt, rng.uniform(60, 90), rng.uniform(20, 35), rng.uniform(8, 20))
        ents.append(plan_drone(rng, 1 + k, ts, pilot, 5 + rng.uniform(0, 15), legs, speed=rng.uniform(11, 14), cruise=75,
                               emitting=False, target=tgt))
    return _ep("swarm", ents + birds(rng, ts), w)


def terrain_masked(rng):
    """Dark drone flies 8-12 m above ground to the fence (below the radar horizon / tree line), pops up to drop."""
    ts = np.arange(0, 480, DT); w = rng.normal(0, 3, 2)
    az = rng.uniform(0, 360); pilot = polar(az, rng.uniform(900, 1300)); tgt = fence_target(rng)
    edge = np.r_[np.clip(polar(az, 1)[:2] * 1e4, [-FENCE_X - 30, -FENCE_Y - 30], [FENCE_X + 30, FENCE_Y + 30]), 10]
    low = rng.uniform(8, 12)
    legs = [("goto", np.r_[edge[:2], low], 12), ("goto", np.r_[tgt, 25], 6), ("hover", rng.uniform(5, 10)), ("drop", tgt),
            ("hover", 3), ("goto", np.r_[edge[:2], low], 12)]
    return _ep("terrain_masked", [plan_drone(rng, 1, ts, pilot, rng.uniform(5, 20), legs, speed=12, cruise=low, emitting=False, target=tgt)]
               + birds(rng, ts), w)


def building_shadow(rng):
    """Dark drone approaches low from the NW, using Housing A as a radar shadow, drops between the building and the fence."""
    ts = np.arange(0, 480, DT); w = rng.normal(0, 3, 2)
    pilot = polar(rng.uniform(300, 330), rng.uniform(900, 1200))
    tgt = np.array([rng.uniform(-160, -100), rng.uniform(128, 140)])
    alt = rng.uniform(8, 11)
    legs = [("goto", np.r_[-420, 420, 30], 12), ("goto", np.r_[-230, 200, alt], 8), ("goto", np.r_[tgt, alt], 4),
            ("hover", rng.uniform(8, 15)), ("drop", tgt), ("hover", 3), ("goto", np.r_[-420, 420, alt], 10)]
    return _ep("building_shadow", [plan_drone(rng, 1, ts, pilot, rng.uniform(5, 20), legs, speed=12, cruise=30, emitting=False, target=tgt)]
               + birds(rng, ts), w)


def bird_mimicry(rng):
    """Dark drone flies slow (5-7 m/s), wanders like a bird, bobs in altitude, never hovers: releases while moving."""
    ts = np.arange(0, 480, DT); w = rng.normal(0, 3, 2)
    az = rng.uniform(0, 360); pilot = polar(az, rng.uniform(900, 1200)); tgt = fence_target(rng, 60)
    alt = rng.uniform(30, 60)
    legs = [("goto", np.r_[tgt, alt], rng.uniform(5, 7)), ("drop", None), ("goto", np.r_[polar(az + 60, 600)[:2], alt], 6)]
    return _ep("bird_mimicry", [plan_drone(rng, 1, ts, pilot, rng.uniform(5, 20), legs, speed=6, cruise=alt, emitting=False,
                                           erratic=True, target=tgt)] + birds(rng, ts, 4, 3), w)


def dive_drop(rng):
    """Dark drone transits high and fast (150-200 m, 20 m/s), dives at ~10 m/s to 15 m, drops after 2 s, climbs out."""
    ts = np.arange(0, 480, DT); w = rng.normal(0, 3, 2)
    az = rng.uniform(0, 360); pilot = polar(az, rng.uniform(1100, 1500)); tgt = fence_target(rng)
    hi = rng.uniform(150, 200)
    legs = [("goto", np.r_[tgt, hi], 20), ("dive", 15, 10), ("hover", 2), ("drop", tgt), ("goto", np.r_[tgt, 120], 20)]
    return _ep("dive_drop", [plan_drone(rng, 1, ts, pilot, rng.uniform(5, 20), legs, speed=20, cruise=hi, emitting=False,
                                       accel=6, target=tgt)] + birds(rng, ts), w)


def tether_drop(rng):
    """RF drone never crosses the fence: hovers 25-45 m outside and lowers the package on a line over it."""
    ts = np.arange(0, 480, DT); w = rng.normal(0, 3, 2)
    side = rng.integers(4)
    if side == 0: hp, dp = [rng.uniform(-150, 150), FENCE_Y + rng.uniform(25, 45)], None
    elif side == 1: hp = [rng.uniform(-150, 150), -FENCE_Y - rng.uniform(25, 45)]
    elif side == 2: hp = [FENCE_X + rng.uniform(25, 45), rng.uniform(-100, 100)]
    else: hp = [-FENCE_X - rng.uniform(25, 45), rng.uniform(-100, 100)]
    hp = np.array(hp)
    dp = np.array([np.clip(hp[0], -FENCE_X + 8, FENCE_X - 8), np.clip(hp[1], -FENCE_Y + 8, FENCE_Y - 8)])
    inward = -np.sign(hp) * (np.abs(hp) > [FENCE_X, FENCE_Y])
    dp = dp + inward * rng.uniform(0, 7)
    az = np.degrees(np.arctan2(hp[0], hp[1])); pilot = polar(az + rng.uniform(-30, 30), rng.uniform(900, 1300))
    alt = rng.uniform(35, 45)
    legs = [("goto", np.r_[hp, 70]), ("goto", np.r_[hp, alt], 4), ("hover", rng.uniform(15, 30)), ("drop", dp), ("hover", 5)]
    return _ep("tether_drop", [plan_drone(rng, 1, ts, pilot, rng.uniform(5, 20), legs, speed=12, cruise=70, emitting=True, target=dp)]
               + birds(rng, ts), w)


def acoustic_flood(rng):
    ep = baseline_dark(rng); ep["name"] = "acoustic_flood"
    nodes = sorted(rng.choice(4, size=rng.integers(2, 5), replace=False).tolist())
    ep["attack"] = {"acoustic_flood": {"win": (0, 480), "nodes": nodes, "rate": 3.0}}
    return ep


def camera_dazzle(rng):
    ep = baseline_dark(rng); ep["name"] = "camera_dazzle"
    t0 = ep["entities"][0].attrs["takeoff_t"] + 30
    ep["attack"] = {"camera_dazzle": {"win": (t0, 480), "duty": 0.85}}
    return ep


def radar_surge(rng):
    ep = baseline_dark(rng); ep["name"] = "radar_surge"
    t0 = ep["entities"][0].attrs["takeoff_t"] + 20
    ep["attack"] = {"radar_surge": {"win": (t0, t0 + 200), "rate": 40.0}}
    return ep


def storm(rng):
    ts = np.arange(0, 480, DT)
    w = polar(rng.uniform(0, 360), rng.uniform(10, 14))[:2]
    d = _standard(rng, ts, emitting=True)
    return _ep("storm", [d] + birds(rng, ts, 2, 2), w, {"weather": {"rain": 0.8, "fog": 0.4}})


SCENARIOS = {f.__name__: f for f in [baseline_rf, baseline_dark, rid_ghost, rid_op_misdirect, rid_pos_offset,
                                     rf_decoy_downlink, ctrl_clone, decoy_saturation, swarm, terrain_masked,
                                     building_shadow, bird_mimicry, dive_drop, tether_drop, acoustic_flood,
                                     camera_dazzle, radar_surge, storm]}

# which defensive indicator each attack should trip (scored as "attack flagged")
EXPECTED_FLAG = {
    "rid_ghost": ("spoof", "rid_spoof"), "rid_pos_offset": ("spoof", "rid_spoof"),
    "rid_op_misdirect": ("spoof", "rid_operator_mismatch"), "rf_decoy_downlink": ("spoof", "rf_emitter_no_airborne_target"),
    "ctrl_clone": ("spoof", "controller_signal_two_sources"), "acoustic_flood": ("sensor", "acoustic"),
    "camera_dazzle": ("sensor", "camera"), "radar_surge": ("sensor", "radar"),
    "building_shadow": ("spoof", "blind_sector_acoustic_contact"),
}
