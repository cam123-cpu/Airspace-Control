"""World model: site geometry, sensor placement, and entity trajectory generators.

Coordinates are local ENU metres (x = east, y = north, z = up), origin at the
centre of the prison yard. Bearings/azimuths are compass style: degrees
clockwise from north, i.e. az = atan2(dx, dy).
"""
import numpy as np

DT = 0.5  # simulation step (s) == radar scan period (2 Hz)

SITE = {
    "name": "Generic state prison (synthetic)",
    "fence": [[-200, -150], [200, -150], [200, 150], [-200, 150]],
    "yard": [[-60, -40], [60, -40], [60, 30], [-60, 30]],
    "buildings": [
        {"name": "Housing A", "poly": [[-170, 45], [-90, 45], [-90, 120], [-170, 120]], "h": 12},
        {"name": "Housing B", "poly": [[90, 45], [170, 45], [170, 120], [90, 120]], "h": 12},
        {"name": "Admin", "poly": [[-110, -130], [-30, -130], [-30, -85], [-110, -85]], "h": 10},
        {"name": "Kitchen", "poly": [[30, -130], [100, -130], [100, -85], [30, -85]], "h": 8},
    ],
    "perimeter_road_radius": 260.0,
    "sensors": {
        "radar": {"pos": [185.0, -140.0, 20.0], "range_max": 3000.0},
        "ptz": {"pos": [185.0, -140.0, 20.0]},
        # RF direction-finding nodes at three fence corners (baseline ~400-500 m)
        "rf": [{"pos": [-195.0, 145.0, 10.0]}, {"pos": [195.0, 145.0, 10.0]},
               {"pos": [-195.0, -145.0, 10.0]}],
        "acoustic": [{"pos": [0.0, 150.0, 3.0]}, {"pos": [200.0, 0.0, 3.0]},
                     {"pos": [0.0, -150.0, 3.0]}, {"pos": [-200.0, 0.0, 3.0]}],
    },
}

FENCE_X, FENCE_Y = 200.0, 150.0


def inside_fence(x, y, margin=0.0):
    return (abs(x) <= FENCE_X + margin) and (abs(y) <= FENCE_Y + margin)


def compass_az(dx, dy):
    return np.degrees(np.arctan2(dx, dy)) % 360.0


class Entity:
    """A thing that flies (or drives). pos is (N,3) on the episode time grid, NaN when absent."""

    def __init__(self, eid, kind, pos, **attrs):
        self.id = eid
        self.kind = kind          # drone | bird | aircraft | bag | truck
        self.pos = pos
        self.vel = np.vstack([np.diff(pos, axis=0) / DT, np.full((1, 3), np.nan)])
        self.vel[-1] = self.vel[-2]
        self.rcs = attrs.pop("rcs", 0.01)
        self.size = attrs.pop("size", 0.5)          # characteristic span (m) for camera
        self.emitting = attrs.pop("emitting", False)  # RF control/video link
        self.remote_id = attrs.pop("remote_id", False)
        self.attrs = attrs

    @property
    def is_drone(self):
        return self.kind == "drone"


def _limit(v, vmax):
    n = np.linalg.norm(v)
    return v if n <= vmax or n == 0 else v * (vmax / n)


def make_drone(rng, eid, ts, t_start, wind, emitting=True, remote_id=False):
    N = len(ts)
    ang = rng.uniform(0, 2 * np.pi)
    dist = rng.uniform(1000, 1500)
    pilot = np.array([dist * np.sin(ang), dist * np.cos(ang), 1.5])
    target = np.array([rng.uniform(-150, 150), rng.uniform(-110, 110)])
    cruise = rng.uniform(60, 120)
    speed = rng.uniform(8, 15)
    hover_alt = rng.uniform(20, 40)
    hover_dur = rng.uniform(10, 40)
    drop_frac = rng.uniform(0.5, 0.95)
    exit_ang = ang + np.radians(rng.uniform(-40, 40))
    exit_d = rng.uniform(900, 1500)
    exit_pt = np.array([exit_d * np.sin(exit_ang), exit_d * np.cos(exit_ang)])

    pos = np.full((N, 3), np.nan)
    p = pilot.copy(); p[2] = 0.0
    v = np.zeros(3)
    phase = "ingress"
    hover_t0 = drop_t = None
    drop_pos = None
    i0 = int(np.searchsorted(ts, t_start))
    for i in range(i0, N):
        t = ts[i]
        dxy = target - p[:2]
        dh = np.linalg.norm(dxy)
        if phase == "ingress":
            vh = dxy / max(dh, 1e-6) * speed
            vz = np.clip(cruise - p[2], -3, 4)
            if dh < 90:
                phase = "descend"
        elif phase == "descend":
            vh = dxy / max(dh, 1e-6) * min(speed, 0.25 * dh + 0.3)
            vz = np.clip(hover_alt - p[2], -3, 3)
            if dh < 3 and abs(p[2] - hover_alt) < 2.5:
                phase, hover_t0 = "hover", t
        elif phase == "hover":
            vh = dxy * 0.5 + rng.normal(0, 0.3, 2)
            vz = (hover_alt - p[2]) * 0.5 + rng.normal(0, 0.2)
            if drop_t is None and t - hover_t0 >= drop_frac * hover_dur:
                drop_t, drop_pos = t, np.array([p[0], p[1], 0.0])
            if t - hover_t0 >= hover_dur:
                phase = "exit"
        else:  # exit: fly back out, then descend and land near the pilot / exit point
            dx = exit_pt - p[:2]
            de = np.linalg.norm(dx)
            vh = dx / max(de, 1e-6) * min(speed, 0.3 * de + 0.2)
            vz = np.clip(cruise - p[2], -3, 4) if de > 60 else -2.5
            if de < 60 and p[2] < 1.0:
                break  # landed: entity disappears
        vdes = np.array([vh[0], vh[1], vz])
        v = v + _limit(vdes - v, 4.0 * DT) + np.r_[rng.normal(0, 0.15, 2), 0]
        p = p + (v + 0.08 * np.r_[wind, 0]) * DT
        p[2] = max(p[2], 0.0)
        pos[i] = p
    takeoff_t = ts[i0]
    return Entity(eid, "drone", pos, rcs=rng.uniform(0.005, 0.05), size=rng.uniform(0.25, 0.9),
                  emitting=emitting, remote_id=remote_id, pilot=pilot, takeoff_t=takeoff_t,
                  drop_t=drop_t, drop_pos=drop_pos, hover_t0=hover_t0, target=target,
                  model=rng.choice(["quad-A (OcuSync-like)", "quad-B (WiFi FPV)", "quad-C (ELRS)"]))


def make_bird(rng, eid, ts, start=None, behaviour=None, hover_p=0.12):
    N = len(ts)
    behaviour = behaviour or rng.choice(["transit", "circle", "forage"], p=[0.6, 0.25, 0.15])
    if start is None:
        r = rng.uniform(50, 1400); a = rng.uniform(0, 2 * np.pi)
        start = np.array([r * np.sin(a), r * np.cos(a), rng.uniform(5, 150)])
    p = start.astype(float).copy()
    speed = rng.uniform(5, 15)
    hdg = rng.uniform(0, 2 * np.pi)
    centre = p[:2] + rng.normal(0, 30, 2)
    if behaviour in ("circle", "forage") and rng.random() < 0.6:
        # gulls / crows working the yard and kitchen bins: loiter over the site itself
        centre = rng.uniform([-180, -130], [180, 130])
        p[:2] = centre + rng.normal(0, 20, 2)
        p[2] = rng.uniform(10, 60)
    radius = rng.uniform(20, 60)
    phi = rng.uniform(0, 2 * np.pi)
    hover_at = rng.uniform(ts[0], ts[-1]) if rng.random() < hover_p else None  # kestrel / gull hanging in wind
    hover_len = rng.uniform(4, 25)
    pos = np.zeros((N, 3))
    for i, t in enumerate(ts):
        flap = 1.0 + 0.15 * np.sin(2 * np.pi * t / rng.uniform(1.5, 4))
        if hover_at is not None and hover_at <= t < hover_at + hover_len:
            v = np.r_[rng.normal(0, 0.4, 2), rng.normal(0, 0.2)]
        elif behaviour == "circle":
            phi += speed / radius * DT
            tgt = centre + radius * np.array([np.sin(phi), np.cos(phi)])
            v = np.r_[(tgt - p[:2]) / DT * 0.5, rng.normal(0, 0.5)]
        elif behaviour == "forage":
            hdg += rng.normal(0, np.radians(35)) * np.sqrt(DT)
            v = np.r_[speed * 0.6 * flap * np.array([np.sin(hdg), np.cos(hdg)]), rng.normal(0, 1.0)]
            v[:2] += (centre - p[:2]) * 0.02
            if p[2] > 25: v[2] -= 1.0
        else:
            hdg += rng.normal(0, np.radians(10)) * np.sqrt(DT)
            v = np.r_[speed * flap * np.array([np.sin(hdg), np.cos(hdg)]), rng.normal(0, 0.6)]
        p = p + v * DT
        p[2] = np.clip(p[2], 2.0, 250.0)
        pos[i] = p
    return Entity(eid, "bird", pos, rcs=10 ** rng.uniform(-3, -1.3), size=rng.uniform(0.3, 1.3),
                  behaviour=behaviour, hover_at=hover_at)


def make_flock(rng, first_id, ts, n):
    leader = make_bird(rng, first_id, ts, behaviour="transit")
    ents = [leader]
    for k in range(1, n):
        off = rng.normal(0, 6, 3)
        jitter = np.cumsum(rng.normal(0, 0.3, (len(ts), 3)), axis=0) * 0.2
        pos = leader.pos + off + jitter
        pos[:, 2] = np.clip(pos[:, 2], 2, 250)
        ents.append(Entity(first_id + k, "bird", pos, rcs=leader.rcs * rng.uniform(0.5, 1.5),
                           size=leader.size * rng.uniform(0.8, 1.2), behaviour="flock"))
    return ents


def make_aircraft(rng, eid, ts):
    a = rng.uniform(0, 2 * np.pi); off = rng.uniform(-2500, 2500)
    alt = rng.uniform(600, 3000); spd = rng.uniform(60, 120)
    d = np.array([np.sin(a), np.cos(a), 0]); n = np.array([np.cos(a), -np.sin(a), 0])
    t_mid = rng.uniform(ts[0], ts[-1])
    pos = (n * off)[None, :] + d[None, :] * spd * (ts - t_mid)[:, None]
    pos[:, 2] = alt
    return Entity(eid, "aircraft", pos, rcs=rng.uniform(2, 20), size=rng.uniform(10, 40))


def make_bag(rng, eid, ts, wind):
    p = np.r_[rng.uniform(-400, 400, 2), rng.uniform(5, 40)]
    v = np.r_[wind, 0.0]
    pos = np.zeros((len(ts), 3))
    for i in range(len(ts)):
        v = v + (np.r_[wind, 0] - v) * 0.1 + rng.normal(0, 0.6, 3)
        p = p + v * DT
        p[2] = np.clip(p[2], 0.5, 80)
        pos[i] = p
    return Entity(eid, "bag", pos, rcs=rng.uniform(0.0005, 0.003), size=rng.uniform(0.3, 0.6))


def make_truck(rng, eid, ts):
    R = SITE["perimeter_road_radius"]
    a0 = rng.uniform(0, 2 * np.pi); w = rng.uniform(10, 20) / R * rng.choice([-1, 1])
    a = a0 + w * (ts - ts[0])
    pos = np.c_[R * np.sin(a), R * np.cos(a), np.full(len(ts), 1.5)]
    return Entity(eid, "truck", pos, rcs=rng.uniform(5, 50), size=8.0)


def make_episode(rng, kind, wind=None):
    """kind: drone | bird | aircraft | bag | truck. Returns dict(ts, entities, ...)."""
    wind = wind if wind is not None else rng.normal(0, 3, 2)
    ents = []
    if kind == "drone":
        ts = np.arange(0, 480, DT)
        emitting = rng.random() < 0.80     # ~20% fly pre-programmed missions with no live link
        rid = rng.random() < 0.25          # stock firmware still broadcasting Remote ID
        ents.append(make_drone(rng, 1, ts, rng.uniform(5, 20), wind, emitting, rid))
        for k in range(rng.integers(0, 4)):
            ents.append(make_bird(rng, 10 + k, ts))
        if rng.random() < 0.3:
            ents.append(make_truck(rng, 30, ts))
    else:
        ts = np.arange(0, 120, DT)
        if kind == "bird":
            n = 1 if rng.random() < 0.6 else int(rng.integers(3, 10))
            ents += make_flock(rng, 10, ts, n) if n > 1 else [make_bird(rng, 10, ts)]
        elif kind == "aircraft":
            ents.append(make_aircraft(rng, 40, ts))
        elif kind == "bag":
            ents.append(make_bag(rng, 50, ts, wind))
        elif kind == "truck":
            ents.append(make_truck(rng, 30, ts))
    return {"kind": kind, "ts": ts, "entities": ents, "wind": wind}
