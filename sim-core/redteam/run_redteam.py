"""Run every red-team scenario with the v1 fusion (hardened=False) and the hardened fusion,
over many seeds, plus attack-free control nights; write redteam/results/redteam_report.md + json.

  python -m redteam.run_redteam --seeds 30 --nights 40
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from redteam.attacks import SCENARIOS, EXPECTED_FLAG          # noqa: E402
from scenario import make_episode                              # noqa: E402
from sim import run_episode                                    # noqa: E402
from sensors import StandInClassifier                          # noqa: E402

TH = 0.9


class PairedStandIn(StandInClassifier):
    """Stand-in classifier whose 'consistently wrong about this object' draw depends only on
    (episode seed, object id), so the v1 and hardened runs face the same hard objects."""

    def __init__(self, seed, **kw):
        super().__init__(**kw)
        self.seed = seed

    def score(self, rng, kind, px, has_payload=False, obj_id=None):
        if obj_id is not None and obj_id not in self._hard:
            self._hard[obj_id] = np.random.default_rng([self.seed, int(obj_id) + 10]).random() < self.hard_rate
        return super().score(rng, kind, px, has_payload, obj_id)
NIGHT_MIX = {"bird": 25, "aircraft": 4, "bag": 1, "truck": 4}


def _job(args):
    name, seed, hardened = args
    rng = np.random.default_rng(seed)
    ep = SCENARIOS[name](rng)
    r = run_episode(ep, seed + 7, {"hardened": hardened, "threshold": TH, "classifier": PairedStandIn(seed)})
    return name, hardened, r


def _night(args):
    seed, hardened = args
    rng = np.random.default_rng(seed)
    fa, sens, spoof = 0, 0, 0
    for k, lam in NIGHT_MIX.items():
        for j in range(rng.poisson(lam)):
            ep = make_episode(rng, k)
            r = run_episode(ep, seed * 997 + j, {"hardened": hardened, "threshold": TH, "classifier": PairedStandIn(seed * 997 + j)})
            fa += r["false_alarms"][TH]
            sens += sum(a["state"] == "degraded" for a in r["sensor_alerts"])
            spoof += len(r["spoof_alerts"])
    return hardened, fa, sens, spoof


def summarise(name, runs):
    pay = [d for r in runs for d in r["drones"] if not d["decoy"]]
    dec = [d for r in runs for d in r["drones"] if d["decoy"]]
    before = [d for d in pay if d["first_alert"][TH] is not None and d["drop_t"] is not None and d["first_alert"][TH] <= d["drop_t"]]
    alerted = [d for d in pay if d["first_alert"][TH] is not None]
    tta = [d["first_alert"][TH] - d["takeoff_t"] for d in alerted]
    lead = [d["drop_t"] - d["first_alert"][TH] for d in before]
    pe = [d["pilot_err"] for d in pay if d["pilot_err"] is not None and d["emitting"]]
    n_emit = sum(d["emitting"] for d in pay)
    de = [d["drop_err"] for d in pay if d["drop_err"] is not None]
    fa = [r["false_alarms"][TH] for r in runs]
    fa_kinds = {}
    for r in runs:
        for k, v in r["fa_kinds"][TH].items():
            fa_kinds[k] = fa_kinds.get(k, 0) + v
    flag = None
    if name in EXPECTED_FLAG:
        kind, what = EXPECTED_FLAG[name]
        if kind == "spoof":
            flag = np.mean([any(a["type"] == what for a in r["spoof_alerts"]) for r in runs])
        else:
            flag = np.mean([any(a["sensor"] == what and a["state"] == "degraded" for a in r["sensor_alerts"]) for r in runs])
    other_flags = np.mean([bool(r["spoof_alerts"]) or any(a["state"] == "degraded" for a in r["sensor_alerts"]) for r in runs])
    m = lambda v: float(np.median(v)) if len(v) else None
    return {"runs": len(runs), "payload_drones": len(pay),
            "alert_before_drop": len(before) / len(pay) if pay else None,
            "alerted": len(alerted) / len(pay) if pay else None,
            "median_takeoff_to_alert_s": m(tta), "median_warning_s": m(lead),
            "false_alarms_per_run": float(np.mean(fa)), "fa_kinds": fa_kinds,
            "decoys_alerted": (sum(d["first_alert"][TH] is not None for d in dec) / len(dec)) if dec else None,
            "attack_flagged": None if flag is None else float(flag), "any_indicator": float(other_flags),
            "pilot_err_median_m": m(pe), "pilot_located": (len(pe) / n_emit) if n_emit else None,
            "drop_err_median_m": m(de)}


def pct(v):
    return "–" if v is None else f"{100 * v:.0f}%"


def num(v, u=""):
    return "–" if v is None else f"{v:.0f}{u}"


def write_md(res, nights, out, args, rt):
    L = [f"# Red-team results: {args.seeds} seeds per scenario, threshold {TH}", "",
         f"v1 = original fusion (`hardened=False`); H = hardened fusion (default). Same seeds, same sensors, "
         f"same stand-in camera classifier. Runtime {rt:.0f} s.", "",
         "| Scenario | Payload drones alerted before drop (v1 → H) | Median takeoff→alert (v1 → H) | False drone alerts / run (v1 → H) | Attack flagged to operator (v1 → H) | Pilot error, median (v1 → H) |",
         "|---|---|---|---|---|---|"]
    for name in SCENARIOS:
        a, b = res[name]["v1"], res[name]["H"]
        L.append(f"| {name} | {pct(a['alert_before_drop'])} → **{pct(b['alert_before_drop'])}** | "
                 f"{num(a['median_takeoff_to_alert_s'], ' s')} → {num(b['median_takeoff_to_alert_s'], ' s')} | "
                 f"{a['false_alarms_per_run']:.2f} → **{b['false_alarms_per_run']:.2f}** | "
                 f"{pct(a['attack_flagged']) if a['attack_flagged'] is not None else 'n/a'} → "
                 f"{pct(b['attack_flagged']) if b['attack_flagged'] is not None else 'n/a'} | "
                 f"{num(a['pilot_err_median_m'], ' m')} → {num(b['pilot_err_median_m'], ' m')} |")
    v1n = [x for x in nights if not x[0]]; hn = [x for x in nights if x[0]]
    L += ["", f"## Attack-free control nights ({len(hn)} nights, ~34 bird / aircraft / bag / truck events each)", "",
          "| Fusion | False drone alerts / night | Sensor-degraded alarms / night | Spoof indicators / night |", "|---|---|---|---|",
          f"| v1 | {np.mean([x[1] for x in v1n]):.2f} | – | – |",
          f"| Hardened | {np.mean([x[1] for x in hn]):.2f} | {np.mean([x[2] for x in hn]):.2f} | {np.mean([x[3] for x in hn]):.2f} |", ""]
    L += ["## False-alarm sources (hardened)", ""]
    for name in SCENARIOS:
        if res[name]["H"]["fa_kinds"]:
            L.append(f"- {name}: {res[name]['H']['fa_kinds']}")
    L += ["", "Decoy drones alerted (they are real drones, so alerting is correct): " +
          ", ".join(f"{n} v1 {pct(res[n]['v1']['decoys_alerted'])} / H {pct(res[n]['H']['decoys_alerted'])}"
                    for n in SCENARIOS if res[n]["H"]["decoys_alerted"] is not None), ""]
    open(os.path.join(out, "redteam_report.md"), "w").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--nights", type=int, default=40)
    ap.add_argument("--only", default="")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    a = ap.parse_args()
    if not (1 <= a.seeds <= 2000 and 0 <= a.nights <= 2000):
        ap.error("seeds/nights out of range")
    os.makedirs(a.out, exist_ok=True)
    names = [n for n in SCENARIOS if not a.only or n in a.only.split(",")]
    jobs = [(n, 1000 + s, h) for n in names for s in range(a.seeds) for h in (False, True)]
    t0 = time.time()
    with Pool(a.procs) as pool:
        out = pool.map(_job, jobs, chunksize=4)
        nights = pool.map(_night, [(50_000 + s, h) for s in range(a.nights) for h in (False, True)]) if a.nights else []
    res = {n: {"v1": summarise(n, [r for nn, h, r in out if nn == n and not h]),
               "H": summarise(n, [r for nn, h, r in out if nn == n and h])} for n in names}
    rt = time.time() - t0
    json.dump({"threshold": TH, "results": res, "nights": [list(map(float, x)) for x in nights]},
              open(os.path.join(a.out, "redteam_summary.json"), "w"), indent=1)
    if set(names) == set(SCENARIOS):
        write_md(res, nights, a.out, a, rt)
        print(open(os.path.join(a.out, "redteam_report.md")).read())
    else:
        for n in names:
            print(n, json.dumps(res[n], default=str))


if __name__ == "__main__":
    main()
