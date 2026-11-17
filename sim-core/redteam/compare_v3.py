"""Compare fusion generations on the red-team scenarios (same seeds as run_redteam.py).

  v1 : original fusion                        (from redteam_summary_prev.json, unchanged code path)
  H1 : first hardened fusion                  (from redteam_summary_prev.json)
  H2 : H1 + radar micro-Doppler + bird-gated RF + drone-overridable ground class
  H3 : H2 + a remote RF direction-finding mast ~1.1 km out

  python -m redteam.compare_v3 --seeds 20 --nights 20
"""
import argparse, json, os, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from redteam.attacks import SCENARIOS                                          # noqa: E402
from redteam.run_redteam import PairedStandIn, NIGHT_MIX, TH, summarise, pct, num  # noqa: E402
from scenario import make_episode                                              # noqa: E402
from sim import run_episode                                                    # noqa: E402

CONFIGS = {"H2": {"hardened": True, "micro_doppler": True},
           "H3": {"hardened": True, "micro_doppler": True, "rf_remote_mast": True}}


def _job(args):
    name, seed, cfg = args
    rng = np.random.default_rng(seed)
    ep = SCENARIOS[name](rng)
    r = run_episode(ep, seed + 7, {**CONFIGS[cfg], "threshold": TH, "classifier": PairedStandIn(seed)})
    return name, cfg, r


def _night(args):
    seed, cfg = args
    rng = np.random.default_rng(seed)
    fa, sens, spoof, kinds = 0, 0, 0, {}
    for k, lam in NIGHT_MIX.items():
        for j in range(rng.poisson(lam)):
            ep = make_episode(rng, k)
            r = run_episode(ep, seed * 997 + j, {**CONFIGS[cfg], "threshold": TH, "classifier": PairedStandIn(seed * 997 + j)})
            fa += r["false_alarms"][TH]
            sens += sum(a["state"] == "degraded" for a in r["sensor_alerts"])
            spoof += len(r["spoof_alerts"])
    return cfg, fa, sens, spoof


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--nights", type=int, default=20)
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args(argv)
    if not (1 <= a.seeds <= 200 and 0 <= a.nights <= 200 and 1 <= a.procs <= 16):
        ap.error("out of range")
    here = os.path.dirname(os.path.abspath(__file__))
    prev = json.load(open(os.path.join(here, "results", "redteam_summary_prev.json")))["results"]
    t0 = time.time()
    jobs = [(n, 1000 + s, c) for n in SCENARIOS for s in range(a.seeds) for c in CONFIGS]
    with Pool(a.procs) as pool:
        out = pool.map(_job, jobs, chunksize=2)
        nights = pool.map(_night, [(50_000 + s, c) for s in range(a.nights) for c in CONFIGS])
    runs = {}
    for name, cfg, r in out:
        runs.setdefault(name, {}).setdefault(cfg, []).append(r)
    res = {n: {"v1": prev[n]["v1"], "H1": prev[n]["H"], **{c: summarise(n, runs[n][c]) for c in CONFIGS}} for n in SCENARIOS}
    nres = {c: {"fa": float(np.mean([x[1] for x in nights if x[0] == c] or [0])),
                "degraded": float(np.mean([x[2] for x in nights if x[0] == c] or [0])),
                "spoof": float(np.mean([x[3] for x in nights if x[0] == c] or [0]))} for c in CONFIGS}
    rt = time.time() - t0
    json.dump({"threshold": TH, "seeds": a.seeds, "nights": a.nights, "results": res, "nights_summary": nres, "runtime_s": rt},
              open(os.path.join(here, "results", "compare_v3.json"), "w"), indent=1, default=float)
    L = [f"Seeds per scenario: {a.seeds}; runtime {rt:.0f} s", "",
         "| Scenario | Alerted before drop v1 / H1 / H2 / H3 | False alerts per run v1 / H1 / H2 / H3 | Median takeoff→alert H1 → H2 | Pilot error median v1 / H1 / H2 / H3 | Attack flagged H1 → H2 |",
         "|---|---|---|---|---|---|"]
    for n in SCENARIOS:
        r = res[n]
        L.append(f"| {n} | " + " / ".join(pct(r[k]["alert_before_drop"]) for k in ("v1", "H1", "H2", "H3")) + " | "
                 + " / ".join(f"{r[k]['false_alarms_per_run']:.2f}" for k in ("v1", "H1", "H2", "H3")) + " | "
                 + f"{num(r['H1']['median_takeoff_to_alert_s'], ' s')} → {num(r['H2']['median_takeoff_to_alert_s'], ' s')} | "
                 + " / ".join(num(r[k]["pilot_err_median_m"], " m") for k in ("v1", "H1", "H2", "H3")) + " | "
                 + f"{pct(r['H1']['attack_flagged']) if r['H1']['attack_flagged'] is not None else 'n/a'} → "
                 + f"{pct(r['H2']['attack_flagged']) if r['H2']['attack_flagged'] is not None else 'n/a'} |")
    L += ["", "| Control nights | False drone alerts / night | Degraded alarms / night | Spoof indicators / night |", "|---|---|---|---|"]
    for c in CONFIGS:
        L.append(f"| {c} | {nres[c]['fa']:.2f} | {nres[c]['degraded']:.2f} | {nres[c]['spoof']:.2f} |")
    txt = "\n".join(L) + "\n"
    open(os.path.join(here, "results", "compare_v3.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
