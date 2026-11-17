"""Monte Carlo evaluation of the full sensor + fusion chain over many simulated nights.

Each night is compressed into independent event episodes (drone runs, bird
passes, aircraft, drifting bags, trucks on the perimeter road); radar clutter
is present in every episode. Quiet time between events is not simulated (see README).

Usage:
  python evaluate.py --nights 100 --threshold 0.9 --classifier-strength 4
  python evaluate.py --export-sample sample_run.json
"""
import argparse, json, math, os, sys, time
from multiprocessing import Pool
import numpy as np
from security.safe_io import (bounded_int, bounded_float, resolve_output_dir, safe_join,
                              atomic_write_text, atomic_write_json, UnsafePathError)

# Resource limits (security audit F-02). One night is ~35 episodes; results are held in memory.
MAX_NIGHTS = 20_000
MAX_PROCS = max(1, min(64, 2 * (os.cpu_count() or 1)))
SWEEP_SEED_OFFSET = 1_000_003  # > MAX_NIGHTS, so sweep nights never reuse main-run seeds (F-08)
from scenario import make_episode, SITE
from sim import run_episode
from fusion import THRESHOLDS
from sensors import StandInClassifier

NIGHT_MIX = {"drone": None, "bird": 25, "aircraft": 4, "bag": 1, "truck": 4}


def simulate_night(args):
    night, seed, strength = args
    rng = np.random.default_rng(seed)
    params = {"classifier": StandInClassifier(strength=strength)}
    n_drone = rng.choice([0, 1, 2], p=[0.4, 0.45, 0.15])
    kinds = ["drone"] * n_drone
    for k, lam in NIGHT_MIX.items():
        if lam:
            kinds += [k] * rng.poisson(lam)
    res = []
    assert len(kinds) < 1000, "episode seeds are seed*1000+j; >1000 episodes would collide across nights"
    for j, k in enumerate(kinds):
        ep = make_episode(rng, k)
        r = run_episode(ep, seed * 1000 + j, params)
        r["night"] = night
        res.append(r)
    return res


def run_nights(n, seed, strength, procs=2):
    jobs = [(i, seed + i, strength) for i in range(n)]
    with Pool(procs) as pool:
        nights = pool.map(simulate_night, jobs)
    return nights


def summarise(nights, th):
    eps = [e for n in nights for e in n]
    drones = [d for e in eps for d in e["drones"]]
    fa_night = np.array([sum(e["false_alarms"][th] for e in n) for n in nights])
    fa_kinds = {}
    for e in eps:
        for k, v in e["fa_kinds"][th].items():
            fa_kinds[k] = fa_kinds.get(k, 0) + v

    def group(ds):
        if not ds:
            return None
        al = [d for d in ds if d["first_alert"][th] is not None]
        before = [d for d in al if d["drop_t"] is not None and d["first_alert"][th] <= d["drop_t"]]
        tta = [d["first_alert"][th] - d["takeoff_t"] for d in al]
        lead = [d["drop_t"] - d["first_alert"][th] for d in before]
        de = [d["drop_err"] for d in ds if d["drop_err"] is not None]
        da = [d["drop_err_at_alert"] for d in ds if d.get("drop_err_at_alert") is not None]
        pe = [d["pilot_err"] for d in ds if d["pilot_err"] is not None]
        return {"n": len(ds), "alerted": len(al) / len(ds), "alert_before_drop": len(before) / len(ds),
                "median_time_to_alert_s": float(np.median(tta)) if tta else None,
                "median_warning_before_drop_s": float(np.median(lead)) if lead else None,
                "drop_zone_err_median_m": float(np.median(de)) if de else None,
                "drop_zone_err_at_alert_median_m": float(np.median(da)) if da else None,
                "pilot_located": len(pe) / len(ds),
                "pilot_err_median_m": float(np.median(pe)) if pe else None,
                "dup_alerts_per_run": float(np.mean([d["dup_tracks"][th] for d in ds]))}

    return {"threshold": th, "nights": len(nights), "episodes": len(eps),
            "false_alarms_per_night": float(fa_night.mean()),
            "nights_with_false_alarm": float((fa_night > 0).mean()),
            "false_alarm_sources": fa_kinds,
            "all": group(drones),
            "rf_emitting": group([d for d in drones if d["emitting"] and not d["remote_id"]]),
            "remote_id": group([d for d in drones if d["remote_id"]]),
            "dark": group([d for d in drones if not d["emitting"] and not d["remote_id"]])}


def fmt(v, pct=False, unit=""):
    if v is None:
        return "–"
    return f"{100 * v:.0f}%" if pct else f"{v:.1f}{unit}"


def write_report(main, sweep, out_dir, args, runtime):
    s = main[args.threshold]
    lines = [f"# Monte Carlo evaluation: {s['nights']} simulated nights",
             "",
             f"Alert threshold P(drone) ≥ {args.threshold}; stand-in camera classifier strength {args.classifier_strength}; "
             f"{s['episodes']} episodes; runtime {runtime:.0f} s.",
             "",
             "## Headline",
             "",
             f"- **False alarms per site-night:** {s['false_alarms_per_night']:.2f} "
             f"({100 * s['nights_with_false_alarm']:.0f}% of nights had at least one)",
             f"- **False alarm sources:** {s['false_alarm_sources'] or 'none'}",
             "",
             "| Drone group | Runs | Alerted | Alerted before drop | Median takeoff→alert | Median warning before drop | Drop-zone error at alert / at drop (median) | Pilot located | Pilot error (median) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, key in [("All drones", "all"), ("RF link, no Remote ID", "rf_emitting"),
                      ("Remote ID broadcasting", "remote_id"), ("Dark (no RF, no RID)", "dark")]:
        g = s[key]
        if g is None:
            continue
        lines.append(f"| {name} | {g['n']} | {fmt(g['alerted'], True)} | {fmt(g['alert_before_drop'], True)} | "
                     f"{fmt(g['median_time_to_alert_s'], unit=' s')} | {fmt(g['median_warning_before_drop_s'], unit=' s')} | "
                     f"{fmt(g['drop_zone_err_at_alert_median_m'], unit=' m')} / {fmt(g['drop_zone_err_median_m'], unit=' m')} | {fmt(g['pilot_located'], True)} | "
                     f"{fmt(g['pilot_err_median_m'], unit=' m')} |")
    lines += ["", "## Threshold trade-off (same nights)", "",
              "| Threshold | False alarms / night | Drones alerted before drop | Dark drones alerted before drop |",
              "|---|---|---|---|"]
    for th in THRESHOLDS:
        m = main[th]
        lines.append(f"| {th} | {m['false_alarms_per_night']:.2f} | {fmt(m['all']['alert_before_drop'], True)} | "
                     f"{fmt(m['dark']['alert_before_drop'] if m['dark'] else None, True)} |")
    if sweep:
        lines += ["", "## How good does the vision model need to be?", "",
                  f"Classifier strength sweep at threshold {args.threshold} ({args.sweep_nights} nights each). "
                  "Strength ≈ how separable drone vs bird logits are at good pixel counts.", "",
                  "| Classifier strength | False alarms / night | All drones before drop | Dark drones before drop |",
                  "|---|---|---|---|"]
        for st, m in sweep.items():
            lines.append(f"| {st} | {m['false_alarms_per_night']:.2f} | {fmt(m['all']['alert_before_drop'], True)} | "
                         f"{fmt(m['dark']['alert_before_drop'] if m['dark'] else None, True)} |")
    lines += ["", "Charts: `tradeoff.png`, `pilot_error_cdf.png`, `time_to_alert.png`.", ""]
    atomic_write_text(safe_join(out_dir, "report.md"), "\n".join(lines))


def _savefig(fig, out_dir, name):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    from security.safe_io import atomic_write_bytes
    atomic_write_bytes(safe_join(out_dir, name), buf.getvalue())


def _clean(o):
    """Make results strict-JSON safe: NaN/inf -> None, numpy scalars -> Python (audit F-06)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def charts(nights, main, out_dir, th):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    # 1. trade-off
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, lab, c in [("all", "All drones", "#2a6fdb"), ("dark", "Dark drones", "#d9822b")]:
        xs = [main[t]["false_alarms_per_night"] for t in THRESHOLDS if main[t][key]]
        ys = [100 * main[t][key]["alert_before_drop"] for t in THRESHOLDS if main[t][key]]
        ax.plot(xs, ys, "o-", color=c, label=lab)
        for t, x, y in zip(THRESHOLDS, xs, ys):
            if key == "all":
                ax.annotate(str(t), (x, y), textcoords="offset points", xytext=(4, -10), fontsize=7, color="#555")
    ax.set_xlabel("False alarms per site-night"); ax.set_ylabel("Alerted before payload drop (%)")
    ax.set_title("Alert threshold trade-off"); ax.legend(frameon=False); ax.set_ylim(0, 102)
    fig.tight_layout(); _savefig(fig, out_dir, "tradeoff.png"); plt.close(fig)
    # 2. pilot error CDF
    drones = [d for n in nights for e in n for d in e["drones"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, lab, c in [("remote_id", "Remote ID operator field", "#2a6fdb"), ("rf_df", "RF triangulation (3 DF nodes)", "#d9822b")]:
        v = np.sort([d["pilot_err"] for d in drones if d["pilot_method"] == m])
        if len(v):
            ax.step(v, np.arange(1, len(v) + 1) / len(v) * 100, where="post", color=c, label=f"{lab} (n={len(v)})")
    ax.set_xscale("log"); ax.set_xlabel("Pilot location error (m)"); ax.set_ylabel("Runs (%)")
    ax.set_title("Pilot location error"); ax.legend(frameon=False)
    fig.tight_layout(); _savefig(fig, out_dir, "pilot_error_cdf.png"); plt.close(fig)
    # 3. time to alert vs drop time
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, lab, c in [(lambda d: d["remote_id"], "Remote ID", "#2a6fdb"),
                        (lambda d: d["emitting"] and not d["remote_id"], "RF link", "#3aa676"),
                        (lambda d: not d["emitting"] and not d["remote_id"], "Dark", "#d9822b")]:
        v = [(d["first_alert"][th] - d["takeoff_t"]) for d in drones if key(d) and d["first_alert"][th] is not None]
        if v:
            ax.hist(v, bins=np.arange(0, 400, 10), alpha=0.6, color=c, label=lab)
    dt = [d["drop_t"] - d["takeoff_t"] for d in drones if d["drop_t"]]
    ax.axvline(np.median(dt), color="#333", ls="--", lw=1)
    ax.text(np.median(dt), ax.get_ylim()[1] * 0.9, " median drop time", fontsize=8)
    ax.set_xlabel("Takeoff → alert (s)"); ax.set_ylabel("Drone runs"); ax.set_title("Time to alert")
    ax.legend(frameon=False)
    fig.tight_layout(); _savefig(fig, out_dir, "time_to_alert.png"); plt.close(fig)


def export_sample(path, seed=7):
    """One drone run (RF link, no Remote ID) with birds, recorded at 2 Hz for the browser replay."""
    for s in range(seed, seed + 200):
        rng = np.random.default_rng(s)
        ep = make_episode(rng, "drone")
        d = ep["entities"][0]
        if d.emitting and not d.remote_id and len(ep["entities"]) >= 3:
            break
    r = run_episode(ep, s, record=True)
    d = ep["entities"][0]
    out = {"site": SITE, "dt": 0.5, "seed": int(s),
           "scenario": {"drone_emitting": d.emitting, "remote_id": d.remote_id, "pilot": [round(float(v), 1) for v in d.attrs["pilot"]],
                        "takeoff_t": float(d.attrs["takeoff_t"]), "drop_t": d.attrs["drop_t"],
                        "drop_pos": [round(float(v), 1) for v in d.attrs["drop_pos"]] if d.attrs["drop_pos"] is not None else None,
                        "entities": [{"id": e.id, "kind": e.kind} for e in ep["entities"]]},
           "score": {k: v for k, v in r.items() if k not in ("frames",)},
           "frames": r["frames"]}
    atomic_write_json(path, _clean(out), separators=(",", ":"))
    print(f"wrote {path} ({os.path.getsize(path) / 1e3:.0f} kB, {len(r['frames'])} frames)")


def _sweep_list(s):
    vals = []
    for x in s.split(","):
        x = x.strip()
        if x:
            vals.append(bounded_float(0.0, 50.0, "--sweep value")(x))
    if len(vals) > 10:
        raise argparse.ArgumentTypeError("--sweep takes at most 10 values")
    return vals


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Monte Carlo evaluation of the sensor + fusion chain.")
    ap.add_argument("--nights", type=bounded_int(1, MAX_NIGHTS, "--nights"), default=100)
    ap.add_argument("--seed", type=bounded_int(0, 2**31 - 1, "--seed"), default=1)
    ap.add_argument("--threshold", type=float, default=0.9, help=f"one of {list(THRESHOLDS)}")
    ap.add_argument("--classifier-strength", type=bounded_float(0.0, 50.0, "--classifier-strength"), default=4.0)
    ap.add_argument("--sweep", type=_sweep_list, default="1,2,6", help="comma list of classifier strengths, '' to skip")
    ap.add_argument("--sweep-nights", type=bounded_int(1, MAX_NIGHTS, "--sweep-nights"), default=40)
    ap.add_argument("--procs", type=bounded_int(1, MAX_PROCS, "--procs"), default=min(MAX_PROCS, os.cpu_count() or 2))
    ap.add_argument("--out", default="results")
    ap.add_argument("--export-sample", default=None, help="write one recorded run to this .json file")
    ap.add_argument("--allow-outside", action="store_true", help="allow output paths outside the working directory")
    a = ap.parse_args(argv)
    if a.threshold not in THRESHOLDS:
        ap.error(f"--threshold must be one of {list(THRESHOLDS)}")
    return a


def main(argv=None):
    a = parse_args(argv)
    try:
        if a.export_sample:
            if not a.export_sample.endswith(".json"):
                raise UnsafePathError("--export-sample must name a .json file")
            d = resolve_output_dir(os.path.dirname(os.path.abspath(a.export_sample)), allow_outside=a.allow_outside)
            export_sample(safe_join(d, os.path.basename(a.export_sample)))
            return
        out = resolve_output_dir(a.out, allow_outside=a.allow_outside)
    except UnsafePathError as e:
        sys.exit(f"error: {e}")
    t0 = time.time()
    nights = run_nights(a.nights, a.seed, a.classifier_strength, a.procs)
    main_s = {th: summarise(nights, th) for th in THRESHOLDS}
    sweep = {}
    for st in a.sweep:
        sweep[st] = summarise(run_nights(a.sweep_nights, a.seed + SWEEP_SEED_OFFSET, st, a.procs), a.threshold)
    if sweep:
        sweep[a.classifier_strength] = main_s[a.threshold]
        sweep = dict(sorted(sweep.items()))
    rt = time.time() - t0
    write_report(main_s, sweep, out, a, rt)
    charts(nights, main_s, out, a.threshold)
    atomic_write_json(safe_join(out, "summary.json"),
                      _clean({"main": {str(k): v for k, v in main_s.items()}, "sweep": {str(k): v for k, v in sweep.items()}}),
                      indent=1)
    with open(safe_join(out, "report.md"), encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
