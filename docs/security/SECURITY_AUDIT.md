# Security audit: `sim-core` + browser simulator

Date: 2026-11-16. Scope: every Python module in this package, `interactive_sim.html` (same file as `apps/simulator/index.html`), and the README's model-integration instructions. Method: manual review, input fuzzing and resource probes of each CLI, symlink and path-traversal tests, and a closed-loop fault-injection run of the fusion engine. `bandit`, `pip-audit` and `semgrep` are not installed in this environment and could not be installed, so there is no automated SAST or CVE scan yet. The commands to run them are at the end.

Fixes were applied only to `render.py`, `generate_dataset.py`, `evaluate.py` and the new `security/` folder. Other files were being edited in parallel by the red-team and data-pipeline work, so their issues are written up as **pending patches**.

Regression tests: `python -m unittest discover -s security/tests -t . -v` runs 20 tests, all passing.

## Summary

| ID | Severity | Finding | Status |
|---|---|---|---|
| F-01 | **High** | Unbounded sprite canvas in `render.py`: a large pixel size allocates gigabytes (px=200 took 2.4 GB and 21 s; px=800 was killed for running out of memory). Reachable from the closed-loop `RenderedModelClassifier` with simulator-derived sizes. | Fixed |
| F-03 | **High** | A camera model that returns NaN or raises an error silently disables alerting. `np.clip` passes NaN through and the evidence becomes NaN, so `p >= threshold` is never true. In 5 of 6 test drone runs, including drones with live RF links, no alert was ever raised. | Mitigated: `security/safe_classifier.py`. Root fix pending (P-1) |
| F-13 | **High (product)** | Remote ID broadcasts are trusted without corroboration. A single broadcast creates a *confirmed* track, and the self-reported operator position becomes the pilot location sent to police. Spoofed Remote ID can fabricate alerts or direct officers to an innocent person's address. | Pending (P-2) |
| F-02 | Medium | CLI arguments were unbounded: `--n`, `--nights`, `--sweep-nights` and `--procs` (a `Pool(100000)` call is effectively a fork bomb). NaN and inf values were accepted for strengths and sweep values. An unknown `--threshold` crashed with a KeyError after the whole run finished. | Fixed |
| F-04 | Medium | Output paths could write anywhere, including through `..` or a planted symlink, and `--export-sample` accepted any path. | Fixed |
| F-09 | Medium (result integrity) | `data.yaml` used the same folder for `train` and `val`, so validation scores measured memorisation and overstated model quality. | Fixed: seeded split into `train.txt` and `val.txt` |
| F-16 | Medium | The README's model example loads Ultralytics `.pt` weights. These are pickle files, so loading untrusted weights runs arbitrary code. Ultralytics itself suffered a PyPI supply-chain compromise in Dec 2024 ([Wiz](https://www.wiz.io/blog/ultralytics-ai-library-hacked-via-github-for-cryptomining)). | Pending (P-4, training pipeline owner) |
| F-17 | Medium (latent) | The browser simulator renders panels with `innerHTML`. Today every string is internal, so nothing is exploitable. In the real operator console, Remote ID serials and operator IDs arrive over the air and are attacker-controlled; rendered the same way, that becomes stored cross-site scripting (XSS) in the dashboard. | Pending (P-5, HTML owner) |
| F-10 | Low/Med | No dependency pinning. | Fixed: `requirements.txt` pinned. A hash-locked file needs network access (pending) |
| F-14 | Low | Per-track lists (`hist`, `cam`, `ctrl`, `rid_ops`) grow without limit. That is harmless in the minutes-long simulator episodes, but in a 24/7 edge deployment memory grows without bound. | Pending (P-3) |
| F-05 | Low | The dataset path was written into YAML unquoted, so a path containing `:`, `#` or a newline could inject keys. | Fixed |
| F-06 | Low | JSON output could contain `NaN`, which is not valid JSON, and `default=str` silently turned unknown objects into strings. | Fixed (`allow_nan=False`, explicit `_clean`) |
| F-07 | Low | `render_frame` silently accepted an invalid mode, class or pixel size (NaN, negative, inf). | Fixed |
| F-08 | Low | Seed reuse: sweep nights reused main-run seeds once `--nights` exceeded 10,000, giving correlated "independent" samples. Episode seeds `seed*1000+j` collide if a night has more than 1,000 events. | Fixed (offset 1,000,003 plus an assert) |
| F-11 | Low | Outputs were world-readable. A real site's geometry and sensor layout (as in `sample_run.json`) is sensitive because it reveals blind spots. | Fixed: files 0640, dirs 0750 |
| F-12 | Low | Re-running the generator silently mixed new files into an old dataset. | Fixed: refuses a non-empty directory unless `--overwrite` |
| F-15 | Info (quality) | `fusion._pilot` runs least squares over raw controller bearings. With near-parallel bearing lines this is biased toward the sensors; the browser simulator hit the same problem and fixed it by averaging per sensor first. | Pending, for the fusion owner |
| F-18 | Info | The browser simulator uses `Math.random` in the render path (insect sprite), so copied YOLO labels are not reproducible from the seed. | Pending (P-5) |

Not found: `pickle` loading, `yaml.load`, `subprocess`, `os.system`, `eval`/`exec`, network calls, or temp files in shared directories. The simulator seeds all of its randomness through `np.random.default_rng`.

## What changed

- `security/safe_io.py`: `resolve_output_dir` normalises the path, keeps it inside the working directory unless `--allow-outside` is passed, refuses symlinked path components and creates directories 0750. `safe_join` blocks escapes and symlinked targets. Atomic writes use a temp file in the same directory followed by `os.replace`, which replaces a planted symlink rather than following it, and set file mode 0640. Also: `bounded_int` and `bounded_float` argparse types that reject NaN and inf, and `yaml_scalar`.
- `security/safe_classifier.py`: wraps any classifier. Non-finite, out-of-range or non-numeric outputs and exceptions become a neutral 0.5 (log-odds 0, so the camera adds no evidence). It counts faults and exposes `healthy` and an `on_fault` callback, so the console can raise a "camera classifier degraded" alarm instead of failing silently.
- `render.py`: `MAX_PX = 128` and `MAX_CANVAS = 512` cap peak memory at about 240 MB for any input. `render_frame` validates mode, class and pixel size.
- `generate_dataset.py`: bounded `--n` (at most 200,000) and `--seed`, safe output directory, `--overwrite`, atomic writes, quoted YAML, and a seeded train/val split (`--val-frac`, default 0.1).
- `evaluate.py`: bounded `--nights`, `--sweep-nights` and `--procs` (at most 2× CPU count, capped at 64); `--threshold` validated up front; finite sweep values, at most 10; safe `--out` and `--export-sample` (must be `.json`); strict-JSON outputs; atomic writes; non-overlapping seeds.
- `requirements.txt`: exact versions; `opencv-python-headless` instead of the GUI build.

Behaviour changes to be aware of: re-running into the existing `data/` or `results/` now needs `--overwrite`; `data.yaml` now points at `train.txt` and `val.txt`; targets larger than 128 px are drawn at 128 px. The frame is 640 px wide and the generator's own sampling tops out at 160 px, so this affects only extreme close-ups.

## Pending patches for other owners

**P-1, `fusion.py` and `sensors.py` (F-03, root fix).** Fail safe on non-finite camera scores, and wrap the classifier by default:
```python
# fusion.py, Fusion.update_camera
        score = float(obs["score"])
        if not np.isfinite(score):          # a broken model must not poison evidence
            self.camera_faults = getattr(self, "camera_faults", 0) + 1
            return
        s = np.clip(score, 1e-4, 1 - 1e-4)
# sensors.py, SensorSuite.__init__
        from security.safe_classifier import SafeClassifier
        self.classifier = SafeClassifier(p.get("classifier", StandInClassifier(p.get("classifier_strength", 4.0))))
```
Once applied, `test_nan_model_without_wrapper_suppresses_alerts` detects the patch and skips its "documents the bug" assertion.

**P-2, `fusion.py` `update_rf` (F-13): corroborate Remote ID before trusting it.** Remote ID is unauthenticated by design. Fielded receivers have had over-the-air injection CVEs ([Nozomi Networks, DroneScout ds230: CVE-2023-31191, CVSS 9.3](https://nozominetworks.com/blog/nozomi-networks-discovers-three-vulnerabilities-affecting-bluemark-dronescout-ds230-remote-id-receiver)), and authenticated Remote ID schemes are still research ([TBRD, 2025](https://arxiv.org/pdf/2510.11343)).
```python
        for m in rid:
            z, R = m["pos"], np.eye(3) * 16.0
            best = min(self.tracks, key=lambda tr: np.linalg.norm(tr.pos - z), default=None)
            corroborated = (best is not None and np.linalg.norm(best.pos - z) <= 40
                            and best.status == "confirmed" and sum(best.hits[-6:]) >= 3)   # radar agrees
            if best is None or np.linalg.norm(best.pos - z) > 40:
                best = Track(t, z, R); self.tracks.append(best)        # stays TENTATIVE: RID alone never confirms
            else:
                best.update(z, R)
            best.last = t
            best.rid_claims = getattr(best, "rid_claims", 0) + 1
            if corroborated:
                best.rid = True
                best.rid_ops.append(m["op"][:2])
            # uncorroborated claims: keep for the evidence log, never for pilot dispatch
```
Also: label the pilot location `remote_id_unverified` unless the RF direction-finding bearing to the Remote ID transmitter agrees with the claimed drone position (within about 6° at two nodes). Raise a "Remote ID inconsistent with radar" alarm when a claim sits more than 100 m from every radar track for more than 5 s. That pattern is the spoofing signature.

**P-3, `fusion.py` (F-14): cap per-track history.** `deque` can't be sliced and `_kinematics` uses `tr.hist[-20:]`, so trim the lists after appending:
```python
        tr.hist.append((t, *x)); del tr.hist[:-600]       # ~5 min at 2 Hz
        tr.cam.append(...);      del tr.cam[:-200]
        best.ctrl.append(...);   del best.ctrl[:-400]
        best.rid_ops.append(...);del best.rid_ops[:-200]
```

**P-4, training pipeline and README (F-16).**
- Pin `ultralytics==<exact>` with `--require-hashes`, and use a version released after December 2024.
- Record the SHA-256 of every downloaded weight file and dataset archive in the manifest, and check it before use.
- For inference in `evaluate.py` and on the edge box, export to ONNX and run it with `onnxruntime`. ONNX is not a pickle format.
- Never `torch.load` or `YOLO(path)` a file whose hash is not in the manifest.
- Sign released model artifacts (Sigstore/cosign) and verify the signature on the edge box before loading.

**P-5, HTML simulator and the future operator console (F-17, F-18).**
- Replace `innerHTML` string building with DOM construction and `textContent`, or escape every interpolation with an `esc()` helper (`&<>"'`).
- Production console: strict CSP (`default-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'`), self-hosted fonts (Google Fonts leaks operator IPs to a third party), and Trusted Types.
- Use the seeded RNG for anything that affects labels.

## Still to do (needs network or other tools)
```bash
pip install pip-tools pip-audit bandit semgrep
pip-compile --generate-hashes -o requirements.lock requirements.txt
pip-audit -r requirements.lock
bandit -r . -x ./security/tests
semgrep --config p/python --config p/secrets .
```
