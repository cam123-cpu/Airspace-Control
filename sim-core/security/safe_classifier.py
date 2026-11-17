"""Fail-safe wrapper for any camera classifier (StandInClassifier, RenderedModelClassifier, ...).

Audit finding F-03: fusion.update_camera() does np.clip(score) then log-odds. np.clip passes
NaN through, so a model that returns NaN (bad weights, fp16 overflow, a corrupted or poisoned
ONNX file, an exception swallowed upstream) turns the track's evidence into NaN. Every later
`p >= threshold` comparison is then False and the track can NEVER alert: the system fails open,
silently.

This wrapper:
  * coerces the model output to a float in [0, 1];
  * replaces non-finite or out-of-range outputs and exceptions with a NEUTRAL 0.5
    (log-odds 0: the camera then adds no evidence either way, so radar/RF/acoustic still decide);
  * counts faults and flips `healthy` to False after `max_consecutive_faults`, so the caller
    can raise a "camera classifier degraded" operator alarm instead of degrading silently.

Usage:
    from security.safe_classifier import SafeClassifier
    params = {"classifier": SafeClassifier(RenderedModelClassifier(predict))}
"""
import math


class SafeClassifier:
    NEUTRAL = 0.5

    def __init__(self, inner, max_consecutive_faults=5, on_fault=None):
        self.inner = inner
        self.max_consecutive_faults = int(max_consecutive_faults)
        self.on_fault = on_fault
        self.faults = 0
        self.consecutive = 0
        self.last_error = None

    @property
    def healthy(self):
        return self.consecutive < self.max_consecutive_faults

    def _fault(self, why):
        self.faults += 1
        self.consecutive += 1
        self.last_error = why
        if self.on_fault is not None:
            self.on_fault(why, self.consecutive)
        return self.NEUTRAL

    def score(self, *args, **kwargs):
        try:
            v = self.inner.score(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - model code is untrusted; never let it kill the pipeline
            return self._fault(f"exception: {type(e).__name__}: {e}")
        try:
            v = float(v)
        except (TypeError, ValueError):
            return self._fault(f"non-numeric output: {v!r}")
        if not math.isfinite(v) or not 0.0 <= v <= 1.0:
            return self._fault(f"output out of range: {v!r}")
        self.consecutive = 0
        return v

    def __getattr__(self, name):  # pass through any extra attributes the simulator reads
        return getattr(self.inner, name)
