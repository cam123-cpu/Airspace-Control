"""Radar micro-Doppler observation model (port of SENS2.radar in sim/sensors2.js).

The sensor side draws a classifier output from the *physical* signature of the true object
(rotor blade flash vs. no blade flash); the fusion side only ever sees the resulting
log-likelihood ratio, never the truth.

Physics and anchors (see sim/SENSOR_MODELS.md for sources):
- Phantom-class blade flash ~140 Hz (2 blades x 70 rev/s), tip speed ~53 m/s -> ~7 kHz bandwidth at X-band.
- Rotor returns sit ~20-25 dB below the body return (review literature) -> micro-Doppler SNR = body SNR - 22.5 dB.
- A rotating surveillance radar dwells ~25-50 ms per beam: long enough to see >= 2 blade-flash periods
  (14 ms) but not a full bird wingbeat (200-330 ms). So the test is "blade flash present or absent";
  its reliability is set by the rotor hypothesis for any target (absence at high SNR = evidence of non-drone).
- Classifier ceiling 0.95 (lab CNNs report 94-98 %; derated for field use, assumed).
"""
from __future__ import annotations

import math

BLADE_FLASH_HZ = 140.0          # Phantom-class [MDP]
ROTOR_BELOW_BODY_DB = 22.5      # props 20-25 dB weaker than body [RR19-review]
PC_MAX = 0.95                   # field-derated classifier ceiling (assumed)
SNR_AT_R50_DB = 13.0            # body SNR where this package's radar Pd model reaches 50 % (Swerling-like, assumed)


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def body_snr_db(rcs: float, R: float) -> float:
    """Body SNR consistent with SensorSuite.pd(): R50 = 1500 m * (rcs/0.01)^0.25 and SNR ~ R^-4."""
    R50 = 1500.0 * (max(rcs, 1e-6) / 0.01) ** 0.25
    return SNR_AT_R50_DB + 40.0 * math.log10(R50 / max(R, 1.0))


def p_correct(snr_db: float, dwell_s: float = 0.025) -> float:
    """Probability the blade-flash test is right, for any target (rotor hypothesis sets the reliability)."""
    md_snr = snr_db - ROTOR_BELOW_BODY_DB
    need = 2.0 / BLADE_FLASH_HZ                         # two blade-flash periods
    dwell_f = min(max(dwell_s / need, 0.0), 1.0) * (1.0 - math.exp(-3.0 * dwell_s / need))
    snr_f = _logistic((md_snr + 3.0) / 2.5)            # 50 % of the way at -3 dB micro-Doppler SNR (assumed)
    return 0.5 + (PC_MAX - 0.5) * snr_f * dwell_f


def observe(is_rotorcraft: bool, snr_db: float, rng, dwell_s: float = 0.025):
    """One look. Returns the signed LLR (positive = 'blade flash seen' = drone-like) or None if uninformative."""
    pc = p_correct(snr_db, dwell_s)
    if pc < 0.52:
        return None
    says_rotor = (rng.random() < pc) == bool(is_rotorcraft)
    llr = math.log(pc / (1.0 - pc))
    return llr if says_rotor else -llr
