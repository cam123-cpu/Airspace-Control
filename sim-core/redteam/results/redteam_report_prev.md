# Red-team results: 20 seeds per scenario, threshold 0.9

v1 = original fusion (`hardened=False`); H = hardened fusion (default). Same seeds, same sensors, same stand-in camera classifier. Runtime 1311 s.

| Scenario | Payload drones alerted before drop (v1 → H) | Median takeoff→alert (v1 → H) | False drone alerts / run (v1 → H) | Attack flagged to operator (v1 → H) | Pilot error, median (v1 → H) |
|---|---|---|---|---|---|
| baseline_rf | 85% → **95%** | 79 s → 68 s | 0.20 → **0.05** | n/a → n/a | 252 m → 127 m |
| baseline_dark | 85% → **85%** | 126 s → 127 s | 0.00 → **0.00** | n/a → n/a | – → – |
| rid_ghost | – → **–** | – → – | 4.50 → **0.00** | 0% → 100% | – → – |
| rid_op_misdirect | 100% → **100%** | 0 s → 8 s | 0.25 → **0.05** | 0% → 100% | 761 m → 117 m |
| rid_pos_offset | 85% → **95%** | 86 s → 78 s | 21.30 → **0.05** | 0% → 100% | 801 m → 98 m |
| rf_decoy_downlink | 85% → **85%** | 114 s → 120 s | 0.45 → **0.10** | 0% → 70% | – → – |
| ctrl_clone | 85% → **95%** | 78 s → 60 s | 0.20 → **0.15** | 0% → 75% | 946 m → 328 m |
| decoy_saturation | 85% → **75%** | 98 s → 104 s | 0.50 → **0.15** | n/a → n/a | – → – |
| swarm | 85% → **89%** | 111 s → 108 s | 0.00 → **0.05** | n/a → n/a | – → – |
| terrain_masked | 50% → **60%** | 139 s → 134 s | 0.00 → **0.00** | n/a → n/a | – → – |
| building_shadow | 0% → **0%** | – → – | 0.00 → **0.00** | 0% → 10% | – → – |
| bird_mimicry | 0% → **0%** | – → – | 0.00 → **0.00** | n/a → n/a | – → – |
| dive_drop | 70% → **85%** | 84 s → 83 s | 0.00 → **0.00** | n/a → n/a | – → – |
| tether_drop | 90% → **100%** | 72 s → 64 s | 0.10 → **0.05** | n/a → n/a | 196 m → 141 m |
| acoustic_flood | 85% → **85%** | 126 s → 133 s | 0.00 → **0.00** | 0% → 100% | – → – |
| camera_dazzle | 0% → **90%** | 176 s → 126 s | 0.00 → **0.00** | 0% → 100% | – → – |
| radar_surge | 85% → **85%** | 128 s → 125 s | 0.00 → **0.00** | 0% → 100% | – → – |
| storm | 85% → **85%** | 99 s → 87 s | 0.05 → **0.00** | n/a → n/a | 290 m → 102 m |

## Attack-free control nights (20 nights, ~34 bird / aircraft / bag / truck events each)

| Fusion | False drone alerts / night | Sensor-degraded alarms / night | Spoof indicators / night |
|---|---|---|---|
| v1 | 0.00 | – | – |
| Hardened | 0.00 | 0.00 | 0.00 |

## False-alarm sources (hardened)

- baseline_rf: {'bird': 1}
- rid_op_misdirect: {'bird': 1}
- rid_pos_offset: {'bird': 1}
- rf_decoy_downlink: {'bird': 2}
- ctrl_clone: {'bird': 3}
- decoy_saturation: {'bird': 1, 'misplaced_alert': 2}
- swarm: {'bird': 1}
- tether_drop: {'bird': 1}

Decoy drones alerted (they are real drones, so alerting is correct): decoy_saturation v1 90% / H 90%
