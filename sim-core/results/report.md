# Monte Carlo evaluation: 100 simulated nights

Alert threshold P(drone) ≥ 0.9; stand-in camera classifier strength 4.0; 3484 episodes; runtime 478 s.

## Headline

- **False alarms per site-night:** 0.01 (1% of nights had at least one)
- **False alarm sources:** {'bird': 1}

| Drone group | Runs | Alerted | Alerted before drop | Median takeoff→alert | Median warning before drop | Drop-zone error at alert / at drop (median) | Pilot located | Pilot error (median) |
|---|---|---|---|---|---|---|---|---|
| All drones | 82 | 98% | 94% | 71.0 s | 78.0 s | 51.8 m / 0.7 m | 87% | 171.5 m |
| RF link, no Remote ID | 50 | 96% | 90% | 82.5 s | 71.0 s | 60.9 m / 0.7 m | 100% | 251.8 m |
| Remote ID broadcasting | 21 | 100% | 100% | 1.0 s | 138.5 s | 155.3 m / 0.7 m | 100% | 0.6 m |
| Dark (no RF, no RID) | 11 | 100% | 100% | 133.5 s | 19.0 s | 1.4 m / 0.9 m | 0% | – |

## Threshold trade-off (same nights)

| Threshold | False alarms / night | Drones alerted before drop | Dark drones alerted before drop |
|---|---|---|---|
| 0.5 | 0.29 | 100% | 100% |
| 0.6 | 0.11 | 99% | 100% |
| 0.7 | 0.08 | 98% | 100% |
| 0.8 | 0.05 | 96% | 100% |
| 0.85 | 0.02 | 95% | 100% |
| 0.9 | 0.01 | 94% | 100% |
| 0.95 | 0.01 | 90% | 73% |
| 0.98 | 0.00 | 82% | 9% |
| 0.99 | 0.00 | 80% | 0% |

## How good does the vision model need to be?

Classifier strength sweep at threshold 0.9 (40 nights each). Strength ≈ how separable drone vs bird logits are at good pixel counts.

| Classifier strength | False alarms / night | All drones before drop | Dark drones before drop |
|---|---|---|---|
| 1.0 | 0.00 | 79% | 0% |
| 2.0 | 0.00 | 100% | 100% |
| 4.0 | 0.01 | 94% | 100% |
| 6.0 | 0.03 | 96% | 100% |

Charts: `tradeoff.png`, `pilot_error_cdf.png`, `time_to_alert.png`.
