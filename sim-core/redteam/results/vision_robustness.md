# Vision robustness benchmark (CPU reference detector (not the product model))

24 drone frames per cell (1/3 thermal), rendered by render.py; match = IoU≥0.1 or centre-in-box. Recall = share of drones still detected. Reference detector trained on 300 rendered frames, once clean and once with harden_augment(). Absolute numbers say little about a trained YOLO; the per-attack drops and the recipe's effect are the point.

## reference, clean training

Clean recall 42%, false drone boxes/frame 0.08

| Attack | sev 1 | sev 2 | sev 3 | sev 4 | sev 5 |
|---|---|---|---|---|---|
| glare | 38% | 33% | 21% | 17% | 4% |
| laser_dazzle | 19% | 0% | 0% | 0% | 0% |
| fog | 54% | 46% | 46% | 42% | 25% |
| rain | 38% | 38% | 25% | 25% | 25% |
| snow | 31% | 56% | 31% | 12% | 25% |
| jpeg | 46% | 50% | 29% | 38% | 38% |
| motion_blur | 33% | 33% | 33% | 33% | 17% |
| defocus | 33% | 38% | 38% | 25% | 17% |
| low_light_noise | 42% | 21% | 12% | 12% | 4% |
| ir_bloom | 50% | 75% | 75% | 75% | 38% |
| hot_decoys | 62% | 50% | 50% | 62% | 62% |
| camo_sky | 38% | 19% | 31% | 12% | 6% |
| bird_shell | 42% | 50% | 42% | 42% | 42% |
| adv_patch_placeholder | 83% | 79% | 62% | 50% | 58% |

## reference, hardening recipe

Clean recall 50%, false drone boxes/frame 0.00

| Attack | sev 1 | sev 2 | sev 3 | sev 4 | sev 5 |
|---|---|---|---|---|---|
| glare | 38% | 25% | 29% | 17% | 0% |
| laser_dazzle | 6% | 0% | 0% | 0% | 0% |
| fog | 38% | 42% | 25% | 29% | 21% |
| rain | 31% | 25% | 19% | 6% | 19% |
| snow | 25% | 31% | 12% | 6% | 0% |
| jpeg | 54% | 33% | 21% | 25% | 33% |
| motion_blur | 29% | 21% | 12% | 8% | 4% |
| defocus | 38% | 25% | 29% | 12% | 8% |
| low_light_noise | 42% | 29% | 12% | 8% | 4% |
| ir_bloom | 62% | 62% | 50% | 50% | 50% |
| hot_decoys | 62% | 38% | 50% | 75% | 50% |
| camo_sky | 25% | 19% | 25% | 12% | 0% |
| bird_shell | 46% | 42% | 25% | 29% | 29% |
| adv_patch_placeholder | 58% | 50% | 29% | 17% | 21% |

## Effect of the hardening recipe (recall, mean over severities 1-5)

| Attack | reference, clean training | reference, hardening recipe | change |
|---|---|---|---|
| glare | 22% | 22% | -1 pts |
| laser_dazzle | 4% | 1% | -2 pts |
| fog | 42% | 31% | -12 pts |
| rain | 30% | 20% | -10 pts |
| snow | 31% | 15% | -16 pts |
| jpeg | 40% | 33% | -7 pts |
| motion_blur | 30% | 15% | -15 pts |
| defocus | 30% | 22% | -7 pts |
| low_light_noise | 18% | 19% | +1 pts |
| ir_bloom | 62% | 55% | -7 pts |
| hot_decoys | 57% | 55% | -2 pts |
| camo_sky | 21% | 16% | -5 pts |
| bird_shell | 43% | 34% | -9 pts |
| adv_patch_placeholder | 67% | 35% | -32 pts |
