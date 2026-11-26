# Training the camera model on real data

This folder turns public drone/bird datasets, licensed stock footage, your own recordings
and the simulator's synthetic frames into a trained detector (`drone`, `drone_payload`,
`bird`, `other`), scores it on **real footage only**, and exports ONNX that plugs straight
into the simulator.

The pipeline was built and unit-tested (44 tests) in a workspace with no GPU and no dataset
access. Nothing has been trained yet; the first real run happens on your cloud GPU.

## One-command run

```bash
# on a GPU machine, from sim-core/
docker build -f training/Dockerfile -t dronesim-train .
docker run --gpus all --shm-size=16g -v /data:/data dronesim-train training/run_pipeline.sh
# commercial-safe model: add -e COMMERCIAL_ONLY=1
```

`run_pipeline.sh` runs these six steps:

1. Fetches every dataset with unattended access.
2. Converts whatever is in `/data/raw/<dataset_id>/`.
3. Generates synthetic frames.
4. Builds the RGB and IR training mixes.
5. Trains with Ultralytics 8.3.172, evaluates on the real-only test split and exports ONNX.
6. Copies `rgb_*.onnx`, `ir_*.onnx` and their model cards to `/data/models/`.

Launch helpers:
- AWS: `training/cloud/aws_launch.sh`
- GCP: `training/cloud/gcp_launch.sh`
- Lambda: `training/cloud/lambda.md`

## Licensing: read this before you ship a model

Every dataset in `training/datasets.yaml` carries a `commercial_ok` tag:

| commercial_ok | Count | Meaning |
|---|---|---|
| yes | 7 | Licence verified to allow commercial use. Includes your own footage and synthetic data. |
| no | 3 | Research / non-commercial only. |
| unclear | 12 | No dataset licence found, only a code or paper licence, or the images came from third parties. |
| permission_required | 1 | The provider's terms forbid this use without written permission (Pexels). |

- **Prototype on everything.** That is the default.
- **Ship only a model trained with `COMMERCIAL_ONLY=1`.** That filter keeps only the "yes" sources.
- **Unclear sources need clearance.** Each one needs an email to the authors, or legal review, before it can count as "yes".
- **Model cards flag uncleared models.** Each model card states whether its weights are commercially cleared, and `mix_report.md` lists every source used and the attribution it requires.

The commercial-safe set is thin today. AOT is large but grayscale and air-to-air; the others
are USC drone clips, the DroneRF and SWINSEG side datasets, synthetic data and **your own
footage**. Your own night recordings at real sites, with and without packages, are the most
valuable data you can add.

I did **not** add YouTube or social-media scraping. Their terms prohibit it, and footage you
have no rights to is a liability in a product you sell. Stock video is handled as follows:

- **Pixabay** is supported with a hard volume cap, 24-hour request caching and per-clip
  attribution. Its API rules forbid "systematic mass downloads", so you must pass
  `--accept-pixabay-terms`.
- **Pexels** fetching refuses to run. Section 8 of its Terms of Service prohibits automated
  collection "for machine learning purposes" without permission.
- **Stock frames are backgrounds only.** They are used as negatives only when the search
  query was a pure background ("night sky", "stadium night"). Clips that may contain drones
  or birds go to `to_label.txt` for a human to label.

## Datasets

All entries were checked against each dataset's own page, paper or repository on 2026-11-16. Anything not confirmed is marked unverified.

| Dataset | Modality | Size | Licence | Commercial | Access |
|---|---|---|---|---|---|
| [AOT (Amazon Prime Air)](https://registry.opendata.aws/airborne-object-tracking/) | grayscale | 4,943 flights, 5.9M+ images, 3.3M+ boxes | CDLA-Permissive-1.0 | yes | direct (public S3) |
| [USC Drone](https://mcl.usc.edu/?p=10642) | RGB | 30 × ~1 min clips, 1080p | USC permissive notice | yes | direct (Hugging Face) |
| [DUT Anti-UAV](https://www.emergentmind.com/topics/dut-anti-uav-dataset) | RGB | 10,000 images + 20 tracking sequences | repo reported Apache-2.0; dataset terms unverified | unclear | manual |
| [Anti-UAV 300/410/600](https://github.com/ZhaoJ9014/Anti-UAV) | RGB + IR | 318 / 410 / 600 sequences | repo MIT; dataset terms not stated | unclear | manual (Drive/Baidu/ModelScope) |
| [Drone-vs-Bird](https://github.com/wosdetc/challenge) | RGB | 77 sequences / ~105K frames | data usage agreement, research only | no | email request |
| [Halmstad multi-sensor](https://arxiv.org/pdf/2111.01888) | IR + RGB + audio | 365 IR + 285 visible clips, 90 audio clips | article CC BY 4.0; data licence not confirmed | unclear | direct (Zenodo) |
| [KIOS UAV](https://www.kios.ucy.ac.cy/evai/datasets/unmanned-aerial-vehicles/) | RGB | 1,535 images | not stated | unclear | direct (Zenodo) |
| [Det-Fly](https://shiyuzhao.westlake.edu.cn/2020IEEEye.pdf) | RGB 4K | 13,271 images | not stated | unclear | manual |
| [MAV-VID](https://arxiv.org/pdf/2103.13933) | RGB | 64 videos / 40,232 images | not stated | unclear | manual |
| [CST Anti-UAV](https://arxiv.org/html/2507.23473v2) | thermal | 220 sequences, 240K+ boxes | not stated | unclear | manual |
| [RGBT-Tiny](https://arxiv.org/html/2406.14482v2) | RGB + thermal | 115 sequences, ~93K frames (drone, plane, ...) | unclear (paper page shows CC BY-NC-SA) | unclear | manual |
| [MMAUD](https://ntu-aris.github.io/MMAUD) | RGB, audio, radar, lidar | rosbags, rooftop + carpark | CC BY-NC-SA 4.0 | no | manual |
| [FBD-SV-2024](https://figshare.com/articles/dataset/FBD-SV-2024/27377595) | RGB birds | 483 clips, 28,366 bird boxes | research only, not commercial | no | direct (figshare) |
| [AOD 4](https://data.mendeley.com/datasets/cd5z895tr2/1) | RGB | 22,516 images; airplane, helicopter, drone, bird | CC BY 4.0, image provenance unknown | unclear | manual |
| [Amateur drone (Mendeley)](https://data.mendeley.com/datasets/zcsj2g2m4c) | RGB | 4,000+ images | CC BY 4.0, but images scraped from YouTube/Google | unclear | manual |
| [DroneRF](https://data.mendeley.com/datasets/f4c2b4n755/1) | RF | 227 segments, 3 drones | CC BY 4.0 | yes | manual (RF sensor model) |
| [SWINSEG](https://ieee-dataport.org/node/1386) | night sky | 115 patches | CC BY 4.0 | yes | IEEE DataPort subscription |
| [Drexel long-range](https://research.coe.drexel.edu/ece/imaple/?p=453) | RGB | not stated | CDLA-Permissive-2.0 | yes | on hold (export control) |
| Drone audio (Al-Emadi) | audio | ~1,332 clips | not found | unclear | manual |
| [Pixabay video API](https://pixabay.com/api/docs/) | RGB plates | capped | Pixabay licence + API rules | unclear | direct, capped |
| [Pexels](https://pexels.com/terms-of-service) | RGB plates | n/a | ToS s.8 forbids ML collection | permission_required | refused |

Coverage by context:
- **Night and thermal:** Anti-UAV, CST Anti-UAV, RGBT-Tiny and Halmstad IR.
- **Long range and tiny targets:** AOT, Drone-vs-Bird and CST.
- **Birds as confusers:** FBD-SV-2024, Halmstad, Drone-vs-Bird, AOD 4 and AOT.
- **Urban and complex backgrounds:** Det-Fly, DUT and CST.
- **Military-style scenes:** only MMAUD V2/V3, which the authors describe as designed for warfare simulation.
- **Prisons, stadiums and malls:** no public dataset covers these sites at night with packages. Only your own recordings can fill that gap.

## Your own footage

Public datasets don't cover prisons, stadiums or malls at night, so your own recordings matter most.
Clips go in as plain video files:

```bash
# from sim-core/; notes.csv says what is in each stretch of each clip
python3 -m training.import_video clips/ --out /data/raw/own_footage --notes notes.csv --model models/rgb.onnx
# fix the boxes listed in review_queue.csv (CVAT or Label Studio, YOLO format), then:
python3 -m training.import_video --accept /data/raw/own_footage
python3 -m training.build_mix --converted /data/raw/own_footage --converted ... --out /data/mix_rgb
```

```csv
clip,start_s,end_s,label
flight03.mp4,0,12.5,empty
flight03.mp4,12.5,41,drone
gulls_dusk.mov,0,90,bird
```

| Frame | What happens |
|---|---|
| In a `bird`, `empty` or `other` stretch, no model box | Trained on as a negative. You said what's in it. |
| Model found a box | Written with that box, marked unreviewed. `build_mix.py` skips it until `--accept`. |
| Model box inside a `bird` or `empty` stretch | Same, and listed first in the review queue: either the notes or the model is wrong. |
| In a `drone` stretch with no box, or a clip with no notes | Review queue only. It never enters training as empty sky. |

Clips are handled as untrusted input. Only video extensions are read and symlinks are skipped. File size,
resolution and duration are capped. Output names come from a hash of the clip's content, not its file
name, and the same clip under two names is imported once. Frames are re-encoded from pixels, so GPS and
other container metadata are not copied. The importer cannot blur faces, so keep people out of the
footage. Video decoding runs FFmpeg on files you didn't make: run imports inside `training/Dockerfile`.

The phone app's label export has its own importer: `python3 -m training.import_liveapp export.json --out /data/raw/own_footage`.

## Pieces

| File | What it does |
|---|---|
| `training/datasets.yaml` | The catalog above, machine-readable. |
| `training/fetch.py` | Downloads from Zenodo, figshare, public S3 (AOT, `--max-sequences`), Hugging Face and the Pixabay API. Downloads are resumable, checked against published md5s, rate-limited and restricted to HTTPS on allowlisted hosts. Gated sources get exact instructions instead. |
| `training/convert/` | One converter per format: VOC XML, COCO JSON, YOLO, Drone-vs-Bird txt, OTB-style rect txt, Anti-UAV json, AOT groundtruth.json, Halmstad (via `export_halmstad_labels.m`). Each writes YOLO plus a manifest row carrying source and licence. A format mismatch stops with a clear error instead of producing empty labels. |
| `training/import_video.py` | Plain clips to YOLO frames: samples at `--fps`, drops still frames, pre-labels with an ONNX model, and keeps unreviewed boxes out of training (see "Your own footage"). |
| `training/convert_all.py` | Converts every raw folder present, with per-dataset defaults. |
| `training/build_mix.py` | Mixes real and synthetic data at a set ratio. Splits **by sequence** to prevent leakage, keeps test and val **real-only**, drops near-duplicate frames (dHash plus box match), tiles 4K or AOT frames for tiny targets, adds night/IR/blur/noise/glare/JPEG variants, has a `--commercial-only` filter, and writes `mix_report.md`. |
| `training/train.py` + `configs/` | Trains with Ultralytics YOLO11-s at 1280 px (RGB) or 960 px (IR); the P2-head variant (`--set model=yolov8s-p2.yaml`) is for targets under 16 px. Then evaluates on the real test split, exports ONNX and writes a model card. `--dry-run` prints the plan. |
| `training/evaluate_real.py` | Reports AP@0.5 per class and for "any drone", recall at 90% precision, AP **by target size in pixels**, per-source results, and the false-drone rate on frames with no drone and on bird-only frames. |
| `training/onnx_detector.py` | onnxruntime inference with letterboxing, NMS and optional tiled inference. `make_sim_classifier(path)` returns the simulator's `RenderedModelClassifier`, so `evaluate.py` runs on the real model. |
| `training/temporal.py` | **Experimental.** A track-level drone-vs-bird classifier using wing-beat oscillation, size stability and path smoothness. Tested only on synthetic tracks. |
| `training/Dockerfile`, `run_pipeline.sh`, `cloud/` | Container (unprivileged user; the build fails if the tests fail), orchestration and launch scripts. |

## Cost and time (estimates)

| Option | Price (checked 2026-11-16, may change) | Notes |
|---|---|---|
| AWS g6.xlarge (1× L4 24 GB) | ~$0.80/hr on demand, ~$0.40–0.72 spot (us-east-1, [doit.com](https://www.doit.com/compute/spot/us-east-1/g6.xlarge)) | Good default. |
| Lambda 1× H100 PCIe | ~$3.29/hr (third-party figure, [Spheron blog](https://www.spheron.network/blog/lambda-cloud-h100-pricing-2026/)) | About 3–4× faster than an L4 for this job. |
| GCP g2-standard-8 (L4) / a2-highgpu-1g (A100) | not verified: check the console | |

**Rough time estimate (not measured):** YOLO11-s at 1280 px on about 50K images for 150
epochs is on the order of 1–2 days on one L4, or under a day on an A100/H100. Do a first
pass with `EPOCHS=40` and `--set imgsz=960` to check the data before paying for a full run.
Disk: allow 500 GB or more if you pull many AOT flights.

## What is verified vs. not

- **Verified on primary pages:**
  - The licence, size and access route of each dataset where the table says so.
  - The Pexels and Pixabay terms quoted above.
  - That the pinned versions exist: Ultralytics 8.3.172, sahi 0.11.35, onnxruntime-gpu 1.20.1, and the `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` image.
- **Unverified:**
  - **Annotation layouts.** The exact layouts for Anti-UAV json, AOT groundtruth.json, DUT, Det-Fly, CST, RGBT-Tiny and MAV-VID come from public starter kits and conventions, not from the datasets themselves. Each converter checks the keys it reads and stops with a clear error on the first mismatch.
  - **Cloud images.** The AWS DLAMI SSM parameter name and the GCP Deep Learning VM image family.
  - **Zenodo files.** Whether Zenodo record 5500576 (Halmstad) exposes its files through the public API.
- **Tested here:** 44 unit tests on synthetic fixtures covering:
  - catalog validation;
  - archive path-traversal, symlink and size-bomb rejection;
  - URL allowlisting, including redirects;
  - md5 checking;
  - the refusals for Pexels and Pixabay without terms;
  - every converter;
  - sequence-level splitting with no leakage;
  - the commercial-only filter, de-duplication and tiling;
  - the evaluation maths;
  - ONNX output decoding and the simulator adapter (with a mocked runtime);
  - config validation and the temporal classifier.
- **Not tested here:** actual downloads, actual training and ONNX export. This workspace has no network access to the dataset hosts and no GPU.

## Plugging the trained model into the simulator

```python
from training.onnx_detector import make_sim_classifier
params = {"classifier": make_sim_classifier("/data/models/rgb_20261120_0400.onnx", imgsz=1280)}
# pass params where evaluate.py / sim.py build SensorSuite (see README, "Swapping in a real vision model")
```
