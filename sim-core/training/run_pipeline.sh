#!/usr/bin/env bash
# End-to-end: fetch -> convert -> synthesize -> mix -> train (RGB + IR) -> evaluate -> export.
# Runs inside the training container (see Dockerfile). All state lives under $DATA.
#
#   COMMERCIAL_ONLY=1 AOT_FLIGHTS=200 SYN_N=20000 training/run_pipeline.sh
#
# Manually obtained datasets (Drone-vs-Bird, Anti-UAV, DUT, Det-Fly, CST, ...) go in
# $DATA/raw/<dataset_id>/ before running; see `python -m training.fetch --list`.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA:-/data}"
COMMERCIAL_ONLY="${COMMERCIAL_ONLY:-0}"
AOT_FLIGHTS="${AOT_FLIGHTS:-100}"
SYN_N="${SYN_N:-20000}"
SYN_RATIO="${SYN_RATIO:-0.3}"
EPOCHS="${EPOCHS:-}"
CO_FLAG=()
[[ "$COMMERCIAL_ONLY" == "1" ]] && CO_FLAG=(--commercial-only)

log() { printf '\n=== %s  %s\n' "$(date -u +%H:%M:%S)" "$*"; }

log "1/6 fetch direct-access datasets"
python -m training.fetch --all-direct --out "$DATA/raw" --max-sequences "$AOT_FLIGHTS" "${CO_FLAG[@]}" || \
  echo "some fetches failed; continuing with what is present"
python -m training.fetch --list

log "2/6 convert everything present in $DATA/raw"
python -m training.convert_all --raw "$DATA/raw" --out "$DATA/converted" "${CO_FLAG[@]}" || \
  echo "some conversions failed; see messages above"

log "3/6 synthetic data"
[[ -d "$DATA/synthetic/images" ]] || python generate_dataset.py --n "$SYN_N" --out "$DATA/synthetic"

CONV_ARGS=()
for d in "$DATA"/converted/*/; do [[ -f "$d/manifest.jsonl" ]] && CONV_ARGS+=(--converted "$d"); done
if [[ ${#CONV_ARGS[@]} -eq 0 ]]; then
  echo "no converted real data: refusing to train on synthetic data only (there would be no real test set)"; exit 1
fi

log "4/6 build RGB and IR mixes"
STAMP="$(date -u +%Y%m%d_%H%M)"
python -m training.build_mix "${CONV_ARGS[@]}" --synthetic "$DATA/synthetic" --out "$DATA/mix_rgb_$STAMP" \
  --modality rgb --synthetic-ratio "$SYN_RATIO" --tile 1280 --augment-frac 0.25 "${CO_FLAG[@]}"
python -m training.build_mix "${CONV_ARGS[@]}" --out "$DATA/mix_ir_$STAMP" --modality ir --augment-frac 0.2 \
  "${CO_FLAG[@]}" || echo "no IR data present; skipping IR model"

log "5/6 train + evaluate + export"
SETS=()
[[ -n "$EPOCHS" ]] && SETS+=(--set "epochs=$EPOCHS")
python -m training.train --config training/configs/rgb.yaml --set "data=$DATA/mix_rgb_$STAMP/data.yaml" \
  --project "$DATA/runs" "${SETS[@]}"
if [[ -f "$DATA/mix_ir_$STAMP/data.yaml" ]] && [[ -n "$(ls -A "$DATA/mix_ir_$STAMP/images/train" 2>/dev/null)" ]]; then
  python -m training.train --config training/configs/ir.yaml --set "data=$DATA/mix_ir_$STAMP/data.yaml" \
    --project "$DATA/runs" "${SETS[@]}"
fi

log "6/6 collect models"
mkdir -p "$DATA/models"
for n in rgb ir; do
  [[ -f "$DATA/runs/$n/$n.onnx" ]] && cp "$DATA/runs/$n/$n.onnx" "$DATA/models/${n}_$STAMP.onnx" \
    && cp "$DATA/runs/$n/model_card.md" "$DATA/models/${n}_$STAMP.model_card.md"
done
ls -la "$DATA/models"
echo "done. Copy \$DATA/models back and plug in with training.onnx_detector.make_sim_classifier()."
