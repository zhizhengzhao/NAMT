set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: bash simulation/run.sh 'A|id|scene|material.yaml|target|job' NBEAM" >&2
  exit 64
fi

IFS='|' read -r TIER SCENE_ID SCENE MATERIAL_CONFIG TARGET_MATERIAL JOB_INDEX <<<"$1"
NBEAM=$2
if [[ "$TIER" != A || -z "$SCENE_ID" || -z "$SCENE" || -z "$MATERIAL_CONFIG" || -z "$TARGET_MATERIAL" || -z "$JOB_INDEX" ]]; then
  exit 64
fi

HERE=$(cd "$(dirname "$0")" && pwd)
CONFIG=$HERE/config
BUILD=${MUPOS_BUILD_DIR:-$HERE/build}
PYTHON=${PYTHON:-python}
EXPECTED_LAYER_Z=${EXPECTED_LAYER_Z:-415,215,-215,-415}
BASE_SEED=${BASE_SEED:-20260710}
WORK=$HERE/work/A/$SCENE
ROOT_DIR=$WORK/root_file
mkdir -p "$ROOT_DIR"
OUT=$ROOT_DIR/job_$JOB_INDEX.root
LOG=$ROOT_DIR/job_$JOB_INDEX.log
LOCK=$ROOT_DIR/.job_$JOB_INDEX.lock

exec 9>"$LOCK"
flock -n 9 || exit 75
if [[ -s "$OUT" ]]; then
  "$PYTHON" "$HERE/dataset.py" validate-root --file "$OUT" --expected-nbeam "$NBEAM" --expected-layer-z "$EXPECTED_LAYER_Z"
  exit 0
fi

TMP_OUT=$ROOT_DIR/.job_$JOB_INDEX.$$.root
TMP_LOG=$ROOT_DIR/.job_$JOB_INDEX.$$.log
MAC=$WORK/mac_$JOB_INDEX.$$.mac
trap 'rm -f "$MAC" "$TMP_OUT" "$TMP_LOG"' EXIT
SEED=$((BASE_SEED + SCENE_ID * 1000000 + JOB_INDEX))

{
  printf '/control/execute %s\n' "$HERE/Cry.mac"
  printf '/run/initialize\n'
  printf '/rlt/SetFileName %s\n' "$TMP_OUT"
  printf '/run/printProgress 100000\n'
  printf '/run/beamOn %s\n' "$NBEAM"
} >"$MAC"

TARGET_ENV=()
if [[ "$TARGET_MATERIAL" != - ]]; then
  TARGET_ENV=(
    "POTATO_SHAPE=polyfile"
    "POTATO_POLYFILE=$CONFIG/u.poly"
    "POTATO_HZ=100"
    "POTATO_X=0"
    "POTATO_Y=0"
    "POTATO_Z=0"
    "POTATO_MAT=$TARGET_MATERIAL"
  )
fi

unset MUPOS_BEAM_MOMENTUM MUPOS_EXTRA_MATERIAL MUPOS_BFIELD
unset MUPOS_SEED MUPOS_SCORING_REFERENCE MUPOS_VOLUME_CONFIG MUPOS_MATERIAL_CONFIG
while IFS= read -r NAME; do
  [[ -n "$NAME" ]] && unset "$NAME"
done < <(compgen -A variable POTATO || true)

env \
  "MUPOS_SEED=$SEED" \
  "MUPOS_SCORING_REFERENCE=module_center" \
  "MUPOS_VOLUME_CONFIG=$CONFIG/newrpc_readout.yaml:$CONFIG/newrpc.yaml:$CONFIG/newlayout4_cube400.yaml" \
  "MUPOS_MATERIAL_CONFIG=$CONFIG/newrpc_material.yaml:$CONFIG/$MATERIAL_CONFIG" \
  "${TARGET_ENV[@]}" \
  "$BUILD/muPos" "$MAC" >"$TMP_LOG" 2>&1

"$PYTHON" "$HERE/dataset.py" validate-root --file "$TMP_OUT" --expected-nbeam "$NBEAM" --expected-layer-z "$EXPECTED_LAYER_Z"
mv "$TMP_OUT" "$OUT"
mv "$TMP_LOG" "$LOG"
