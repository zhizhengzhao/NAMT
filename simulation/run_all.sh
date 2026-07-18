set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: bash simulation/run_all.sh FIRST_JOB COUNT NBEAM" >&2
  exit 64
fi

FIRST_JOB=$1
COUNT=$2
NBEAM=$3
NPROC=${NPROC:-24}
HERE=$(cd "$(dirname "$0")" && pwd)
LAST_JOB=$((FIRST_JOB + COUNT - 1))
JOBS=$HERE/work/jobs_$FIRST_JOB-$LAST_JOB.list
mkdir -p "$HERE/work"
: >"$JOBS"

while IFS='|' read -r ID SCENE MATERIAL TARGET; do
  for ((I=FIRST_JOB; I<=LAST_JOB; ++I)); do
    printf 'A|%s|%s|%s|%s|%s\n' "$ID" "$SCENE" "$MATERIAL" "$TARGET" "$I" >>"$JOBS"
  done
done <"$HERE/scenes.tsv"

xargs -a "$JOBS" -d $'\n' -P "$NPROC" -I{} bash "$HERE/run.sh" "{}" "$NBEAM"
