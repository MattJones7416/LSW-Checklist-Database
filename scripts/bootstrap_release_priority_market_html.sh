#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SETS_JSON="${SETS_JSON:-dist/Lego Star Wars Database.json}"
TARGETS_PATH="${TARGETS_PATH:-dist/bootstrap-release-targets.txt}"
TARGETS_METADATA_PATH="${TARGETS_METADATA_PATH:-dist/bootstrap-release-targets.json}"
MAX_TOTAL="${MAX_TOTAL:-1500}"
POPULAR_THEME_LIMIT="${POPULAR_THEME_LIMIT:-90}"
ACTIVE_FALLBACK_LIMIT="${ACTIVE_FALLBACK_LIMIT:-500}"
PER_THEME_CAP="${PER_THEME_CAP:-40}"

python3 scripts/select_release_market_targets.py \
  --sets-json "${SETS_JSON}" \
  --output-path "${TARGETS_PATH}" \
  --metadata-path "${TARGETS_METADATA_PATH}" \
  --max-total "${MAX_TOTAL}" \
  --per-theme-cap "${PER_THEME_CAP}" \
  --popular-theme-limit "${POPULAR_THEME_LIMIT}" \
  --active-fallback-limit "${ACTIVE_FALLBACK_LIMIT}"

TARGET_COUNT="$(python3 - <<'PY' "${TARGETS_PATH}"
import sys
from pathlib import Path
path = Path(sys.argv[1])
count = 0
if path.exists():
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            count += 1
print(count)
PY
)"

echo "[ReleaseTargets] count=${TARGET_COUNT} file=${TARGETS_PATH}"

if [[ "${TARGET_COUNT}" == "0" ]]; then
  echo "[ReleaseTargets] no targets selected."
  exit 0
fi

export ITEM_TYPE="${ITEM_TYPE:-set}"
export WORKERS="${WORKERS:-2}"
export BATCH_SIZE="${BATCH_SIZE:-150}"
export BATCH_PAUSE_SECONDS="${BATCH_PAUSE_SECONDS:-20}"
export DELAY_SECONDS="${DELAY_SECONDS:-0.40}"
export JITTER_SECONDS="${JITTER_SECONDS:-0.15}"
export ONLY_ITEM_NOS_FILE="${TARGETS_PATH}"
export PROGRESS_PATH="${PROGRESS_PATH:-dist/bootstrap-html-priority-progress.json}"

bash scripts/bootstrap_sets_minifigs_market_html_release.sh
