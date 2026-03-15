#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SETS_JSON="${SETS_JSON:-dist/Lego Star Wars Database.json}"
MINIFIGS_JSON="${MINIFIGS_JSON:-dist/Lego-Star-Wars-Minifigure-Database.json}"
PARTS_JSON="${PARTS_JSON:-dist/parts/parts-catalog.json}"
MARKET_DETAILS_DIR="${MARKET_DETAILS_DIR:-dist/market-details}"

ITEM_TYPE="${ITEM_TYPE:-both}"
WORKERS="${WORKERS:-4}"
MAX_WORKERS="${MAX_WORKERS:-4}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-20}"
RETRIES="${RETRIES:-2}"
DELAY_SECONDS="${DELAY_SECONDS:-0.25}"
JITTER_SECONDS="${JITTER_SECONDS:-0.10}"
LIMIT="${LIMIT:-}"
START_INDEX="${START_INDEX:-0}"
MISSING_ONLY="${MISSING_ONLY:-0}"
BATCH_SIZE="${BATCH_SIZE:-250}"
BATCH_PAUSE_SECONDS="${BATCH_PAUSE_SECONDS:-12}"
MAX_BATCHES_PER_RUN="${MAX_BATCHES_PER_RUN:-0}"
RESUME="${RESUME:-1}"
PROGRESS_PATH="${PROGRESS_PATH:-dist/bootstrap-html-progress.json}"
ONLY_ITEM_NOS_FILE="${ONLY_ITEM_NOS_FILE:-}"
REBUILD_ARTIFACTS="${REBUILD_ARTIFACTS:-1}"

if [[ ! -f "${SETS_JSON}" ]]; then
  echo "Missing sets JSON: ${SETS_JSON}" >&2
  exit 1
fi

if [[ "${ITEM_TYPE}" == "both" || "${ITEM_TYPE}" == "minifig" || "${ITEM_TYPE}" == "all" ]]; then
  if [[ ! -f "${MINIFIGS_JSON}" ]]; then
    echo "Missing minifigs JSON: ${MINIFIGS_JSON}" >&2
    exit 1
  fi
fi

if (( WORKERS > MAX_WORKERS )); then
  echo "[BootstrapHTMLRelease] capping workers from ${WORKERS} to ${MAX_WORKERS} for BrickLink stability."
  WORKERS="${MAX_WORKERS}"
fi

json_count() {
  local path="$1"
  python3 - <<'PY' "$path"
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print(0)
    raise SystemExit(0)
data = json.loads(path.read_text(encoding="utf-8"))
print(len(data) if isinstance(data, list) else 0)
PY
}

determine_total_rows() {
  if [[ -n "${ONLY_ITEM_NOS_FILE}" ]]; then
    python3 - <<'PY' "${ONLY_ITEM_NOS_FILE}"
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print(0)
    raise SystemExit(0)
count = 0
for line in path.read_text(encoding="utf-8").splitlines():
    if line.strip():
        count += 1
print(count)
PY
    return
  fi
  local sets_count=0
  local minifigs_count=0
  local parts_count=0
  case "${ITEM_TYPE}" in
    set)
      sets_count="$(json_count "${SETS_JSON}")"
      echo "${sets_count}"
      ;;
    minifig)
      minifigs_count="$(json_count "${MINIFIGS_JSON}")"
      echo "${minifigs_count}"
      ;;
    part)
      parts_count="$(json_count "${PARTS_JSON}")"
      echo "${parts_count}"
      ;;
    both|all)
      sets_count="$(json_count "${SETS_JSON}")"
      minifigs_count="$(json_count "${MINIFIGS_JSON}")"
      if [[ "${ITEM_TYPE}" == "all" ]]; then
        parts_count="$(json_count "${PARTS_JSON}")"
      fi
      python3 - <<'PY' "${sets_count}" "${minifigs_count}" "${parts_count}"
import sys
values = [int(value) for value in sys.argv[1:] if str(value).strip()]
print(max(values) if values else 0)
PY
      ;;
    *)
      echo 0
      ;;
  esac
}

read_resume_start() {
  local default_start="$1"
  if [[ "${RESUME}" != "1" || ! -f "${PROGRESS_PATH}" ]]; then
    echo "${default_start}"
    return
  fi
  python3 - <<'PY' "${PROGRESS_PATH}" "${ITEM_TYPE}" "${default_start}"
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
item_type = sys.argv[2]
default_start = int(sys.argv[3])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print(default_start)
    raise SystemExit(0)
if payload.get("item_type") != item_type:
    print(default_start)
    raise SystemExit(0)
next_start = payload.get("next_start")
print(int(next_start) if isinstance(next_start, int) and next_start >= default_start else default_start)
PY
}

write_resume_state() {
  local next_start="$1"
  mkdir -p "$(dirname "${PROGRESS_PATH}")"
  python3 - <<'PY' "${PROGRESS_PATH}" "${ITEM_TYPE}" "${next_start}" "${WORKERS}" "${BATCH_SIZE}"
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "item_type": sys.argv[2],
    "next_start": int(sys.argv[3]),
    "workers": int(sys.argv[4]),
    "batch_size": int(sys.argv[5]),
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

TOTAL_ROWS="$(determine_total_rows)"
END_INDEX="${TOTAL_ROWS}"
if [[ -n "${LIMIT}" ]]; then
  END_INDEX="$(( START_INDEX + LIMIT ))"
  if (( END_INDEX > TOTAL_ROWS )); then
    END_INDEX="${TOTAL_ROWS}"
  fi
fi

CURRENT_START="$(read_resume_start "${START_INDEX}")"
if (( CURRENT_START >= END_INDEX )); then
  echo "[BootstrapHTMLRelease] nothing to do. start=${CURRENT_START} end=${END_INDEX}"
else
  BATCHES_RUN=0
  while (( CURRENT_START < END_INDEX )); do
    if (( MAX_BATCHES_PER_RUN > 0 && BATCHES_RUN >= MAX_BATCHES_PER_RUN )); then
      echo "[BootstrapHTMLRelease] reached MAX_BATCHES_PER_RUN=${MAX_BATCHES_PER_RUN}; pausing at start=${CURRENT_START}."
      break
    fi
    CURRENT_LIMIT="${BATCH_SIZE}"
    REMAINING="$(( END_INDEX - CURRENT_START ))"
    if (( CURRENT_LIMIT > REMAINING )); then
      CURRENT_LIMIT="${REMAINING}"
    fi

    echo "[BootstrapHTMLRelease] batch start=${CURRENT_START} limit=${CURRENT_LIMIT} workers=${WORKERS}"

    CMD=(
      python3 scripts/bootstrap_market_html_release.py
      --sets-json "${SETS_JSON}"
      --minifigs-json "${MINIFIGS_JSON}"
      --parts-json "${PARTS_JSON}"
      --market-details-dir "${MARKET_DETAILS_DIR}"
      --item-type "${ITEM_TYPE}"
      --workers "${WORKERS}"
      --timeout "${TIMEOUT_SECONDS}"
      --retries "${RETRIES}"
      --delay "${DELAY_SECONDS}"
      --jitter "${JITTER_SECONDS}"
      --start-index "${CURRENT_START}"
      --limit "${CURRENT_LIMIT}"
      --verbose
    )

    if [[ -n "${ONLY_ITEM_NOS_FILE}" ]]; then
      CMD+=(--only-item-nos-file "${ONLY_ITEM_NOS_FILE}")
    fi

    if [[ "${MISSING_ONLY}" == "1" ]]; then
      CMD+=(--missing-only)
    fi

    "${CMD[@]}"

    CURRENT_START="$(( CURRENT_START + CURRENT_LIMIT ))"
    BATCHES_RUN="$(( BATCHES_RUN + 1 ))"
    write_resume_state "${CURRENT_START}"

    if (( CURRENT_START < END_INDEX )); then
      if (( MAX_BATCHES_PER_RUN > 0 && BATCHES_RUN >= MAX_BATCHES_PER_RUN )); then
        echo "[BootstrapHTMLRelease] pausing after ${BATCHES_RUN} batch(es); next_start=${CURRENT_START}."
        break
      fi
      echo "[BootstrapHTMLRelease] cooling down for ${BATCH_PAUSE_SECONDS}s before next batch."
      sleep "${BATCH_PAUSE_SECONDS}"
    fi
  done
fi

if [[ "${REBUILD_ARTIFACTS}" == "1" ]]; then
  rm -rf "dist/chunks" "dist/market-details"

  python3 scripts/split_catalog_chunks.py \
    --sets-json "${SETS_JSON}" \
    --minifigs-json "${MINIFIGS_JSON}" \
    --output-dir "dist/chunks" \
    --manifest-path "dist/catalog-index.json" \
    --base-url "https://raw.githubusercontent.com/MattJones7416/LSW-Checklist-Database/refs/heads/main/dist" \
    --market-details-dir "dist/market-details" \
    --strip-market-detail-fields \
    --max-items-per-chunk 800

  python3 scripts/build_sync_artifacts.py \
    --manifest-path "dist/catalog-index.json" \
    --sync-state-path "dist/sync-state.json" \
    --sets-json "${SETS_JSON}" \
    --minifigs-json "${MINIFIGS_JSON}" \
    --delta-manifest-path "dist/catalog-delta-index.json" \
    --client-config-path "dist/client-config.json" \
    --market-price-seed-path "dist/market-price-seed.json" \
    --base-url "https://raw.githubusercontent.com/MattJones7416/LSW-Checklist-Database/refs/heads/main/dist" \
    --market-currency-code "${BRICKLINK_CURRENCY:-GBP}" \
    --verbose
else
  echo "[BootstrapHTMLRelease] skipping chunk/sync artifact rebuild."
fi

if (( CURRENT_START >= END_INDEX )); then
  rm -f "${PROGRESS_PATH}"
  echo "[BootstrapHTMLRelease] all batches complete."
else
  echo "[BootstrapHTMLRelease] progress saved to ${PROGRESS_PATH}; next_start=${CURRENT_START}."
fi

echo "[BootstrapHTMLRelease] complete."
