#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SETS_JSON="${SETS_JSON:-dist/Lego Star Wars Database.json}"
MINIFIGS_JSON="${MINIFIGS_JSON:-dist/Lego-Star-Wars-Minifigure-Database.json}"
PARTS_JSON="${PARTS_JSON:-dist/parts/parts-catalog.json}"
MARKET_STATE_JSON="${MARKET_STATE_JSON:-dist/market-sync-state.json}"
CATALOG_SYNC_STATE_JSON="${CATALOG_SYNC_STATE_JSON:-dist/sync-state.json}"

BRICKLINK_CURRENCY="${BRICKLINK_CURRENCY:-GBP}"
BRICKLINK_FALLBACK_CURRENCIES="${BRICKLINK_FALLBACK_CURRENCIES:-USD,EUR,GBP}"

SETS_PER_PASS="${SETS_PER_PASS:-4500}"
MINIFIGS_PER_PASS="${MINIFIGS_PER_PASS:-4000}"
SETS_MAX_API_CALLS_PER_PASS="${SETS_MAX_API_CALLS_PER_PASS:-30000}"
MINIFIGS_MAX_API_CALLS_PER_PASS="${MINIFIGS_MAX_API_CALLS_PER_PASS:-20000}"
MAX_SET_PASSES="${MAX_SET_PASSES:-30}"
MAX_MINIFIG_PASSES="${MAX_MINIFIG_PASSES:-30}"
NO_PROGRESS_PASSES="${NO_PROGRESS_PASSES:-4}"

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-20}"
RETRIES="${RETRIES:-2}"
DELAY_SECONDS="${DELAY_SECONDS:-0.08}"
JITTER_SECONDS="${JITTER_SECONDS:-0.02}"

if [[ ! -f "${SETS_JSON}" ]]; then
  echo "Missing sets JSON: ${SETS_JSON}" >&2
  exit 1
fi
if [[ ! -f "${MINIFIGS_JSON}" ]]; then
  echo "Missing minifigs JSON: ${MINIFIGS_JSON}" >&2
  exit 1
fi

coverage_counts() {
  python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(rows, list):
    print("0 0")
    raise SystemExit(0)

total = len(rows)
priced = 0
for row in rows:
    if not isinstance(row, dict):
        continue
    new_price = str(row.get("New") or "").strip()
    used_price = str(row.get("Used") or "").strip()
    if new_price or used_price:
        priced += 1
print(f"{priced} {total}")
PY
}

market_cursor() {
  python3 - "${MARKET_STATE_JSON}" "$1" <<'PY'
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
cursor_key = sys.argv[2]
if not state_path.exists():
    print("0")
    raise SystemExit(0)
try:
    data = json.loads(state_path.read_text(encoding="utf-8"))
except Exception:
    print("0")
    raise SystemExit(0)
value = data.get(cursor_key, 0)
try:
    print(str(int(value)))
except Exception:
    print("0")
PY
}

run_hydration() {
  local kind="$1"
  local label="$2"
  local cursor_key="$3"
  local data_json="$4"
  local per_pass_limit="$5"
  local max_api_calls="$6"
  local max_passes="$7"

  read -r start_priced total_rows <<<"$(coverage_counts "${data_json}")"
  local start_cursor
  start_cursor="$(market_cursor "${cursor_key}")"
  local current_priced="${start_priced}"
  local current_cursor="${start_cursor}"
  local no_progress_count=0

  echo "[Hydrate${label}] start priced=${start_priced}/${total_rows} cursor=${start_cursor}"
  echo "[Hydrate${label}] per_pass_limit=${per_pass_limit} max_api_calls=${max_api_calls} max_passes=${max_passes}"

  if (( total_rows <= 0 )); then
    return 0
  fi

  for ((pass=1; pass<=max_passes; pass++)); do
    local before_priced="${current_priced}"
    local before_cursor="${current_cursor}"

    python3 scripts/update_market_values.py \
      --item-type "${kind}" \
      --sets-json "${SETS_JSON}" \
      --minifigs-json "${MINIFIGS_JSON}" \
      --parts-json "${PARTS_JSON}" \
      --currency-code "${BRICKLINK_CURRENCY}" \
      --fallback-currencies "${BRICKLINK_FALLBACK_CURRENCIES}" \
      --limit "${per_pass_limit}" \
      --max-api-calls "${max_api_calls}" \
      --market-state-json "${MARKET_STATE_JSON}" \
      --catalog-sync-state-json "${CATALOG_SYNC_STATE_JSON}" \
      --timeout "${TIMEOUT_SECONDS}" \
      --retries "${RETRIES}" \
      --delay "${DELAY_SECONDS}" \
      --jitter "${JITTER_SECONDS}" \
      --verbose

    read -r current_priced total_rows <<<"$(coverage_counts "${data_json}")"
    current_cursor="$(market_cursor "${cursor_key}")"
    local delta=$(( current_priced - before_priced ))

    echo "[Hydrate${label}] pass=${pass} priced=${current_priced}/${total_rows} delta=${delta} cursor=${before_cursor}->${current_cursor}"

    if (( delta <= 0 )); then
      no_progress_count=$(( no_progress_count + 1 ))
    else
      no_progress_count=0
    fi

    if (( pass > 1 )) && [[ "${current_cursor}" == "${start_cursor}" ]]; then
      echo "[Hydrate${label}] completed full cursor rotation."
      break
    fi

    if (( no_progress_count >= NO_PROGRESS_PASSES )); then
      echo "[Hydrate${label}] stopping after ${NO_PROGRESS_PASSES} consecutive no-progress passes."
      break
    fi
  done

  local coverage_percent="0.00"
  if (( total_rows > 0 )); then
    coverage_percent="$(python3 - <<PY
total=${total_rows}
priced=${current_priced}
print(f"{(priced/total)*100:.2f}")
PY
)"
  fi
  echo "[Hydrate${label}] done priced=${current_priced}/${total_rows} coverage=${coverage_percent}%"
}

run_hydration "set" "Sets" "nextSetIndex" "${SETS_JSON}" "${SETS_PER_PASS}" "${SETS_MAX_API_CALLS_PER_PASS}" "${MAX_SET_PASSES}"
run_hydration "minifig" "Minifigs" "nextMinifigIndex" "${MINIFIGS_JSON}" "${MINIFIGS_PER_PASS}" "${MINIFIGS_MAX_API_CALLS_PER_PASS}" "${MAX_MINIFIG_PASSES}"

echo "[HydrateRelease] sets+minifigs hydration complete."
