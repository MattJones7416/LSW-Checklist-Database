#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PARTS_JSON="${PARTS_JSON:-dist/parts/parts-catalog.json}"
SETS_JSON="${SETS_JSON:-dist/Lego Star Wars Database.json}"
MINIFIGS_JSON="${MINIFIGS_JSON:-dist/Lego-Star-Wars-Minifigure-Database.json}"
MARKET_STATE_JSON="${MARKET_STATE_JSON:-dist/market-sync-state.json}"
CATALOG_SYNC_STATE_JSON="${CATALOG_SYNC_STATE_JSON:-dist/sync-state.json}"

BRICKLINK_CURRENCY="${BRICKLINK_CURRENCY:-GBP}"
BRICKLINK_FALLBACK_CURRENCIES="${BRICKLINK_FALLBACK_CURRENCIES:-USD,EUR,GBP}"

PER_RUN_LIMIT="${PER_RUN_LIMIT:-2500}"
MAX_API_CALLS_PER_RUN="${MAX_API_CALLS_PER_RUN:-20000}"
MAX_PASSES="${MAX_PASSES:-120}"
NO_PROGRESS_PASSES="${NO_PROGRESS_PASSES:-5}"

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-20}"
RETRIES="${RETRIES:-2}"
DELAY_SECONDS="${DELAY_SECONDS:-0.08}"
JITTER_SECONDS="${JITTER_SECONDS:-0.02}"

if [[ ! -f "${PARTS_JSON}" ]]; then
  echo "Missing parts catalog: ${PARTS_JSON}" >&2
  exit 1
fi
if [[ ! -f "${SETS_JSON}" ]]; then
  echo "Missing sets JSON: ${SETS_JSON}" >&2
  exit 1
fi
if [[ ! -f "${MINIFIGS_JSON}" ]]; then
  echo "Missing minifigs JSON: ${MINIFIGS_JSON}" >&2
  exit 1
fi

coverage_counts() {
  python3 - "${PARTS_JSON}" <<'PY'
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
    new_price = str(row.get("market_price_new") or "").strip()
    used_price = str(row.get("market_price_used") or "").strip()
    if new_price or used_price:
        priced += 1
print(f"{priced} {total}")
PY
}

part_cursor() {
  python3 - "${MARKET_STATE_JSON}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("0")
    raise SystemExit(0)
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("0")
    raise SystemExit(0)

value = data.get("nextPartIndex", 0)
try:
    print(str(int(value)))
except Exception:
    print("0")
PY
}

read -r start_priced total_parts <<<"$(coverage_counts)"
start_cursor="$(part_cursor)"
current_priced="${start_priced}"
current_cursor="${start_cursor}"
no_progress_count=0

echo "[HydrateParts] start priced=${start_priced}/${total_parts} cursor=${start_cursor}"
echo "[HydrateParts] per_run_limit=${PER_RUN_LIMIT} max_api_calls_per_run=${MAX_API_CALLS_PER_RUN} max_passes=${MAX_PASSES}"

for ((pass=1; pass<=MAX_PASSES; pass++)); do
  before_priced="${current_priced}"
  before_cursor="${current_cursor}"

  python3 scripts/update_market_values.py \
    --item-type part \
    --sets-json "${SETS_JSON}" \
    --minifigs-json "${MINIFIGS_JSON}" \
    --parts-json "${PARTS_JSON}" \
    --currency-code "${BRICKLINK_CURRENCY}" \
    --fallback-currencies "${BRICKLINK_FALLBACK_CURRENCIES}" \
    --limit "${PER_RUN_LIMIT}" \
    --max-api-calls "${MAX_API_CALLS_PER_RUN}" \
    --market-state-json "${MARKET_STATE_JSON}" \
    --catalog-sync-state-json "${CATALOG_SYNC_STATE_JSON}" \
    --timeout "${TIMEOUT_SECONDS}" \
    --retries "${RETRIES}" \
    --delay "${DELAY_SECONDS}" \
    --jitter "${JITTER_SECONDS}" \
    --verbose

  read -r current_priced total_parts <<<"$(coverage_counts)"
  current_cursor="$(part_cursor)"
  delta=$(( current_priced - before_priced ))

  echo "[HydrateParts] pass=${pass} priced=${current_priced}/${total_parts} delta=${delta} cursor=${before_cursor}->${current_cursor}"

  if (( delta <= 0 )); then
    no_progress_count=$(( no_progress_count + 1 ))
  else
    no_progress_count=0
  fi

  if (( pass > 1 )) && [[ "${current_cursor}" == "${start_cursor}" ]]; then
    echo "[HydrateParts] Completed a full cursor rotation."
    break
  fi

  if (( no_progress_count >= NO_PROGRESS_PASSES )); then
    echo "[HydrateParts] Stopping after ${NO_PROGRESS_PASSES} consecutive no-progress passes."
    break
  fi
done

coverage_percent="0.00"
if (( total_parts > 0 )); then
  coverage_percent="$(python3 - <<PY
total=${total_parts}
priced=${current_priced}
print(f"{(priced/total)*100:.2f}")
PY
)"
fi

echo "[HydrateParts] done priced=${current_priced}/${total_parts} coverage=${coverage_percent}%"
echo "[HydrateParts] next: commit dist/parts/parts-catalog.json and dist/market-sync-state.json"
