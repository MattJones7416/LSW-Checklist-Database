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
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-20}"
RETRIES="${RETRIES:-2}"
DELAY_SECONDS="${DELAY_SECONDS:-0.25}"
JITTER_SECONDS="${JITTER_SECONDS:-0.10}"
LIMIT="${LIMIT:-}"
START_INDEX="${START_INDEX:-0}"
MISSING_ONLY="${MISSING_ONLY:-0}"

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
  --start-index "${START_INDEX}"
  --verbose
)

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

if [[ "${MISSING_ONLY}" == "1" ]]; then
  CMD+=(--missing-only)
fi

"${CMD[@]}"

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

echo "[BootstrapHTMLRelease] complete."
