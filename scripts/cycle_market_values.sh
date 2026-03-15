#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SETS_JSON="${SETS_JSON:-dist/Lego Star Wars Database.json}"
MINIFIGS_JSON="${MINIFIGS_JSON:-dist/Lego-Star-Wars-Minifigure-Database.json}"
PARTS_JSON="${PARTS_JSON:-dist/parts/parts-catalog.json}"
MARKET_STATE_JSON="${MARKET_STATE_JSON:-dist/market-sync-state.json}"
CATALOG_SYNC_STATE_JSON="${CATALOG_SYNC_STATE_JSON:-dist/sync-state.json}"
SET_ALIASES_JSON="${SET_ALIASES_JSON:-dist/bricklink-set-aliases.json}"

BRICKLINK_CURRENCY="${BRICKLINK_CURRENCY:-GBP}"
BRICKLINK_FALLBACK_CURRENCIES="${BRICKLINK_FALLBACK_CURRENCIES:-USD,EUR,GBP}"
ACTIVE_SET_THEMES="${ACTIVE_SET_THEMES:-Star Wars,Marvel Super Heroes,Disney,NINJAGO}"
ACTIVE_MINIFIG_CATEGORIES="${ACTIVE_MINIFIG_CATEGORIES:-Star Wars,Marvel Super Heroes,Disney,NINJAGO}"
ACTIVE_PART_CATEGORIES="${ACTIVE_PART_CATEGORIES:-Bricks,Plates,Tiles,Minifigure}"
MARKET_NO_DATA_COOLDOWN_HOURS="${MARKET_NO_DATA_COOLDOWN_HOURS:-168}"

SETS_PER_RUN="${SETS_PER_RUN:-450}"
MINIFIGS_PER_RUN="${MINIFIGS_PER_RUN:-300}"
PARTS_PER_RUN="${PARTS_PER_RUN:-200}"

SETS_MAX_API_CALLS="${SETS_MAX_API_CALLS:-2200}"
MINIFIGS_MAX_API_CALLS="${MINIFIGS_MAX_API_CALLS:-1200}"
PARTS_MAX_API_CALLS="${PARTS_MAX_API_CALLS:-800}"

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-20}"
RETRIES="${RETRIES:-2}"
DELAY_SECONDS="${DELAY_SECONDS:-0.10}"
JITTER_SECONDS="${JITTER_SECONDS:-0.03}"
BASE_URL="${BASE_URL:-https://raw.githubusercontent.com/MattJones7416/LSW-Checklist-Database/refs/heads/main/dist}"

if [[ ! -f "${SETS_JSON}" ]]; then
  echo "Missing sets JSON: ${SETS_JSON}" >&2
  exit 1
fi
if [[ ! -f "${MINIFIGS_JSON}" ]]; then
  echo "Missing minifigs JSON: ${MINIFIGS_JSON}" >&2
  exit 1
fi

python3 scripts/update_market_values.py \
  --item-type set \
  --sets-json "${SETS_JSON}" \
  --minifigs-json "${MINIFIGS_JSON}" \
  --parts-json "${PARTS_JSON}" \
  --currency-code "${BRICKLINK_CURRENCY}" \
  --fallback-currencies "${BRICKLINK_FALLBACK_CURRENCIES}" \
  --priority-themes "${ACTIVE_SET_THEMES}" \
  --priority-minifig-categories "${ACTIVE_MINIFIG_CATEGORIES}" \
  --priority-part-categories "${ACTIVE_PART_CATEGORIES}" \
  --no-data-cooldown-hours "${MARKET_NO_DATA_COOLDOWN_HOURS}" \
  --limit "${SETS_PER_RUN}" \
  --max-api-calls "${SETS_MAX_API_CALLS}" \
  --market-state-json "${MARKET_STATE_JSON}" \
  --catalog-sync-state-json "${CATALOG_SYNC_STATE_JSON}" \
  --set-aliases-json "${SET_ALIASES_JSON}" \
  --delay "${DELAY_SECONDS}" \
  --jitter "${JITTER_SECONDS}" \
  --timeout "${TIMEOUT_SECONDS}" \
  --retries "${RETRIES}" \
  --disable-html-fallback \
  --verbose

python3 scripts/update_market_values.py \
  --item-type minifig \
  --sets-json "${SETS_JSON}" \
  --minifigs-json "${MINIFIGS_JSON}" \
  --parts-json "${PARTS_JSON}" \
  --currency-code "${BRICKLINK_CURRENCY}" \
  --fallback-currencies "${BRICKLINK_FALLBACK_CURRENCIES}" \
  --priority-themes "${ACTIVE_SET_THEMES}" \
  --priority-minifig-categories "${ACTIVE_MINIFIG_CATEGORIES}" \
  --priority-part-categories "${ACTIVE_PART_CATEGORIES}" \
  --no-data-cooldown-hours "${MARKET_NO_DATA_COOLDOWN_HOURS}" \
  --limit "${MINIFIGS_PER_RUN}" \
  --max-api-calls "${MINIFIGS_MAX_API_CALLS}" \
  --market-state-json "${MARKET_STATE_JSON}" \
  --catalog-sync-state-json "${CATALOG_SYNC_STATE_JSON}" \
  --set-aliases-json "${SET_ALIASES_JSON}" \
  --delay "${DELAY_SECONDS}" \
  --jitter "${JITTER_SECONDS}" \
  --timeout "${TIMEOUT_SECONDS}" \
  --retries "${RETRIES}" \
  --disable-html-fallback \
  --verbose

if [[ -f "${PARTS_JSON}" ]]; then
  python3 scripts/update_market_values.py \
    --item-type part \
    --sets-json "${SETS_JSON}" \
    --minifigs-json "${MINIFIGS_JSON}" \
    --parts-json "${PARTS_JSON}" \
    --currency-code "${BRICKLINK_CURRENCY}" \
    --fallback-currencies "${BRICKLINK_FALLBACK_CURRENCIES}" \
    --priority-themes "${ACTIVE_SET_THEMES}" \
    --priority-minifig-categories "${ACTIVE_MINIFIG_CATEGORIES}" \
    --priority-part-categories "${ACTIVE_PART_CATEGORIES}" \
    --no-data-cooldown-hours "${MARKET_NO_DATA_COOLDOWN_HOURS}" \
    --limit "${PARTS_PER_RUN}" \
    --max-api-calls "${PARTS_MAX_API_CALLS}" \
    --market-state-json "${MARKET_STATE_JSON}" \
    --catalog-sync-state-json "${CATALOG_SYNC_STATE_JSON}" \
    --set-aliases-json "${SET_ALIASES_JSON}" \
    --delay "${DELAY_SECONDS}" \
    --jitter "${JITTER_SECONDS}" \
    --timeout "${TIMEOUT_SECONDS}" \
    --retries "${RETRIES}" \
    --disable-html-fallback \
    --verbose
fi

rm -rf "dist/chunks" "dist/market-details"

python3 scripts/split_catalog_chunks.py \
  --sets-json "${SETS_JSON}" \
  --minifigs-json "${MINIFIGS_JSON}" \
  --output-dir "dist/chunks" \
  --manifest-path "dist/catalog-index.json" \
  --base-url "${BASE_URL}" \
  --market-details-dir "dist/market-details" \
  --strip-market-detail-fields \
  --max-items-per-chunk 800

python3 scripts/build_sync_artifacts.py \
  --manifest-path "dist/catalog-index.json" \
  --sync-state-path "${CATALOG_SYNC_STATE_JSON}" \
  --sets-json "${SETS_JSON}" \
  --minifigs-json "${MINIFIGS_JSON}" \
  --delta-manifest-path "dist/catalog-delta-index.json" \
  --client-config-path "dist/client-config.json" \
  --market-price-seed-path "dist/market-price-seed.json" \
  --base-url "${BASE_URL}" \
  --market-currency-code "${BRICKLINK_CURRENCY}" \
  --verbose

python3 scripts/compact_release_monoliths.py \
  --sets-json "${SETS_JSON}" \
  --minifigs-json "${MINIFIGS_JSON}"

echo "[CycleMarket] complete."
