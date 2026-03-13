#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SETS_JSON="${SETS_JSON:-dist/Lego Star Wars Database.json}"
MINIFIGS_JSON="${MINIFIGS_JSON:-dist/Lego-Star-Wars-Minifigure-Database.json}"
THEMES_JSON="${THEMES_JSON:-dist/Themes.json}"
PARTS_OUTPUT_DIR="${PARTS_OUTPUT_DIR:-dist/parts}"
REBRICKABLE_DIR="${REBRICKABLE_DIR:-/tmp/lsw_rebrickable}"
REBRICKABLE_BASE_URL="${REBRICKABLE_BASE_URL:-https://cdn.rebrickable.com/media/downloads}"
DIST_BASE_URL="${DIST_BASE_URL:-}"
MARKET_CURRENCY_CODE="${MARKET_CURRENCY_CODE:-GBP}"
VERBOSE="${VERBOSE:-1}"

mkdir -p "${REBRICKABLE_DIR}" "${PARTS_OUTPUT_DIR}"

EXTRA_ARGS=()
if [[ "${VERBOSE}" == "1" ]]; then
  EXTRA_ARGS+=(--verbose)
fi

fetch_csv() {
  local name="$1"
  local url="${REBRICKABLE_BASE_URL}/${name}"
  local target="${REBRICKABLE_DIR}/${name}"
  echo "[Rebrickable] Downloading ${name}"
  curl -L --fail --retry 4 --retry-all-errors --retry-delay 2 -o "${target}.tmp" "${url}"
  mv "${target}.tmp" "${target}"
}

fetch_csv "themes.csv.gz"
fetch_csv "sets.csv.gz"
fetch_csv "minifigs.csv.gz"
fetch_csv "inventories.csv.gz"
fetch_csv "inventory_minifigs.csv.gz"
fetch_csv "parts.csv.gz"
fetch_csv "part_categories.csv.gz"
fetch_csv "colors.csv.gz"
fetch_csv "inventory_parts.csv.gz"

python3 scripts/bootstrap_rebrickable_catalog.py \
  --rebrickable-dir "${REBRICKABLE_DIR}" \
  --sets-json "${SETS_JSON}" \
  --themes-json "${THEMES_JSON}" \
  --minifigs-json "${MINIFIGS_JSON}" \
  --fill-missing-fields \
  --refresh-existing-fields \
  "${EXTRA_ARGS[@]}"

python3 scripts/build_parts_inventory_from_rebrickable.py \
  --rebrickable-dir "${REBRICKABLE_DIR}" \
  --sets-json "${SETS_JSON}" \
  --output-dir "${PARTS_OUTPUT_DIR}"

rm -rf "dist/chunks" "dist/market-details"

python3 scripts/split_catalog_chunks.py \
  --sets-json "${SETS_JSON}" \
  --minifigs-json "${MINIFIGS_JSON}" \
  --output-dir "dist/chunks" \
  --manifest-path "dist/catalog-index.json" \
  --base-url "${DIST_BASE_URL}" \
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
  --base-url "${DIST_BASE_URL}" \
  --market-currency-code "${MARKET_CURRENCY_CODE}" \
  "${EXTRA_ARGS[@]}"

python3 scripts/compact_release_monoliths.py \
  --sets-json "${SETS_JSON}" \
  --minifigs-json "${MINIFIGS_JSON}"

echo "[Sync] Rebrickable catalog + parts + chunk rebuild complete."
