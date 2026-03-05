# LSW-Checklist-Database

This repo auto-maintains the set and minifigure catalogs consumed by the app.

## Automated daily sync

GitHub Actions runs daily and can also be triggered manually.

- Workflow: `.github/workflows/update-market-values.yml`
- Scripts:
  - `scripts/bootstrap_bricklink_catalog.py` (catalog bootstrap/update from BrickLink feeds)
  - `scripts/update_market_values.py` (BrickLink API market refresh)
  - `scripts/split_catalog_chunks.py` (chunked catalog + market-details files)
  - `scripts/build_sync_artifacts.py` (client config + delta manifest + compact market seed)

## Pipeline summary

1. Bootstrap/update sets + minifigs from BrickLink catalog feeds.
2. Commit/push a catalog checkpoint if changed.
3. Refresh market fields via BrickLink API in request-budgeted chunks.
4. Commit/push market checkpoints after each chunk.
5. Build chunked catalog + per-item market detail files.
6. Build client sync artifacts for the app.
7. Commit/push final chunk/artifact outputs.

## Published artifacts

Core catalogs:

- `dist/Lego Star Wars Database.json`
- `dist/Lego-Star-Wars-Minifigure-Database.json`
- `dist/Themes.json`

Chunked sync artifacts:

- `dist/catalog-index.json`
- `dist/catalog-delta-index.json`
- `dist/chunks/**`
- `dist/market-details/**`
- `dist/client-config.json`
- `dist/market-price-seed.json`

State files:

- `dist/sync-state.json`
- `dist/market-sync-state.json`

## Required GitHub Secrets / Variables

Required (Secrets or Variables):

- `BRICKLINK_CONSUMER_KEY`
- `BRICKLINK_CONSUMER_SECRET`
- `BRICKLINK_TOKEN_VALUE`
- `BRICKLINK_TOKEN_SECRET`

Optional:

- `BRICKLINK_CATALOG_COOKIE` (if feed endpoints require authenticated cookie)
- `BRICKLINK_CURRENCY` (default `GBP`)
- `MARKET_PRIORITY_THEMES` (default `Star Wars,Marvel Super Heroes,Disney,NINJAGO`)
- `MARKET_PRIORITY_MINIFIG_CATEGORIES` (default `Star Wars,Marvel Super Heroes,Disney,NINJAGO`)

## Local run

```bash
export BRICKLINK_CONSUMER_KEY="..."
export BRICKLINK_CONSUMER_SECRET="..."
export BRICKLINK_TOKEN_VALUE="..."
export BRICKLINK_TOKEN_SECRET="..."
export BRICKLINK_CURRENCY="GBP"

python scripts/bootstrap_bricklink_catalog.py   --sets-json "dist/Lego Star Wars Database.json"   --themes-json "dist/Themes.json"   --minifigs-json "dist/Lego-Star-Wars-Minifigure-Database.json"   --timeout 45   --retries 5   --verbose

python scripts/update_market_values.py   --item-type both   --sets-json "dist/Lego Star Wars Database.json"   --minifigs-json "dist/Lego-Star-Wars-Minifigure-Database.json"   --currency-code "$BRICKLINK_CURRENCY"   --priority-themes "Star Wars,Marvel Super Heroes,Disney,NINJAGO"   --priority-minifig-categories "Star Wars,Marvel Super Heroes,Disney,NINJAGO"   --max-api-calls 4800   --market-state-json "dist/market-sync-state.json"   --catalog-sync-state-json "dist/sync-state.json"   --delay 0.18   --jitter 0.04   --timeout 20   --retries 3   --verbose

python scripts/split_catalog_chunks.py   --sets-json "dist/Lego Star Wars Database.json"   --minifigs-json "dist/Lego-Star-Wars-Minifigure-Database.json"   --output-dir "dist/chunks"   --manifest-path "dist/catalog-index.json"   --base-url "https://raw.githubusercontent.com/<owner>/<repo>/refs/heads/main/dist"   --market-details-dir "dist/market-details"   --strip-market-detail-fields   --max-items-per-chunk 800

python scripts/build_sync_artifacts.py   --manifest-path "dist/catalog-index.json"   --sync-state-path "dist/sync-state.json"   --sets-json "dist/Lego Star Wars Database.json"   --minifigs-json "dist/Lego-Star-Wars-Minifigure-Database.json"   --delta-manifest-path "dist/catalog-delta-index.json"   --client-config-path "dist/client-config.json"   --market-price-seed-path "dist/market-price-seed.json"   --base-url "https://raw.githubusercontent.com/<owner>/<repo>/refs/heads/main/dist"   --market-currency-code "$BRICKLINK_CURRENCY"   --verbose
```
