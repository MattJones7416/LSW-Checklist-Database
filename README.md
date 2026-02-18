# LSW-Checklist-Database

This repo auto-maintains the set and minifigure JSON catalogs used by the app.

## Automated daily sync

GitHub Actions runs daily and can also be run manually.

- Workflow: `.github/workflows/update-market-values.yml`
- Scripts:
  - `scripts/bootstrap_rebrickable_catalog.py` (bulk seed from Rebrickable CSV dumps)
  - `scripts/sync_brickset_catalog.py` (catalog + themes + multiplicity)
  - `scripts/update_market_values.py` (BrickLink API market refresh)

### Pipeline summary

1. Bootstrap missing sets from Rebrickable CSV dumps (bulk source).
2. Run a low-call incremental set pass using Brickset `updatedSince` (new/changed sets first).
3. Use remaining API budget on rolling theme refresh (cursor-based across runs).
4. Optionally crawl Brickset minifig pages (currently disabled in workflow for stability).
5. Update/add entries in:
   - `dist/Lego Star Wars Database.json`
   - `dist/Lego-Star-Wars-Minifigure-Database.json`
   - `dist/Themes.json`
6. Expand set `MinifigNumbers` multiplicities from BrickLink inventory pages.
7. Refresh market fields via **BrickLink API** (OAuth, no HTML scraping):
   - `New` / `Used` display values
   - sold/current summary stats
   - monthly market snapshot series
   - forecast helper fields
   - latest sale snapshot fields
8. Refresh cross-catalog helper fields:
   - `ExclusiveMinifigNumbers`, `ExclusiveMinifigCount`
   - `AppearsInSetNumbers`, `IsSetExclusive`, `ExclusiveToSetNumber`
9. Push checkpoint commits during the workflow (catalog checkpoint + market chunk checkpoints).

State files written by automation:

- `dist/sync-state.json` (catalog cursor + incremental cutoff + recently changed set codes)
- `dist/market-sync-state.json` (market refresh cursor + API usage metadata)

## Bootstrap data sources (for full seed)

Use these to seed large catalogs quickly before daily API deltas:

- Rebrickable CSV download docs: `https://rebrickable.com/downloads/`
- Direct CSV endpoints:
  - `https://cdn.rebrickable.com/media/downloads/themes.csv.gz`
  - `https://cdn.rebrickable.com/media/downloads/sets.csv.gz`
  - `https://cdn.rebrickable.com/media/downloads/minifigs.csv.gz`
  - `https://cdn.rebrickable.com/media/downloads/inventories.csv.gz`
  - `https://cdn.rebrickable.com/media/downloads/inventory_sets.csv.gz`
  - `https://cdn.rebrickable.com/media/downloads/inventory_minifigs.csv.gz`

Recommended model:

1. Bootstrap sets/minifigs/themes/inventories from Rebrickable CSV.
2. Daily: Brickset API incremental (`updatedSince`) + rolling minor refresh.
3. Daily: BrickLink market refresh with request budget and cursor rotation.
4. Priority queue: changed/new set codes from `dist/sync-state.json` first, then rotating backlog.

## Required GitHub Secrets

Set these in repository Secrets (or Variables):

- `BRICKSET_API_KEY`
- `BRICKLINK_CONSUMER_KEY`
- `BRICKLINK_CONSUMER_SECRET`
- `BRICKLINK_TOKEN_VALUE`
- `BRICKLINK_TOKEN_SECRET`

Optional:

- `BRICKLINK_CURRENCY` (default `GBP`)

## Local run

```bash
export BRICKSET_API_KEY="..."
export BRICKLINK_CONSUMER_KEY="..."
export BRICKLINK_CONSUMER_SECRET="..."
export BRICKLINK_TOKEN_VALUE="..."
export BRICKLINK_TOKEN_SECRET="..."
export BRICKLINK_CURRENCY="GBP"

python scripts/bootstrap_rebrickable_catalog.py \
  --sets-json "dist/Lego Star Wars Database.json" \
  --themes-json "dist/Themes.json" \
  --minifigs-json "dist/Lego-Star-Wars-Minifigure-Database.json" \
  --fill-missing-fields \
  --timeout 45 \
  --retries 5 \
  --verbose

python scripts/sync_brickset_catalog.py \
  --sets-json "dist/Lego Star Wars Database.json" \
  --minifigs-json "dist/Lego-Star-Wars-Minifigure-Database.json" \
  --themes-json "dist/Themes.json" \
  --api-key "$BRICKSET_API_KEY" \
  --api-page-size 500 \
  --max-api-calls 80 \
  --api-daily-limit 100 \
  --api-usage-safety-margin 5 \
  --sync-state-json "dist/sync-state.json" \
  --verbose

python scripts/update_market_values.py \
  --sets-json "dist/Lego Star Wars Database.json" \
  --minifigs-json "dist/Lego-Star-Wars-Minifigure-Database.json" \
  --currency-code "$BRICKLINK_CURRENCY" \
  --max-api-calls 4800 \
  --market-state-json "dist/market-sync-state.json" \
  --catalog-sync-state-json "dist/sync-state.json" \
  --delay 0.18 \
  --jitter 0.04 \
  --timeout 20 \
  --retries 3 \
  --verbose
```

Useful debug flags:

- `--limit 25 --start-index 0`
- `--skip-cross-enrichment`
