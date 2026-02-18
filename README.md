# LSW-Checklist-Database

This repo auto-maintains the set and minifigure JSON catalogs used by the app.

## Automated daily sync

GitHub Actions runs daily and can also be run manually.

- Workflow: `.github/workflows/update-market-values.yml`
- Scripts:
  - `scripts/sync_brickset_catalog.py` (catalog + themes + multiplicity)
  - `scripts/update_market_values.py` (BrickLink API market refresh)

### Pipeline summary

1. Pull sets from Brickset API v3 (`getSets`) across themes.
2. Optionally crawl Brickset minifig pages (currently disabled in workflow for stability).
3. Update/add entries in:
   - `dist/Lego Star Wars Database.json`
   - `dist/Lego-Star-Wars-Minifigure-Database.json`
   - `dist/Themes.json`
4. Expand set `MinifigNumbers` multiplicities from BrickLink inventory pages.
5. Refresh market fields via **BrickLink API** (OAuth, no HTML scraping):
   - `New` / `Used` display values
   - sold/current summary stats
   - monthly market snapshot series
   - forecast helper fields
   - latest sale snapshot fields
6. Refresh cross-catalog helper fields:
   - `ExclusiveMinifigNumbers`, `ExclusiveMinifigCount`
   - `AppearsInSetNumbers`, `IsSetExclusive`, `ExclusiveToSetNumber`

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

python scripts/sync_brickset_catalog.py \
  --sets-json "dist/Lego Star Wars Database.json" \
  --minifigs-json "dist/Lego-Star-Wars-Minifigure-Database.json" \
  --themes-json "dist/Themes.json" \
  --api-key "$BRICKSET_API_KEY" \
  --api-page-size 500 \
  --verbose

python scripts/update_market_values.py \
  --sets-json "dist/Lego Star Wars Database.json" \
  --minifigs-json "dist/Lego-Star-Wars-Minifigure-Database.json" \
  --currency-code "$BRICKLINK_CURRENCY" \
  --delay 0.18 \
  --jitter 0.04 \
  --timeout 20 \
  --retries 3 \
  --verbose
```

Useful debug flags:

- `--limit 25 --start-index 0`
- `--skip-cross-enrichment`
