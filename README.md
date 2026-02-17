# LSW-Checklist

This repo auto-maintains the set and minifigure JSON catalogs used by the app.

## Automated daily sync

GitHub Actions runs daily and can also be run manually:

- Workflow: `.github/workflows/update-market-values.yml`
- Scripts:
  - `scripts/sync_brickset_catalog.py`
  - `scripts/update_market_values.py`

The sync process:

1. Pulls sets from Brickset API v3 (`getSets`) across all themes by default
2. Optionally crawls Brickset minifig pages (disabled by default to keep runs stable)
3. Updates/adds entries in:
   - `dist/Lego Star Wars Database.json`
   - `dist/Lego-Star-Wars-Minifigure-Database.json`
   - `dist/Themes.json` (theme index for UI chips/filters)
4. Expands set `MinifigNumbers` multiplicities from BrickLink inventory pages
   - by default only for rows that already have `MinifigNumbers` (avoids massive request volume)
5. Refreshes market fields from BrickLink Price Guide pages (`catalogPG.asp`) including:
   - `New` / `Used` display values
   - 6-month sold stats (new/used)
   - current listing stats (new/used)
   - monthly sales series + full transaction rows + latest sale snapshot
   - RRP comparison and 2Y/5Y forecast helper fields
6. Adds cross-catalog exclusivity/appears-in helper fields
7. Commits and pushes if changes are detected

## Local run

Set API key first:

```bash
export BRICKSET_API_KEY="your_api_key"
```

```bash
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
  --delay 0.65 \
  --jitter 0.15 \
  --timeout 20 \
  --retries 2 \
  --limit 1200 \
  --verbose
```

Useful debug flags:

- `--theme "Star Wars"` (API set filter)
- `--crawl-minifigs` (enable web minifigure crawl)
- `--multiplicity-include-empty` (expensive; disabled by default)
- `--limit 5 --start-index 0`
- `--skip-cross-enrichment`
- `--link-fallback`
- `--insecure` (local cert troubleshooting only)
