# LSW-Checklist

This repo auto-maintains the Star Wars set and minifigure JSON catalogs used by the app.

## Automated daily sync

GitHub Actions runs daily and can also be run manually:

- Workflow: `.github/workflows/update-market-values.yml`
- Scripts:
  - `scripts/sync_brickset_catalog.py`
  - `scripts/update_market_values.py`

The sync process:

1. Crawls Brickset Star Wars sets: `https://brickset.com/sets/theme-Star-Wars`
2. Crawls Brickset Star Wars minifigures: `https://brickset.com/minifigs/category-Star-Wars`
3. Updates/adds entries in:
   - `dist/Lego Star Wars Database.json`
   - `dist/Lego-Star-Wars-Minifigure-Database.json`
4. Expands set `MinifigNumbers` multiplicities from BrickLink inventory pages
5. Refreshes market fields from BrickLink Price Guide pages (`catalogPG.asp`) including:
   - `New` / `Used` display values
   - 6-month sold stats (new/used)
   - current listing stats (new/used)
   - monthly sales series + full transaction rows + latest sale snapshot
   - RRP comparison and 2Y/5Y forecast helper fields
6. Adds cross-catalog exclusivity/appears-in helper fields
7. Commits and pushes if changes are detected

## Local run

```bash
python scripts/sync_brickset_catalog.py \
  --sets-json "dist/Lego Star Wars Database.json" \
  --minifigs-json "dist/Lego-Star-Wars-Minifigure-Database.json" \
  --verbose

python scripts/update_market_values.py \
  --sets-json "dist/Lego Star Wars Database.json" \
  --minifigs-json "dist/Lego-Star-Wars-Minifigure-Database.json" \
  --delay 0.65 \
  --jitter 0.15 \
  --timeout 20 \
  --retries 2 \
  --verbose
```

Useful debug flags:

- `--limit 5 --start-index 0`
- `--skip-cross-enrichment`
- `--link-fallback`
- `--insecure` (local cert troubleshooting only)
