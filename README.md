# LSW-Checklist

This repo auto-maintains the Star Wars set and minifigure JSON catalogs used by the app.

## Automated daily sync

GitHub Actions runs daily and can also be run manually:

- Workflow: `.github/workflows/update-market-values.yml`
- Script: `scripts/sync_brickset_catalog.py`

The sync process:

1. Crawls Brickset Star Wars sets: `https://brickset.com/sets/theme-Star-Wars`
2. Crawls Brickset Star Wars minifigures: `https://brickset.com/minifigs/category-Star-Wars`
3. Updates/adds entries in:
   - `dist/Lego Star Wars Database.json`
   - `dist/Lego-Star-Wars-Minifigure-Database.json`
4. Refreshes `New` / `Used` values from Brickset listing data
5. Refreshes `New` / `Used` values again from each item detail page (`Current value`)
6. Expands set `MinifigNumbers` multiplicities from BrickLink inventory pages
7. Commits and pushes if changes are detected

## Local run

```bash
python scripts/sync_brickset_catalog.py \
  --sets-json "dist/Lego Star Wars Database.json" \
  --minifigs-json "dist/Lego-Star-Wars-Minifigure-Database.json" \
  --verbose
```

Useful debug flags:

- `--dry-run`
- `--skip-multiplicity`
- `--max-set-pages 1 --max-minifig-pages 1`
- `--max-multiplicity-fetches 5`
