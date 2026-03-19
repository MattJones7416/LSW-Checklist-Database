# LSW-Checklist-Database

This repo maintains the set, minifigure, and piece catalogs consumed by the app.

## Current model

The catalog is now split into two layers:

1. Core catalog data from Rebrickable and the curated app catalog.
2. Optional per-item market data written into `dist/catalog/.../market.json`.

The old BrickLink market-refresh pipeline has been removed from this repo.
Market data is now expected to come from an external scraper import step.

## Automated daily sync

GitHub Actions runs the Rebrickable catalog sync daily and can also be triggered manually.

- Workflow: `.github/workflows/sync-rebrickable-catalog.yml`
- Core scripts:
  - `scripts/sync_rebrickable_catalog_release.sh`
  - `scripts/build_parts_inventory_from_rebrickable.py`
  - `scripts/compact_release_monoliths.py`
  - `scripts/split_catalog_chunks.py`
  - `scripts/build_sync_artifacts.py`
  - `scripts/build_item_folder_catalog.py`

## Pipeline summary

1. Download the latest Rebrickable source files.
2. Rebuild the core set and minifigure catalogs.
3. Rebuild the parts catalog and set-parts index.
4. Strip legacy market fields from the monolith JSON files.
5. Rebuild chunked sync artifacts.
6. Rebuild per-item folders under `dist/catalog`.
7. Commit/push the refreshed catalog artifacts.

## Published artifacts

Core catalogs:

- `dist/Lego Star Wars Database.json`
- `dist/Lego-Star-Wars-Minifigure-Database.json`
- `dist/Themes.json`
- `dist/parts/parts-catalog.json`
- `dist/parts/set-parts-index.json`
- `dist/parts/set-parts/**`

Chunked sync artifacts:

- `dist/catalog-index.json`
- `dist/catalog-delta-index.json`
- `dist/chunks/**`
- `dist/client-config.json`

Per-item catalog:

- `dist/catalog/sets/<theme>/<setnumber-itemname>/item.json`
- `dist/catalog/minifigs/<theme>/<minifignumber-itemname>/item.json`
- `dist/catalog/pieces/<partnumber-itemname>/item.json`

Optional per-item companion files:

- `market.json`
- `parts.json`
- `minifigures.json`
- `appears-in-sets.json`

## External market import

When the standalone scraper is ready, import its output into the per-item catalog with:

```bash
python scripts/import_scraper_market_data.py \
  --scraper-dist-dir "/path/to/Scraper/dist" \
  --catalog-dir "dist/catalog" \
  --verbose
```

That step writes normalized `market.json` files beside each item folder.

## Local run

```bash
./scripts/sync_rebrickable_catalog_release.sh
```

Then, when market scraper output is ready:

```bash
python scripts/import_scraper_market_data.py \
  --scraper-dist-dir "/path/to/Scraper/dist" \
  --catalog-dir "dist/catalog"
```
