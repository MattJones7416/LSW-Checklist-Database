#!/usr/bin/env python3
"""Build lightweight sync artifacts for client chunk/delta configuration.

Outputs:
- client-config.json (remote endpoint profile for the app)
- catalog-delta-index.json (subset manifest for recently changed chunks)
- market-price-seed.json (compact number/new/used rows)
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def collapse_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_set_code(number: Any, variant: Any) -> str:
    raw_number = collapse_ws(number)
    if not raw_number:
        return ""
    if re.search(r"-[0-9]+$", raw_number):
        return raw_number.lower()
    try:
        variant_no = int(float(collapse_ws(variant) or "1"))
    except ValueError:
        variant_no = 1
    return f"{raw_number}-{max(1, variant_no)}".lower()


def load_json_object(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    raise ValueError(f"Expected JSON object in {path}")


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [row for row in data if isinstance(row, dict)]


def parse_codes(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    output: List[str] = []
    seen: set[str] = set()
    for value in values:
        code = collapse_ws(value).lower()
        if not code or code in seen:
            continue
        seen.add(code)
        output.append(code)
    return output


def dedupe_entries(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        url = collapse_ws(entry.get("url"))
        key = url.lower() if url else json.dumps(entry, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def build_market_price_seed_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        number = collapse_ws(row.get("Number"))
        if not number:
            continue
        new_value = collapse_ws(row.get("New"))
        used_value = collapse_ws(row.get("Used"))
        if not new_value and not used_value:
            continue

        out.append(
            {
                "Number": number,
                "New": new_value if new_value else None,
                "Used": used_value if used_value else None,
            }
        )
    return out


def chunk_entries_for_themes(entries: Sequence[Dict[str, Any]], themes: set[str]) -> List[Dict[str, Any]]:
    if not themes:
        return []
    selected = [entry for entry in entries if collapse_ws(entry.get("theme")).casefold() in themes]
    return dedupe_entries(selected)


def chunk_entries_for_categories(entries: Sequence[Dict[str, Any]], categories: set[str]) -> List[Dict[str, Any]]:
    if not categories:
        return []
    selected = [entry for entry in entries if collapse_ws(entry.get("category")).casefold() in categories]
    return dedupe_entries(selected)


def resolve_url(base_url: str, relative: str) -> str:
    rel = relative.lstrip("/")
    return f"{base_url.rstrip('/')}/{rel}"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sync artifacts for chunked/delta app sync.")
    parser.add_argument("--manifest-path", default="dist/catalog-index.json", help="Primary chunk manifest JSON path.")
    parser.add_argument("--sync-state-path", default="dist/sync-state.json", help="Catalog sync state JSON path.")
    parser.add_argument("--sets-json", default="dist/Lego Star Wars Database.json", help="Sets JSON path.")
    parser.add_argument("--minifigs-json", default="dist/Lego-Star-Wars-Minifigure-Database.json", help="Minifigs JSON path.")
    parser.add_argument("--delta-manifest-path", default="dist/catalog-delta-index.json", help="Output delta manifest path.")
    parser.add_argument("--client-config-path", default="dist/client-config.json", help="Output client config path.")
    parser.add_argument("--market-price-seed-path", default="dist/market-price-seed.json", help="Output compact market seed path.")
    parser.add_argument("--base-url", required=True, help="Base raw URL for dist artifacts.")
    parser.add_argument("--strategy", default="chunked", help="Client default sync strategy.")
    parser.add_argument("--profile-name", default="github-chunked-v2", help="Client profile name.")
    parser.add_argument("--market-currency-code", default="GBP", help="Market currency code.")
    parser.add_argument("--marketplace-deals-url", default="", help="Primary marketplace deals URL.")
    parser.add_argument("--marketplace-deals-fallback-url", default="", help="Fallback marketplace deals URL.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    manifest_path = Path(args.manifest_path)
    sync_state_path = Path(args.sync_state_path)
    sets_path = Path(args.sets_json)
    minifigs_path = Path(args.minifigs_json)
    delta_manifest_path = Path(args.delta_manifest_path)
    client_config_path = Path(args.client_config_path)
    market_seed_path = Path(args.market_price_seed_path)

    manifest = load_json_object(manifest_path)
    sync_state = load_json_object(sync_state_path)
    sets = load_json_array(sets_path)
    minifigs = load_json_array(minifigs_path)

    set_entries = [entry for entry in manifest.get("sets", []) if isinstance(entry, dict)]
    minifig_entries = [entry for entry in manifest.get("minifigures", []) if isinstance(entry, dict)]

    changed_set_codes = parse_codes(sync_state.get("lastUpdatedSetCodes"))
    changed_minifig_codes = parse_codes(
        sync_state.get("lastUpdatedMinifigNumbers")
        or sync_state.get("lastUpdatedMinifigCodes")
        or sync_state.get("lastUpdatedMinifigs")
    )

    set_theme_by_code: Dict[str, str] = {}
    for row in sets:
        code = normalize_set_code(row.get("Number"), row.get("Variant"))
        if not code:
            continue
        set_theme_by_code[code] = collapse_ws(row.get("Theme"))

    minifig_category_by_code: Dict[str, str] = {}
    for row in minifigs:
        code = collapse_ws(row.get("Number")).lower()
        if not code:
            continue
        minifig_category_by_code[code] = collapse_ws(row.get("Category") or row.get("Theme"))

    changed_set_themes = {
        collapse_ws(set_theme_by_code.get(code)).casefold()
        for code in changed_set_codes
        if collapse_ws(set_theme_by_code.get(code))
    }
    changed_minifig_categories = {
        collapse_ws(minifig_category_by_code.get(code)).casefold()
        for code in changed_minifig_codes
        if collapse_ws(minifig_category_by_code.get(code))
    }

    delta_set_entries = chunk_entries_for_themes(set_entries, changed_set_themes)
    delta_minifig_entries = chunk_entries_for_categories(minifig_entries, changed_minifig_categories)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    delta_manifest = {
        "version": manifest.get("version", 1),
        "mode": "delta",
        "generatedAt": generated_at,
        "sourceManifestGeneratedAt": manifest.get("generatedAt"),
        "sets": delta_set_entries,
        "minifigures": delta_minifig_entries,
        "themes": sorted({collapse_ws(e.get("theme")) for e in delta_set_entries if collapse_ws(e.get("theme"))}, key=lambda v: v.casefold()),
        "summary": {
            "setChunkCount": len(delta_set_entries),
            "minifigureChunkCount": len(delta_minifig_entries),
            "setThemeCount": len(changed_set_themes),
            "minifigureCategoryCount": len(changed_minifig_categories),
            "changedSetCodes": len(changed_set_codes),
            "changedMinifigCodes": len(changed_minifig_codes),
        },
    }

    delta_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    delta_manifest_path.write_text(json.dumps(delta_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    market_seed_rows = build_market_price_seed_rows(sets) + build_market_price_seed_rows(minifigs)
    market_seed_path.parent.mkdir(parents=True, exist_ok=True)
    market_seed_path.write_text(json.dumps(market_seed_rows, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    base_url = args.base_url.rstrip("/")
    set_catalog_url = resolve_url(base_url, "Lego%20Star%20Wars%20Database.json")
    minifigure_catalog_url = resolve_url(base_url, "Lego-Star-Wars-Minifigure-Database.json")
    chunk_manifest_url = resolve_url(base_url, manifest_path.name)
    delta_manifest_url = resolve_url(base_url, delta_manifest_path.name)
    market_details_base_url = resolve_url(base_url, "market-details")
    market_seed_url = resolve_url(base_url, market_seed_path.name)

    deals_url = collapse_ws(args.marketplace_deals_url) or resolve_url(base_url, "deals")
    deals_fallback_url = collapse_ws(args.marketplace_deals_fallback_url) or resolve_url(base_url, "marketplace-deals.json")

    client_config = {
        "profileName": collapse_ws(args.profile_name) or "github-chunked-v2",
        "strategy": collapse_ws(args.strategy) or "chunked",
        "setCatalogURL": set_catalog_url,
        "minifigureCatalogURL": minifigure_catalog_url,
        "chunkManifestURL": chunk_manifest_url,
        "deltaManifestURL": delta_manifest_url,
        "marketDetailsBaseURL": market_details_base_url,
        "marketplaceDealsURL": deals_url,
        "marketplaceDealsFallbackURL": deals_fallback_url,
        "marketPriceSeedURL": market_seed_url,
        "marketCurrencyCode": collapse_ws(args.market_currency_code).upper() or "GBP",
        "marketDataVersion": generated_at,
        "generatedAt": generated_at,
        "summary": {
            "setCount": len(sets),
            "minifigureCount": len(minifigs),
            "deltaSetChunkCount": len(delta_set_entries),
            "deltaMinifigureChunkCount": len(delta_minifig_entries),
            "marketSeedItemCount": len(market_seed_rows),
        },
        "sync": {
            "strategy": collapse_ws(args.strategy) or "chunked",
            "setCatalogURL": set_catalog_url,
            "minifigureCatalogURL": minifigure_catalog_url,
            "chunkManifestURL": chunk_manifest_url,
            "deltaManifestURL": delta_manifest_url,
            "marketDetailsBaseURL": market_details_base_url,
            "marketPriceSeedURL": market_seed_url,
            "marketplaceDealsURL": deals_url,
            "marketplaceDealsFallbackURL": deals_fallback_url,
            "marketCurrencyCode": collapse_ws(args.market_currency_code).upper() or "GBP",
        },
    }

    client_config_path.parent.mkdir(parents=True, exist_ok=True)
    client_config_path.write_text(json.dumps(client_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.verbose:
        print(
            (
                f"[Artifacts] sets={len(sets)} minifigs={len(minifigs)} "
                f"delta_sets={len(delta_set_entries)} delta_minifigs={len(delta_minifig_entries)} "
                f"market_seed_items={len(market_seed_rows)}"
            ),
            flush=True,
        )

    print(
        (
            f"[Artifacts] wrote {delta_manifest_path} {client_config_path} {market_seed_path}"
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
