#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import update_market_values as market

MARKET_DETAIL_FIELDS: Set[str] = {
    "BrickLinkPriceGuideURL",
    "BrickLinkMonthlySalesNew",
    "BrickLinkMonthlySalesUsed",
    "BrickLinkTransactionsNew",
    "BrickLinkTransactionsUsed",
    "BrickLinkTransactionsNewCount",
    "BrickLinkTransactionsUsedCount",
    "PriceForecastMethod",
    "BrickLinkPriceGuideCurrency",
    "BrickLink6MSoldNewTimesSold",
    "BrickLink6MSoldNewTotalQty",
    "BrickLink6MSoldNewMinPrice",
    "BrickLink6MSoldNewAvgPrice",
    "BrickLink6MSoldNewQtyAvgPrice",
    "BrickLink6MSoldNewMaxPrice",
    "BrickLink6MSoldUsedTimesSold",
    "BrickLink6MSoldUsedTotalQty",
    "BrickLink6MSoldUsedMinPrice",
    "BrickLink6MSoldUsedAvgPrice",
    "BrickLink6MSoldUsedQtyAvgPrice",
    "BrickLink6MSoldUsedMaxPrice",
    "BrickLinkCurrentNewTotalLots",
    "BrickLinkCurrentNewTotalQty",
    "BrickLinkCurrentNewMinPrice",
    "BrickLinkCurrentNewAvgPrice",
    "BrickLinkCurrentNewQtyAvgPrice",
    "BrickLinkCurrentNewMaxPrice",
    "BrickLinkCurrentUsedTotalLots",
    "BrickLinkCurrentUsedTotalQty",
    "BrickLinkCurrentUsedMinPrice",
    "BrickLinkCurrentUsedAvgPrice",
    "BrickLinkCurrentUsedQtyAvgPrice",
    "BrickLinkCurrentUsedMaxPrice",
    "BrickLinkLatestSaleNewMonth",
    "BrickLinkLatestSaleNewPrice",
    "BrickLinkLatestSaleUsedMonth",
    "BrickLinkLatestSaleUsedPrice",
    "CurrentNewVsRRPPercent",
    "CurrentRRPBaseline",
    "CurrentRRPBaselineCurrency",
    "PriceForecast2YNew",
    "PriceForecast5YNew",
    "PriceForecast2YUsed",
    "PriceForecast5YUsed",
    "BrickLinkNewPriceRangeMin",
    "BrickLinkNewPriceRangeMax",
    "BrickLinkUsedPriceRangeMin",
    "BrickLinkUsedPriceRangeMax",
    "BrickLinkCurrentListingsNew",
    "BrickLinkCurrentListingsUsed",
    "PriceTrendAnnualizedNewPercent",
    "PriceTrendAnnualizedUsedPercent",
    "BrickLinkSoldPriceNew",
    "BrickLinkSoldPriceUsed",
    "MarketFetchStatus",
    "MarketLastUpdatedUTC",
    "MarketNoDataRetryAfterUTC",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-item folder catalog under dist/catalog.")
    parser.add_argument("--sets-json", default="dist/Lego Star Wars Database.json")
    parser.add_argument("--minifigs-json", default="dist/Lego-Star-Wars-Minifigure-Database.json")
    parser.add_argument("--parts-json", default="dist/parts/parts-catalog.json")
    parser.add_argument("--set-parts-index-json", default="dist/parts/set-parts-index.json")
    parser.add_argument("--set-parts-dir", default="dist/parts/set-parts")
    parser.add_argument("--market-details-dir", default="dist/market-details")
    parser.add_argument("--piece-live-pricing-json", default="dist/parts/piece-live-pricing.json")
    parser.add_argument("--output-dir", default="dist/catalog")
    parser.add_argument("--full-rebuild", action="store_true")
    parser.add_argument("--changed-state-json", default="")
    parser.add_argument(
        "--item-type",
        choices=["all", "set", "minifig", "piece"],
        default="all",
        help="Build all folders or only a single item type.",
    )
    parser.add_argument(
        "--number",
        action="append",
        default=[],
        help="Optional item number/part number to rebuild. Can be repeated.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


class BuildStats:
    def __init__(self) -> None:
        self.written = 0
        self.removed = 0
        self.processed_sets = 0
        self.processed_minifigs = 0
        self.processed_pieces = 0



def load_json_array(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [row for row in data if isinstance(row, dict)]



def load_json_object(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data



def safe_folder_name(value: Any) -> str:
    text = market.collapse_ws(value)
    if not text:
        return "Unknown"
    text = text.replace("/", " - ").replace("\\", " - ")
    text = re.sub(r'[:*?"<>|]+', "-", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    return text or "Unknown"



def item_folder_name(number: Any, name: Any) -> str:
    number_text = market.collapse_ws(number)
    name_text = market.collapse_ws(name)
    if number_text and name_text:
        return safe_folder_name(f"{number_text}-{name_text}")
    if number_text:
        return safe_folder_name(number_text)
    if name_text:
        return safe_folder_name(name_text)
    return "Unknown"



def resolved_theme_components(
    *,
    model_type: str,
    theme_group: Optional[str],
    theme: Optional[str],
    subtheme: Optional[str],
    category_fallback: Optional[str],
) -> Tuple[str, str]:
    def cleaned(value: Optional[str]) -> str:
        return market.collapse_ws(value)

    def append_unique(value: str, into: List[str]) -> None:
        if not value:
            return
        if any(existing.casefold() == value.casefold() for existing in into):
            return
        into.append(value)

    raw_theme_group = cleaned(theme_group)
    raw_theme = cleaned(theme)
    raw_subtheme = cleaned(subtheme)
    raw_category = cleaned(category_fallback)
    normalized_type = cleaned(model_type).lower()
    is_minifigure = normalized_type in {"minifigure", "minifigures", "minifig"}

    if is_minifigure:
        primary_theme = raw_theme or raw_theme_group or raw_category or "Uncategorized"
        secondary: List[str] = []
        append_unique(raw_subtheme, secondary)
        return (primary_theme, " • ".join(secondary))

    normalized_theme_group = raw_theme_group.lower()
    normalized_theme = raw_theme.lower()
    normalized_subtheme = raw_subtheme.lower()
    should_flatten_ucs = normalized_theme in {
        "ultimate collector series",
        "star wars ultimate collector series",
    }

    if raw_theme_group:
        primary_theme = raw_theme_group
    elif should_flatten_ucs:
        primary_theme = "Star Wars"
    else:
        primary_theme = raw_theme or raw_category or "Uncategorized"

    secondary: List[str] = []
    is_star_wars = primary_theme.casefold() == "star wars"
    if should_flatten_ucs and (is_star_wars or not normalized_theme_group):
        if raw_subtheme and normalized_subtheme != "ultimate collector series" and raw_subtheme.casefold() != primary_theme.casefold():
            append_unique(raw_subtheme, secondary)
        else:
            append_unique("Ultimate Collector Series", secondary)
        return (primary_theme, " • ".join(secondary))

    if raw_theme and raw_theme.casefold() != primary_theme.casefold():
        append_unique(raw_theme, secondary)
    if raw_subtheme and raw_subtheme.casefold() != primary_theme.casefold():
        append_unique(raw_subtheme, secondary)
    return (primary_theme, " • ".join(secondary))



def parse_number_tokens(value: Any) -> List[str]:
    text = market.collapse_ws(value)
    if not text:
        return []
    tokens: List[str] = []
    seen: Set[str] = set()
    for raw in re.split(r"[,;|]", text):
        token = market.collapse_ws(raw)
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        tokens.append(token)
    return tokens



def maybe_write_json(path: Path, payload: Any, stats: BuildStats) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = path.read_text(encoding="utf-8") if path.exists() else None
    if previous == content:
        return
    path.write_text(content, encoding="utf-8")
    stats.written += 1



def maybe_remove(path: Path, stats: BuildStats) -> None:
    if path.exists():
        path.unlink()
        stats.removed += 1



def compact_lookup_key(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", value.lower())



def market_detail_candidates(item_type: str, number: str) -> List[str]:
    raw = market.collapse_ws(number).lower()
    if not raw:
        return []
    candidates: List[str] = []
    seen: Set[str] = set()

    def add(value: str) -> None:
        key = market.collapse_ws(value).lower()
        if not key or key in seen:
            return
        seen.add(key)
        candidates.append(key)

    add(raw)
    if item_type == "set":
        base = re.sub(r"-[0-9]+$", "", raw)
        if base != raw:
            add(base)
    return candidates



def market_detail_path_candidates(root: Path, item_type: str, number: str) -> List[Path]:
    kind = "minifigures" if item_type == "minifig" else "sets"
    paths: List[Path] = []
    seen: Set[str] = set()
    for candidate in market_detail_candidates(item_type, number):
        compact = compact_lookup_key(candidate)
        bucket = "misc"
        if compact:
            bucket = f"{compact}0" if len(compact) == 1 else compact[:2]
        variants = [candidate]
        if compact and compact != candidate:
            variants.append(compact)
        for value in variants:
            for path in [root / kind / bucket / f"{value}.json", root / kind / f"{value}.json"]:
                key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                paths.append(path)
    return paths



def load_market_detail(root: Path, item_type: str, number: str) -> Optional[Dict[str, Any]]:
    for path in market_detail_path_candidates(root, item_type, number):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0]
    return None



def extract_market_payload_from_row(row: Dict[str, Any], *, number_key: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"Number": number_key}
    for key in MARKET_DETAIL_FIELDS:
        if key in row:
            payload[key] = row.get(key)
    return payload



def build_piece_market_payload(
    row: Dict[str, Any],
    live_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    part_num = market.collapse_ws(row.get("part_num"))
    bricklink_part_num = market.collapse_ws(row.get("bricklink_part_num"))
    for key in [part_num.lower(), bricklink_part_num.lower()]:
        if key and key in live_lookup:
            return live_lookup[key]
    return {
        "part_num": part_num,
        "bricklink_part_num": bricklink_part_num or None,
        "average_listing_price": None,
        "min_listing_price": None,
        "max_listing_price": None,
        "listing_count": None,
        "currency_code": None,
        "price_guide_url": None,
        "last_updated_utc": row.get("MarketLastUpdatedUTC"),
    }



def load_set_parts_entries(index_lookup: Dict[str, str], set_parts_dir: Path, set_number: str) -> Optional[List[Dict[str, Any]]]:
    relative_path = index_lookup.get(set_number.lower())
    if not relative_path:
        return None
    path = set_parts_dir / relative_path
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, list) else None



def build_minifig_template_parts(
    index_lookup: Dict[str, str],
    set_parts_dir: Path,
    *,
    target_numbers: Optional[Set[str]],
    verbose: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    templates: Dict[str, List[Dict[str, Any]]] = {}
    for relative_path in index_lookup.values():
        path = set_parts_dir / relative_path
        if not path.exists():
            continue
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(entries, list):
            continue
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if not entry.get("from_minifigure"):
                continue
            minifig_number = market.collapse_ws(entry.get("minifigure_number")).lower()
            if not minifig_number:
                continue
            if target_numbers is not None and minifig_number not in target_numbers:
                continue
            grouped.setdefault(minifig_number, []).append(entry)
        for minifig_number, grouped_entries in grouped.items():
            current = templates.get(minifig_number)
            if current is None or len(grouped_entries) > len(current):
                templates[minifig_number] = grouped_entries
    if verbose:
        print(f"[ItemCatalog] built minifig templates for {len(templates)} minifigs", flush=True)
    return templates



def load_changed_targets(path: Path) -> Dict[str, Set[str]]:
    payload = load_json_object(path)
    keys = {
        "set": ["lastUpdatedSetCodes", "last_updated_set_numbers"],
        "minifig": ["lastUpdatedMinifigNumbers", "last_updated_minifig_numbers"],
        "piece": ["lastUpdatedPartNumbers", "last_updated_part_numbers"],
    }
    output: Dict[str, Set[str]] = {"set": set(), "minifig": set(), "piece": set()}
    for item_type, names in keys.items():
        for name in names:
            raw = payload.get(name)
            if not isinstance(raw, list):
                continue
            for value in raw:
                text = market.collapse_ws(value)
                if text:
                    output[item_type].add(text.lower())
    return output



def build_set_folder(
    row: Dict[str, Any],
    *,
    output_dir: Path,
    market_details_dir: Path,
    index_lookup: Dict[str, str],
    set_parts_dir: Path,
    minifig_lookup: Dict[str, Dict[str, Any]],
    stats: BuildStats,
) -> None:
    number = market.normalize_set_code(row.get("Number"), row.get("Variant"))
    if not number:
        return
    theme, _ = resolved_theme_components(
        model_type="set",
        theme_group=row.get("ThemeGroup"),
        theme=row.get("Theme"),
        subtheme=row.get("Subtheme"),
        category_fallback=row.get("Category"),
    )
    item_name = row.get("SetName") or row.get("name") or row.get("setName")
    item_dir = output_dir / "sets" / safe_folder_name(theme) / item_folder_name(number, item_name)
    maybe_write_json(item_dir / "item.json", row, stats)

    market_payload = load_market_detail(market_details_dir, "set", number) or extract_market_payload_from_row(row, number_key=number)
    maybe_write_json(item_dir / "market.json", market_payload, stats)

    parts_entries = load_set_parts_entries(index_lookup, set_parts_dir, number)
    if parts_entries is not None:
        maybe_write_json(item_dir / "parts.json", parts_entries, stats)
    else:
        maybe_remove(item_dir / "parts.json", stats)

    minifig_numbers = parse_number_tokens(row.get("MinifigNumbers"))
    related_minifigs = [minifig_lookup[key] for key in [value.lower() for value in minifig_numbers] if key in minifig_lookup]
    if related_minifigs:
        maybe_write_json(item_dir / "minifigures.json", related_minifigs, stats)
    else:
        maybe_remove(item_dir / "minifigures.json", stats)

    stats.processed_sets += 1



def build_minifig_folder(
    row: Dict[str, Any],
    *,
    output_dir: Path,
    market_details_dir: Path,
    set_lookup: Dict[str, Dict[str, Any]],
    template_parts_lookup: Dict[str, List[Dict[str, Any]]],
    stats: BuildStats,
) -> None:
    number = market.collapse_ws(row.get("Number"))
    if not number:
        return
    theme, _ = resolved_theme_components(
        model_type="minifigure",
        theme_group=row.get("ThemeGroup"),
        theme=row.get("Theme"),
        subtheme=row.get("Subtheme"),
        category_fallback=row.get("Category"),
    )
    item_name = row.get("Minifig name") or row.get("name") or row.get("minifigName")
    item_dir = output_dir / "minifigs" / safe_folder_name(theme) / item_folder_name(number, item_name)
    maybe_write_json(item_dir / "item.json", row, stats)

    market_payload = load_market_detail(market_details_dir, "minifig", number) or extract_market_payload_from_row(row, number_key=number)
    maybe_write_json(item_dir / "market.json", market_payload, stats)

    appears_tokens = parse_number_tokens(row.get("AppearsInSetNumbers"))
    exclusive = market.collapse_ws(row.get("ExclusiveToSetNumber"))
    if exclusive:
        appears_tokens.append(exclusive)
    seen: Set[str] = set()
    appears_rows: List[Dict[str, Any]] = []
    for token in appears_tokens:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        if key in set_lookup:
            appears_rows.append(set_lookup[key])
    if appears_rows:
        maybe_write_json(item_dir / "appears-in-sets.json", appears_rows, stats)
    else:
        maybe_remove(item_dir / "appears-in-sets.json", stats)

    template = template_parts_lookup.get(number.lower())
    if template:
        maybe_write_json(item_dir / "parts.json", template, stats)
    else:
        maybe_remove(item_dir / "parts.json", stats)

    stats.processed_minifigs += 1



def build_piece_folder(
    row: Dict[str, Any],
    *,
    output_dir: Path,
    live_lookup: Dict[str, Dict[str, Any]],
    stats: BuildStats,
) -> None:
    part_num = market.collapse_ws(row.get("part_num"))
    if not part_num:
        return
    item_dir = output_dir / "pieces" / item_folder_name(part_num, row.get("name"))
    maybe_write_json(item_dir / "item.json", row, stats)
    maybe_write_json(item_dir / "market.json", build_piece_market_payload(row, live_lookup), stats)
    stats.processed_pieces += 1



def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    output_dir = Path(args.output_dir)
    if args.full_rebuild and output_dir.exists():
        shutil.rmtree(output_dir)

    sets_rows = load_json_array(Path(args.sets_json))
    minifigs_rows = load_json_array(Path(args.minifigs_json))
    parts_rows = load_json_array(Path(args.parts_json))
    set_parts_index_raw = load_json_object(Path(args.set_parts_index_json))
    set_parts_index = {
        market.collapse_ws(key).lower(): market.collapse_ws(value)
        for key, value in set_parts_index_raw.items()
        if market.collapse_ws(key) and market.collapse_ws(value)
    }
    set_parts_dir = Path(args.set_parts_dir)
    market_details_dir = Path(args.market_details_dir)

    live_rows = load_json_array(Path(args.piece_live_pricing_json))
    live_lookup: Dict[str, Dict[str, Any]] = {}
    for row in live_rows:
        part_key = market.collapse_ws(row.get("part_num")).lower()
        bricklink_key = market.collapse_ws(row.get("bricklink_part_num")).lower()
        if part_key and part_key not in live_lookup:
            live_lookup[part_key] = row
        if bricklink_key and bricklink_key not in live_lookup:
            live_lookup[bricklink_key] = row

    set_lookup: Dict[str, Dict[str, Any]] = {}
    for row in sets_rows:
        key = market.normalize_set_code(row.get("Number"), row.get("Variant")).lower()
        if key:
            set_lookup[key] = row
    minifig_lookup: Dict[str, Dict[str, Any]] = {}
    for row in minifigs_rows:
        key = market.collapse_ws(row.get("Number")).lower()
        if key:
            minifig_lookup[key] = row
    part_lookup: Dict[str, Dict[str, Any]] = {}
    for row in parts_rows:
        part_key = market.collapse_ws(row.get("part_num")).lower()
        bricklink_key = market.collapse_ws(row.get("bricklink_part_num")).lower()
        if part_key and part_key not in part_lookup:
            part_lookup[part_key] = row
        if bricklink_key and bricklink_key not in part_lookup:
            part_lookup[bricklink_key] = row

    target_numbers: Dict[str, Set[str]] = {"set": set(), "minifig": set(), "piece": set()}
    for raw in args.number:
        value = market.collapse_ws(raw).lower()
        if not value:
            continue
        if args.item_type == "set":
            target_numbers["set"].add(value)
        elif args.item_type == "minifig":
            target_numbers["minifig"].add(value)
        elif args.item_type == "piece":
            target_numbers["piece"].add(value)
        else:
            target_numbers["set"].add(value)
            target_numbers["minifig"].add(value)
            target_numbers["piece"].add(value)

    if args.changed_state_json:
        changed = load_changed_targets(Path(args.changed_state_json))
        for key in target_numbers:
            target_numbers[key].update(changed[key])

    build_sets = args.item_type in {"all", "set"}
    build_minifigs = args.item_type in {"all", "minifig"}
    build_pieces = args.item_type in {"all", "piece"}

    if build_minifigs:
        minifig_targets = target_numbers["minifig"] or None
        template_parts_lookup = build_minifig_template_parts(
            set_parts_index,
            set_parts_dir,
            target_numbers=minifig_targets,
            verbose=args.verbose,
        )
    else:
        template_parts_lookup = {}

    stats = BuildStats()

    if build_sets:
        target_sets = target_numbers["set"]
        selected_sets = sets_rows if not target_sets else [
            row for row in sets_rows
            if market.normalize_set_code(row.get("Number"), row.get("Variant")).lower() in target_sets
        ]
        for row in selected_sets:
            build_set_folder(
                row,
                output_dir=output_dir,
                market_details_dir=market_details_dir,
                index_lookup=set_parts_index,
                set_parts_dir=set_parts_dir,
                minifig_lookup=minifig_lookup,
                stats=stats,
            )

    if build_minifigs:
        target_minifigs = target_numbers["minifig"]
        selected_minifigs = minifigs_rows if not target_minifigs else [
            row for row in minifigs_rows
            if market.collapse_ws(row.get("Number")).lower() in target_minifigs
        ]
        for row in selected_minifigs:
            build_minifig_folder(
                row,
                output_dir=output_dir,
                market_details_dir=market_details_dir,
                set_lookup=set_lookup,
                template_parts_lookup=template_parts_lookup,
                stats=stats,
            )

    if build_pieces:
        target_pieces = target_numbers["piece"]
        if target_pieces:
            selected_pieces = []
            seen_ids: Set[int] = set()
            for target in target_pieces:
                row = part_lookup.get(target)
                if row is None:
                    continue
                row_id = id(row)
                if row_id in seen_ids:
                    continue
                seen_ids.add(row_id)
                selected_pieces.append(row)
        else:
            selected_pieces = parts_rows
        for row in selected_pieces:
            build_piece_folder(
                row,
                output_dir=output_dir,
                live_lookup=live_lookup,
                stats=stats,
            )

    summary = {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fullRebuild": bool(args.full_rebuild),
        "processed": {
            "sets": stats.processed_sets,
            "minifigs": stats.processed_minifigs,
            "pieces": stats.processed_pieces,
        },
    }
    maybe_write_json(output_dir / "index.json", summary, stats)

    print(
        (
            f"[ItemCatalog] sets={stats.processed_sets} minifigs={stats.processed_minifigs} "
            f"pieces={stats.processed_pieces} written={stats.written} removed={stats.removed}"
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
