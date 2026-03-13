#!/usr/bin/env python3
"""Bootstrap/expand set and minifigure catalogs from Rebrickable CSV dumps.

This script is designed for large-scale seeding so the daily API jobs only
need incremental updates. It inserts missing rows and can fill/refresh metadata
on existing rows without any HTML crawling.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests


DEFAULT_THEMES_CSV_URL = "https://cdn.rebrickable.com/media/downloads/themes.csv.gz"
DEFAULT_SETS_CSV_URL = "https://cdn.rebrickable.com/media/downloads/sets.csv.gz"
DEFAULT_MINIFIGS_CSV_URL = "https://cdn.rebrickable.com/media/downloads/minifigs.csv.gz"
DEFAULT_INVENTORIES_CSV_URL = "https://cdn.rebrickable.com/media/downloads/inventories.csv.gz"
DEFAULT_INVENTORY_MINIFIGS_CSV_URL = "https://cdn.rebrickable.com/media/downloads/inventory_minifigs.csv.gz"

LOCAL_CSV_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "themes": ("themes.csv.gz", "themes.csv"),
    "sets": ("sets.csv.gz", "sets.csv"),
    "minifigs": ("minifigs.csv.gz", "minifigs.csv"),
    "inventories": ("inventories.csv.gz", "inventories.csv"),
    "inventory_minifigs": ("inventory_minifigs.csv.gz", "inventory_minifigs.csv"),
}

SET_NUM_RE = re.compile(r"^(.+)-([0-9]+)$")


@dataclass
class FetchConfig:
    timeout: float
    retries: int
    verbose: bool


@dataclass
class ThemeNode:
    theme_id: int
    name: str
    parent_id: Optional[int]


@dataclass
class SetCatalogMeta:
    set_code: str
    number_key: str
    variant: int
    number_out: Any
    set_name: str
    year_from: Optional[int]
    pieces: Optional[int]
    theme_name: str
    theme_group: str
    product_image: str


@dataclass
class RebrickableMinifigAggregate:
    set_codes: set[str]
    year_from: Optional[int]
    theme_counts: Dict[str, int]
    theme_group_counts: Dict[str, int]


@dataclass
class BootstrapSummary:
    sets_scanned: int = 0
    sets_created: int = 0
    sets_patched: int = 0
    sets_skipped: int = 0
    minifigs_scanned: int = 0
    minifigs_created: int = 0
    minifigs_patched: int = 0
    minifigs_skipped: int = 0


def log(msg: str, *, enabled: bool) -> None:
    if enabled:
        print(msg, flush=True)


def collapse_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = collapse_ws(value)
    if not text:
        return None
    match = re.search(r"-?[0-9][0-9,]*", text)
    if not match:
        return None
    try:
        return int(match.group(0).replace(",", ""))
    except ValueError:
        return None


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level array in {path}")
    return [row for row in data if isinstance(row, dict)]


def write_json_array(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def ordered_columns(rows: List[Dict[str, Any]]) -> List[str]:
    columns: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)
    return columns


def normalize_number_token(value: Any) -> str:
    token = collapse_ws(value).upper()
    if token.isdigit():
        return str(int(token))
    return token


def set_key(number: Any, variant: Any) -> Tuple[str, int]:
    return (normalize_number_token(number), parse_int(variant) or 1)


def parse_set_num(raw: str) -> Optional[Tuple[str, int, Any]]:
    text = collapse_ws(raw).upper()
    if not text:
        return None
    match = SET_NUM_RE.match(text)
    if not match:
        return None
    left = collapse_ws(match.group(1)).upper()
    variant = parse_int(match.group(2)) or 1
    if not left:
        return None
    if left.isdigit():
        number_out: Any = int(left)
    else:
        number_out = left
    return (normalize_number_token(left), variant, number_out)


def minifig_key(value: Any) -> str:
    return collapse_ws(value).lower()


def minifig_sort_key(row: Dict[str, Any]) -> str:
    return minifig_key(row.get("Number"))


def minifig_numbers_string(values: Iterable[str]) -> str:
    items = [minifig_key(v) for v in values if minifig_key(v)]
    return ",".join(items)


def parse_boolish(value: Any) -> bool:
    text = collapse_ws(value).lower()
    return text in {"1", "true", "t", "y", "yes"}


def choose_top_label(counts: Dict[str, int], fallback: str = "Unknown") -> str:
    if not counts:
        return fallback
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))[0][0]


def derive_character_name(minifig_name: str) -> str:
    text = collapse_ws(minifig_name)
    if not text:
        return ""
    parts = [p.strip() for p in text.split(" - ", 1)]
    if parts and parts[0]:
        return parts[0]
    return text


def download_gz_csv(
    session: requests.Session,
    url: str,
    cfg: FetchConfig,
    *,
    label: str,
) -> List[Dict[str, str]]:
    attempts = max(1, cfg.retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(
                url,
                timeout=cfg.timeout,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                    ),
                    "Accept": "*/*",
                },
            )
        except requests.RequestException as exc:
            if attempt == attempts:
                raise RuntimeError(f"{label}: download failed: {exc}") from exc
            time.sleep(min(10.0, attempt * 1.5))
            continue

        if response.status_code >= 500:
            if attempt == attempts:
                raise RuntimeError(f"{label}: HTTP {response.status_code}")
            time.sleep(min(15.0, attempt * 2.0))
            continue
        if response.status_code >= 400:
            raise RuntimeError(f"{label}: HTTP {response.status_code}")

        try:
            with gzip.open(io.BytesIO(response.content), mode="rt", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = []
                for row in reader:
                    if isinstance(row, dict):
                        rows.append({str(k): str(v or "") for k, v in row.items()})
                log(f"[{label}] rows={len(rows)}", enabled=cfg.verbose)
                return rows
        except Exception as exc:
            if attempt == attempts:
                raise RuntimeError(f"{label}: failed to parse gzip csv: {exc}") from exc
            time.sleep(min(10.0, attempt * 1.5))
            continue

    raise RuntimeError(f"{label}: unreachable failure")


def resolve_local_csv_path(base_dir: Path, key: str) -> Path:
    for name in LOCAL_CSV_CANDIDATES[key]:
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    expected = ", ".join(LOCAL_CSV_CANDIDATES[key])
    raise FileNotFoundError(f"Missing local {key} CSV. Expected one of: {expected}")


def load_csv_rows_from_path(path: Path, *, label: str) -> List[Dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, mode="rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows: List[Dict[str, str]] = []
        for row in reader:
            if isinstance(row, dict):
                rows.append({str(k): str(v or "") for k, v in row.items()})
        return rows


def build_theme_maps(theme_rows: List[Dict[str, str]]) -> Dict[int, ThemeNode]:
    nodes: Dict[int, ThemeNode] = {}
    for row in theme_rows:
        theme_id = parse_int(row.get("id"))
        if theme_id is None:
            continue
        name = collapse_ws(row.get("name")) or "Unknown"
        parent_id = parse_int(row.get("parent_id"))
        nodes[theme_id] = ThemeNode(theme_id=theme_id, name=name, parent_id=parent_id)
    return nodes


def resolve_theme(theme_id: Optional[int], nodes: Dict[int, ThemeNode]) -> Tuple[str, str]:
    if theme_id is None:
        return ("Unknown", "Unknown")
    current = nodes.get(theme_id)
    if current is None:
        return ("Unknown", "Unknown")

    theme_name = current.name or "Unknown"
    top = current
    seen: set[int] = set()
    while top.parent_id is not None and top.parent_id not in seen:
        seen.add(top.theme_id)
        parent = nodes.get(top.parent_id)
        if parent is None:
            break
        top = parent

    group_name = top.name or theme_name or "Unknown"
    return (theme_name, group_name)


def next_set_id(rows: List[Dict[str, Any]]) -> int:
    value = 1
    for row in rows:
        sid = parse_int(row.get("SetID"))
        if sid is not None:
            value = max(value, sid + 1)
    return value


def sort_key(row: Dict[str, Any]) -> Tuple[int, int, str, int]:
    year = parse_int(row.get("YearFrom"))
    if year is None:
        year = 9999
    number_raw = normalize_number_token(row.get("Number"))
    number_is_numeric = 1 if number_raw.isdigit() else 2
    number_sort = str(int(number_raw)) if number_raw.isdigit() else number_raw
    variant = parse_int(row.get("Variant")) or 1
    return (year, number_is_numeric, number_sort, variant)


def build_theme_index(set_rows: List[Dict[str, Any]], minifig_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    set_counts: Dict[str, int] = {}
    for row in set_rows:
        theme = collapse_ws(row.get("Theme")) or "Unknown"
        set_counts[theme] = set_counts.get(theme, 0) + 1

    minifig_counts: Dict[str, int] = {}
    for row in minifig_rows:
        raw_theme = row.get("Theme") or row.get("Category") or "Unknown"
        theme = collapse_ws(raw_theme) or "Unknown"
        minifig_counts[theme] = minifig_counts.get(theme, 0) + 1

    themes = sorted(set(set_counts) | set(minifig_counts), key=lambda v: v.lower())
    return [
        {
            "Theme": theme,
            "SetCount": set_counts.get(theme, 0),
            "MinifigCount": minifig_counts.get(theme, 0),
        }
        for theme in themes
    ]


def build_set_catalog_meta(
    set_rows: List[Dict[str, str]],
    theme_nodes: Dict[int, ThemeNode],
) -> Dict[Tuple[str, int], SetCatalogMeta]:
    result: Dict[Tuple[str, int], SetCatalogMeta] = {}
    for csv_row in set_rows:
        parsed = parse_set_num(csv_row.get("set_num", ""))
        if not parsed:
            continue
        number_key, variant, number_out = parsed
        theme_id = parse_int(csv_row.get("theme_id"))
        theme_name, theme_group = resolve_theme(theme_id, theme_nodes)
        product_image = collapse_ws(csv_row.get("img_url") or csv_row.get("set_img_url"))
        set_name = collapse_ws(csv_row.get("name")) or f"{number_key}-{variant}"
        year_from = parse_int(csv_row.get("year"))
        pieces = parse_int(csv_row.get("num_parts"))
        set_code = f"{number_key}-{variant}"
        result[(number_key, variant)] = SetCatalogMeta(
            set_code=set_code,
            number_key=number_key,
            variant=variant,
            number_out=number_out,
            set_name=set_name,
            year_from=year_from,
            pieces=pieces,
            theme_name=theme_name,
            theme_group=theme_group,
            product_image=product_image,
        )
    return result


def build_rebrickable_inventory_indexes(
    inventory_rows: List[Dict[str, str]],
    inventory_minifig_rows: List[Dict[str, str]],
    set_meta_by_key: Dict[Tuple[str, int], SetCatalogMeta],
) -> Tuple[Dict[str, List[str]], Dict[str, RebrickableMinifigAggregate]]:
    set_meta_by_code = {meta.set_code.upper(): meta for meta in set_meta_by_key.values()}

    # Rebrickable can have multiple inventories per set/version. Use the highest version, then highest ID.
    best_inventory_by_set_code: Dict[str, Tuple[int, int]] = {}
    inventory_id_to_set_code: Dict[int, str] = {}
    for row in inventory_rows:
        inventory_id = parse_int(row.get("id"))
        set_num = collapse_ws(row.get("set_num")).upper()
        if inventory_id is None or not set_num or set_num not in set_meta_by_code:
            continue
        version = parse_int(row.get("version")) or 1
        previous = best_inventory_by_set_code.get(set_num)
        if previous is None or (version, inventory_id) > previous:
            best_inventory_by_set_code[set_num] = (version, inventory_id)

    for set_code, (_, inventory_id) in best_inventory_by_set_code.items():
        inventory_id_to_set_code[inventory_id] = set_code

    set_minifig_numbers: Dict[str, List[str]] = {}
    minifig_aggregates: Dict[str, RebrickableMinifigAggregate] = {}

    for row in inventory_minifig_rows:
        inventory_id = parse_int(row.get("inventory_id"))
        if inventory_id is None:
            continue
        set_code = inventory_id_to_set_code.get(inventory_id)
        if not set_code:
            continue
        if parse_boolish(row.get("is_spare")):
            continue

        fig_number = minifig_key(row.get("fig_num"))
        if not fig_number:
            continue

        quantity = max(1, parse_int(row.get("quantity")) or 1)
        bucket = set_minifig_numbers.setdefault(set_code, [])
        bucket.extend([fig_number] * quantity)

        meta = set_meta_by_code.get(set_code)
        if meta is None:
            continue

        aggregate = minifig_aggregates.get(fig_number)
        if aggregate is None:
            aggregate = RebrickableMinifigAggregate(
                set_codes=set(),
                year_from=None,
                theme_counts={},
                theme_group_counts={},
            )
            minifig_aggregates[fig_number] = aggregate

        aggregate.set_codes.add(set_code)
        if meta.year_from is not None:
            if aggregate.year_from is None or meta.year_from < aggregate.year_from:
                aggregate.year_from = meta.year_from
        if meta.theme_name:
            aggregate.theme_counts[meta.theme_name] = aggregate.theme_counts.get(meta.theme_name, 0) + 1
        if meta.theme_group:
            aggregate.theme_group_counts[meta.theme_group] = aggregate.theme_group_counts.get(meta.theme_group, 0) + 1

    return set_minifig_numbers, minifig_aggregates


def upsert_sets_from_rebrickable(
    existing_sets: List[Dict[str, Any]],
    set_meta_by_key: Dict[Tuple[str, int], SetCatalogMeta],
    set_minifig_numbers_by_code: Dict[str, List[str]],
    *,
    fill_missing_fields: bool,
    refresh_existing_fields: bool,
) -> Tuple[List[Dict[str, Any]], BootstrapSummary]:
    summary = BootstrapSummary()
    summary.sets_scanned = len(set_meta_by_key)

    existing_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in existing_sets:
        existing_by_key[set_key(row.get("Number"), row.get("Variant"))] = row

    columns = ordered_columns(existing_sets)
    template: Dict[str, Any] = {column: None for column in columns}
    template.update(
        {
            "Category": "Normal",
            "Theme": "Unknown",
            "ThemeGroup": "Unknown",
            "Image": "X",
            "instructionsLink": "",
            "link": "",
            "productImage": "",
            "type": "Set",
        }
    )
    for required in (
        "Number",
        "Variant",
        "SetName",
        "YearFrom",
        "Theme",
        "ThemeGroup",
        "Pieces",
        "Minifigs",
        "MinifigNumbers",
        "link",
        "instructionsLink",
        "productImage",
        "type",
    ):
        template.setdefault(required, None if required not in {"link", "instructionsLink", "productImage", "type"} else "")

    next_id = next_set_id(existing_sets)

    for key in sorted(set_meta_by_key.keys(), key=lambda k: (str(k[0]), k[1])):
        meta = set_meta_by_key[key]
        set_code = meta.set_code.upper()
        set_minifigs = set_minifig_numbers_by_code.get(set_code, [])
        minifigs_total = len(set_minifigs) if set_minifigs else None
        minifig_numbers_value = minifig_numbers_string(set_minifigs) or None
        default_link = f"https://www.bricklink.com/v2/catalog/catalogitem.page?S={meta.set_code}#T=P"

        if key in existing_by_key:
            row = existing_by_key[key]
            if not fill_missing_fields and not refresh_existing_fields:
                summary.sets_skipped += 1
                continue
            changed = False
            if refresh_existing_fields and meta.theme_name and collapse_ws(row.get("Theme")) != meta.theme_name:
                row["Theme"] = meta.theme_name
                changed = True
            elif (not collapse_ws(row.get("Theme"))) or collapse_ws(row.get("Theme")).lower() == "unknown":
                row["Theme"] = meta.theme_name
                changed = True
            if refresh_existing_fields and meta.theme_group and collapse_ws(row.get("ThemeGroup")) != meta.theme_group:
                row["ThemeGroup"] = meta.theme_group
                changed = True
            elif (not collapse_ws(row.get("ThemeGroup"))) or collapse_ws(row.get("ThemeGroup")).lower() == "unknown":
                row["ThemeGroup"] = meta.theme_group
                changed = True
            if refresh_existing_fields and meta.set_name and collapse_ws(row.get("SetName")) != meta.set_name:
                row["SetName"] = meta.set_name
                changed = True
            elif not collapse_ws(row.get("SetName")) and meta.set_name:
                row["SetName"] = meta.set_name
                changed = True
            if refresh_existing_fields and meta.year_from is not None and parse_int(row.get("YearFrom")) != meta.year_from:
                row["YearFrom"] = meta.year_from
                changed = True
            elif parse_int(row.get("YearFrom")) is None and meta.year_from is not None:
                row["YearFrom"] = meta.year_from
                changed = True
            if refresh_existing_fields and meta.pieces is not None and parse_int(row.get("Pieces")) != meta.pieces:
                row["Pieces"] = meta.pieces
                changed = True
            elif parse_int(row.get("Pieces")) is None and meta.pieces is not None:
                row["Pieces"] = meta.pieces
                changed = True
            if refresh_existing_fields and collapse_ws(row.get("link")) != default_link:
                row["link"] = default_link
                changed = True
            elif not collapse_ws(row.get("link")):
                row["link"] = default_link
                changed = True
            if refresh_existing_fields and meta.product_image and collapse_ws(row.get("productImage")) != meta.product_image:
                row["productImage"] = meta.product_image
                changed = True
            elif not collapse_ws(row.get("productImage")) and meta.product_image:
                row["productImage"] = meta.product_image
                changed = True
            if refresh_existing_fields and collapse_ws(row.get("type")) != "Set":
                row["type"] = "Set"
                changed = True
            elif not collapse_ws(row.get("type")):
                row["type"] = "Set"
                changed = True
            if refresh_existing_fields and collapse_ws(row.get("ImageFilename")) != meta.set_code:
                row["ImageFilename"] = meta.set_code
                changed = True
            # Refresh minifigure mappings from Rebrickable inventory data (CSV, not crawl).
            if minifigs_total is not None and parse_int(row.get("Minifigs")) != minifigs_total:
                row["Minifigs"] = minifigs_total
                changed = True
            if minifig_numbers_value and collapse_ws(row.get("MinifigNumbers")) != minifig_numbers_value:
                row["MinifigNumbers"] = minifig_numbers_value
                changed = True
            if changed:
                summary.sets_patched += 1
            else:
                summary.sets_skipped += 1
            continue

        row = dict(template)
        row.update(
            {
                "Number": meta.number_out,
                "Variant": meta.variant,
                "SetName": meta.set_name or meta.set_code,
                "YearFrom": meta.year_from,
                "Theme": meta.theme_name,
                "ThemeGroup": meta.theme_group,
                "Pieces": meta.pieces,
                "Minifigs": minifigs_total,
                "MinifigNumbers": minifig_numbers_value,
                "SetID": next_id,
                "link": default_link,
                "instructionsLink": collapse_ws(row.get("instructionsLink")) or "",
                "productImage": meta.product_image or "",
                "ImageFilename": meta.set_code,
                "type": "Set",
            }
        )
        next_id += 1
        existing_sets.append(row)
        existing_by_key[key] = row
        summary.sets_created += 1

    existing_sets.sort(key=sort_key)
    return existing_sets, summary


def upsert_minifigs_from_rebrickable(
    existing_minifigs: List[Dict[str, Any]],
    minifig_rows_csv: List[Dict[str, str]],
    minifig_aggregates: Dict[str, RebrickableMinifigAggregate],
    *,
    fill_missing_fields: bool,
    refresh_existing_fields: bool,
) -> Tuple[List[Dict[str, Any]], BootstrapSummary]:
    summary = BootstrapSummary()
    summary.minifigs_scanned = len(minifig_rows_csv)

    existing_by_key: Dict[str, Dict[str, Any]] = {}
    for row in existing_minifigs:
        key = minifig_key(row.get("Number"))
        if key:
            existing_by_key[key] = row

    columns = ordered_columns(existing_minifigs)
    template: Dict[str, Any] = {column: None for column in columns}
    template.update(
        {
            "Number": "",
            "Minifig name": "",
            "Character name": "",
            "Category": "Unknown",
            "Theme": "Unknown",
            "Year": "",
            "In sets": "0",
            "New": "",
            "Used": "",
            "link": "",
            "instructionsLink": "",
            "productImage": "",
            "type": "Minifigure",
            "AppearsInSetNumbers": "",
        }
    )

    for csv_row in minifig_rows_csv:
        number = minifig_key(csv_row.get("fig_num"))
        if not number:
            summary.minifigs_skipped += 1
            continue

        minifig_name = collapse_ws(csv_row.get("name"))
        product_image = collapse_ws(csv_row.get("img_url"))
        aggregate = minifig_aggregates.get(number)

        appears_in = ""
        in_sets = "0"
        year_text = ""
        theme = "Unknown"
        if aggregate is not None:
            sorted_set_codes = sorted(aggregate.set_codes)
            appears_in = ",".join(sorted_set_codes)
            in_sets = str(len(sorted_set_codes))
            if aggregate.year_from is not None:
                year_text = str(aggregate.year_from)
            theme = choose_top_label(aggregate.theme_counts, fallback="Unknown")

        category = theme
        brickset_link = f"https://brickset.com/minifigs/{number}"
        character_name = derive_character_name(minifig_name)

        if number in existing_by_key:
            row = existing_by_key[number]
            if not fill_missing_fields and not refresh_existing_fields:
                summary.minifigs_skipped += 1
                continue
            changed = False
            if refresh_existing_fields and collapse_ws(row.get("Number")) != number:
                row["Number"] = number
                changed = True
            elif not collapse_ws(row.get("Number")):
                row["Number"] = number
                changed = True
            if refresh_existing_fields and minifig_name and collapse_ws(row.get("Minifig name")) != minifig_name:
                row["Minifig name"] = minifig_name
                changed = True
            elif not collapse_ws(row.get("Minifig name")) and minifig_name:
                row["Minifig name"] = minifig_name
                changed = True
            if refresh_existing_fields and character_name and collapse_ws(row.get("Character name")) != character_name:
                row["Character name"] = character_name
                changed = True
            elif not collapse_ws(row.get("Character name")) and character_name:
                row["Character name"] = character_name
                changed = True
            if refresh_existing_fields and category and collapse_ws(row.get("Category")) != category:
                row["Category"] = category
                changed = True
            elif (not collapse_ws(row.get("Category"))) or collapse_ws(row.get("Category")).lower() == "unknown":
                row["Category"] = category
                changed = True
            if refresh_existing_fields and theme and collapse_ws(row.get("Theme")) != theme:
                row["Theme"] = theme
                changed = True
            elif (not collapse_ws(row.get("Theme"))) or collapse_ws(row.get("Theme")).lower() == "unknown":
                row["Theme"] = theme
                changed = True
            if refresh_existing_fields and year_text and collapse_ws(row.get("Year")) != year_text:
                row["Year"] = year_text
                changed = True
            elif year_text and (not collapse_ws(row.get("Year"))):
                row["Year"] = year_text
                changed = True
            if refresh_existing_fields and collapse_ws(row.get("In sets")) != in_sets:
                row["In sets"] = in_sets
                changed = True
            elif collapse_ws(row.get("In sets")) in {"", "0"} and in_sets != "0":
                row["In sets"] = in_sets
                changed = True
            if refresh_existing_fields and appears_in and collapse_ws(row.get("AppearsInSetNumbers")) != appears_in:
                row["AppearsInSetNumbers"] = appears_in
                changed = True
            elif appears_in and not collapse_ws(row.get("AppearsInSetNumbers")):
                row["AppearsInSetNumbers"] = appears_in
                changed = True
            if refresh_existing_fields and collapse_ws(row.get("link")) != brickset_link:
                row["link"] = brickset_link
                changed = True
            elif not collapse_ws(row.get("link")):
                row["link"] = brickset_link
                changed = True
            if refresh_existing_fields and product_image and collapse_ws(row.get("productImage")) != product_image:
                row["productImage"] = product_image
                changed = True
            elif not collapse_ws(row.get("productImage")) and product_image:
                row["productImage"] = product_image
                changed = True
            if refresh_existing_fields and collapse_ws(row.get("instructionsLink")) != "":
                row["instructionsLink"] = ""
                changed = True
            elif not collapse_ws(row.get("instructionsLink")):
                row["instructionsLink"] = ""
                changed = True
            if refresh_existing_fields and collapse_ws(row.get("type")) != "Minifigure":
                row["type"] = "Minifigure"
                changed = True
            elif not collapse_ws(row.get("type")):
                row["type"] = "Minifigure"
                changed = True
            if changed:
                summary.minifigs_patched += 1
            else:
                summary.minifigs_skipped += 1
            continue

        row = dict(template)
        row.update(
            {
                "Number": number,
                "Minifig name": minifig_name or number,
                "Character name": character_name,
                "Category": category,
                "Theme": theme,
                "Year": year_text,
                "In sets": in_sets,
                "New": "",
                "Used": "",
                "link": brickset_link,
                "instructionsLink": "",
                "productImage": product_image or "",
                "type": "Minifigure",
                "AppearsInSetNumbers": appears_in,
            }
        )
        existing_minifigs.append(row)
        existing_by_key[number] = row
        summary.minifigs_created += 1

    existing_minifigs.sort(key=minifig_sort_key)
    return existing_minifigs, summary


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap set and minifigure catalogs using Rebrickable CSV dumps.")
    parser.add_argument("--sets-json", default="dist/Lego Star Wars Database.json", help="Target sets JSON path.")
    parser.add_argument("--themes-json", default="dist/Themes.json", help="Target themes JSON path.")
    parser.add_argument(
        "--minifigs-json",
        default="dist/Lego-Star-Wars-Minifigure-Database.json",
        help="Minifigs JSON path (used for Theme index counts).",
    )
    parser.add_argument("--themes-csv-url", default=DEFAULT_THEMES_CSV_URL, help="Rebrickable themes CSV (.gz) URL.")
    parser.add_argument("--sets-csv-url", default=DEFAULT_SETS_CSV_URL, help="Rebrickable sets CSV (.gz) URL.")
    parser.add_argument("--minifigs-csv-url", default=DEFAULT_MINIFIGS_CSV_URL, help="Rebrickable minifigs CSV (.gz) URL.")
    parser.add_argument("--inventories-csv-url", default=DEFAULT_INVENTORIES_CSV_URL, help="Rebrickable inventories CSV (.gz) URL.")
    parser.add_argument(
        "--inventory-minifigs-csv-url",
        default=DEFAULT_INVENTORY_MINIFIGS_CSV_URL,
        help="Rebrickable inventory_minifigs CSV (.gz) URL.",
    )
    parser.add_argument("--timeout", type=float, default=45.0, help="HTTP timeout.")
    parser.add_argument("--retries", type=int, default=4, help="Retry count.")
    parser.add_argument(
        "--rebrickable-dir",
        default="",
        help="Optional local directory containing Rebrickable CSV/CSV.GZ files. Skips remote downloads when set.",
    )
    parser.add_argument(
        "--fill-missing-fields",
        action="store_true",
        help="Fill empty fields on existing rows (Theme/Pieces/link/productImage).",
    )
    parser.add_argument(
        "--refresh-existing-fields",
        action="store_true",
        help="Refresh existing set/minifigure metadata from Rebrickable when values changed.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write files.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    sets_path = Path(args.sets_json)
    themes_path = Path(args.themes_json)
    minifigs_path = Path(args.minifigs_json)

    if not sets_path.exists():
        print(f"Missing sets JSON: {sets_path}", file=sys.stderr)
        return 1

    cfg = FetchConfig(timeout=max(5.0, args.timeout), retries=max(0, args.retries), verbose=bool(args.verbose))
    existing_sets = load_json_array(sets_path)
    existing_minifigs = load_json_array(minifigs_path) if minifigs_path.exists() else []

    log(f"[Start] existing sets={len(existing_sets)} minifigs={len(existing_minifigs)}", enabled=cfg.verbose)

    local_rebrickable_dir = Path(args.rebrickable_dir).expanduser().resolve() if collapse_ws(args.rebrickable_dir) else None
    if local_rebrickable_dir and not local_rebrickable_dir.exists():
        print(f"Missing Rebrickable directory: {local_rebrickable_dir}", file=sys.stderr)
        return 1

    session = requests.Session()
    if local_rebrickable_dir is not None:
        theme_rows = load_csv_rows_from_path(resolve_local_csv_path(local_rebrickable_dir, "themes"), label="Themes CSV")
        set_rows = load_csv_rows_from_path(resolve_local_csv_path(local_rebrickable_dir, "sets"), label="Sets CSV")
        minifigs_rows = load_csv_rows_from_path(resolve_local_csv_path(local_rebrickable_dir, "minifigs"), label="Minifigs CSV")
        inventories_rows = load_csv_rows_from_path(resolve_local_csv_path(local_rebrickable_dir, "inventories"), label="Inventories CSV")
        inventory_minifigs_rows = load_csv_rows_from_path(
            resolve_local_csv_path(local_rebrickable_dir, "inventory_minifigs"),
            label="Inventory Minifigs CSV",
        )
    else:
        theme_rows = download_gz_csv(session, args.themes_csv_url, cfg, label="Themes CSV")
        set_rows = download_gz_csv(session, args.sets_csv_url, cfg, label="Sets CSV")
        minifigs_rows = download_gz_csv(session, args.minifigs_csv_url, cfg, label="Minifigs CSV")
        inventories_rows = download_gz_csv(session, args.inventories_csv_url, cfg, label="Inventories CSV")
        inventory_minifigs_rows = download_gz_csv(
            session,
            args.inventory_minifigs_csv_url,
            cfg,
            label="Inventory Minifigs CSV",
        )

    theme_nodes = build_theme_maps(theme_rows)
    set_meta_by_key = build_set_catalog_meta(set_rows, theme_nodes)
    set_minifig_numbers_by_code, minifig_aggregates = build_rebrickable_inventory_indexes(
        inventories_rows,
        inventory_minifigs_rows,
        set_meta_by_key,
    )

    existing_sets, set_summary = upsert_sets_from_rebrickable(
        existing_sets,
        set_meta_by_key,
        set_minifig_numbers_by_code,
        fill_missing_fields=bool(args.fill_missing_fields),
        refresh_existing_fields=bool(args.refresh_existing_fields),
    )
    existing_minifigs, minifig_summary = upsert_minifigs_from_rebrickable(
        existing_minifigs,
        minifigs_rows,
        minifig_aggregates,
        fill_missing_fields=bool(args.fill_missing_fields),
        refresh_existing_fields=bool(args.refresh_existing_fields),
    )

    themes_index = build_theme_index(existing_sets, existing_minifigs)

    print(
        (
            f"[Bootstrap:Sets] scanned={set_summary.sets_scanned} created={set_summary.sets_created} "
            f"patched={set_summary.sets_patched} skipped={set_summary.sets_skipped} total_sets={len(existing_sets)}"
        ),
        flush=True,
    )
    print(
        (
            f"[Bootstrap:Minifigs] scanned={minifig_summary.minifigs_scanned} created={minifig_summary.minifigs_created} "
            f"patched={minifig_summary.minifigs_patched} skipped={minifig_summary.minifigs_skipped} "
            f"total_minifigs={len(existing_minifigs)}"
        ),
        flush=True,
    )
    print(f"[Themes] total={len(themes_index)}", flush=True)

    if args.dry_run:
        print("[Dry run] no files written", flush=True)
        return 0

    write_json_array(sets_path, existing_sets)
    write_json_array(minifigs_path, existing_minifigs)
    write_json_array(themes_path, themes_index)
    print("[Write] sets/minifigs/themes updated", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
