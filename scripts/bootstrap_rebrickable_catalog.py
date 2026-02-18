#!/usr/bin/env python3
"""Bootstrap/expand set catalog from Rebrickable CSV dumps.

This script is designed for large-scale seeding so the daily API jobs only
need incremental updates. It inserts missing sets and can fill empty metadata
on existing rows.
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
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap set catalog using Rebrickable CSV dumps.")
    parser.add_argument("--sets-json", default="dist/Lego Star Wars Database.json", help="Target sets JSON path.")
    parser.add_argument("--themes-json", default="dist/Themes.json", help="Target themes JSON path.")
    parser.add_argument(
        "--minifigs-json",
        default="dist/Lego-Star-Wars-Minifigure-Database.json",
        help="Minifigs JSON path (used for Theme index counts).",
    )
    parser.add_argument("--themes-csv-url", default=DEFAULT_THEMES_CSV_URL, help="Rebrickable themes CSV (.gz) URL.")
    parser.add_argument("--sets-csv-url", default=DEFAULT_SETS_CSV_URL, help="Rebrickable sets CSV (.gz) URL.")
    parser.add_argument("--timeout", type=float, default=45.0, help="HTTP timeout.")
    parser.add_argument("--retries", type=int, default=4, help="Retry count.")
    parser.add_argument(
        "--fill-missing-fields",
        action="store_true",
        help="Fill empty fields on existing rows (Theme/Pieces/link/productImage).",
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

    log(f"[Start] existing sets={len(existing_sets)}", enabled=cfg.verbose)

    session = requests.Session()
    theme_rows = download_gz_csv(session, args.themes_csv_url, cfg, label="Themes CSV")
    set_rows = download_gz_csv(session, args.sets_csv_url, cfg, label="Sets CSV")

    theme_nodes = build_theme_maps(theme_rows)
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
        "link",
        "instructionsLink",
        "productImage",
        "type",
    ):
        template.setdefault(required, None if required not in {"link", "instructionsLink", "productImage", "type"} else "")

    created = 0
    patched = 0
    skipped = 0
    next_id = next_set_id(existing_sets)

    for csv_row in set_rows:
        parsed = parse_set_num(csv_row.get("set_num", ""))
        if not parsed:
            skipped += 1
            continue

        number_key, variant, number_out = parsed
        key = (number_key, variant)

        set_name = collapse_ws(csv_row.get("name"))
        year_from = parse_int(csv_row.get("year"))
        pieces = parse_int(csv_row.get("num_parts"))
        theme_id = parse_int(csv_row.get("theme_id"))
        theme_name, theme_group = resolve_theme(theme_id, theme_nodes)
        product_image = collapse_ws(csv_row.get("img_url") or csv_row.get("set_img_url"))
        set_code = f"{number_key}-{variant}"
        default_link = f"https://www.bricklink.com/v2/catalog/catalogitem.page?S={set_code}#T=P"

        if key in existing_by_key:
            if not args.fill_missing_fields:
                continue
            row = existing_by_key[key]
            changed = False
            if (not collapse_ws(row.get("Theme"))) or collapse_ws(row.get("Theme")).lower() == "unknown":
                row["Theme"] = theme_name
                changed = True
            if (not collapse_ws(row.get("ThemeGroup"))) or collapse_ws(row.get("ThemeGroup")).lower() == "unknown":
                row["ThemeGroup"] = theme_group
                changed = True
            if not collapse_ws(row.get("SetName")) and set_name:
                row["SetName"] = set_name
                changed = True
            if parse_int(row.get("YearFrom")) is None and year_from is not None:
                row["YearFrom"] = year_from
                changed = True
            if parse_int(row.get("Pieces")) is None and pieces is not None:
                row["Pieces"] = pieces
                changed = True
            if not collapse_ws(row.get("link")):
                row["link"] = default_link
                changed = True
            if not collapse_ws(row.get("productImage")) and product_image:
                row["productImage"] = product_image
                changed = True
            if not collapse_ws(row.get("type")):
                row["type"] = "Set"
                changed = True
            if changed:
                patched += 1
            continue

        row = dict(template)
        row.update(
            {
                "Number": number_out,
                "Variant": variant,
                "SetName": set_name or set_code,
                "YearFrom": year_from,
                "Theme": theme_name,
                "ThemeGroup": theme_group,
                "Pieces": pieces,
                "SetID": next_id,
                "link": default_link,
                "instructionsLink": collapse_ws(row.get("instructionsLink")) or "",
                "productImage": product_image or "",
                "ImageFilename": set_code,
                "type": "Set",
            }
        )
        next_id += 1
        existing_sets.append(row)
        existing_by_key[key] = row
        created += 1

    existing_sets.sort(key=sort_key)
    themes_index = build_theme_index(existing_sets, existing_minifigs)

    print(
        (
            f"[Bootstrap] scanned={len(set_rows)} created={created} patched={patched} "
            f"skipped={skipped} total_sets={len(existing_sets)}"
        ),
        flush=True,
    )
    print(f"[Themes] total={len(themes_index)}", flush=True)

    if args.dry_run:
        print("[Dry run] no files written", flush=True)
        return 0

    write_json_array(sets_path, existing_sets)
    write_json_array(themes_path, themes_index)
    print("[Write] sets/themes updated", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
