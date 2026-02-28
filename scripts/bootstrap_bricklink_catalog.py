#!/usr/bin/env python3
"""Bootstrap/refresh catalogs from BrickLink catalog download feeds.

This script uses BrickLink catalog download endpoints (XML preferred, CSV fallback)
for Sets and Minifigures, with no HTML crawling.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests


DEFAULT_SETS_XML_URL = "https://www.bricklink.com/downloadxml.asp?a=a&itemType=S"
DEFAULT_SETS_CSV_URL = "https://www.bricklink.com/catalogDownload.asp?a=a&itemType=S"
DEFAULT_MINIFIGS_XML_URL = "https://www.bricklink.com/downloadxml.asp?a=a&itemType=M"
DEFAULT_MINIFIGS_CSV_URL = "https://www.bricklink.com/catalogDownload.asp?a=a&itemType=M"

SET_NUM_RE = re.compile(r"^(.+)-([0-9]+)$")


@dataclass
class FetchConfig:
    timeout: float
    retries: int
    verbose: bool
    cookie: str


@dataclass
class SourceStats:
    parsed: int = 0
    created: int = 0
    patched: int = 0
    skipped: int = 0


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


def ordered_columns(rows: List[Dict[str, Any]]) -> List[str]:
    columns: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)
    return columns


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level array in {path}")
    return [row for row in data if isinstance(row, dict)]


def write_json_array(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def decode_bytes(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def clean_tag(tag: str) -> str:
    if "}" in tag:
        tag = tag.split("}", 1)[1]
    token = collapse_ws(tag).lower()
    token = re.sub(r"[^a-z0-9]+", "_", token).strip("_")
    return token


def first_of(record: Dict[str, str], keys: Iterable[str]) -> str:
    for key in keys:
        value = record.get(key, "")
        if collapse_ws(value):
            return collapse_ws(value)
    return ""


def parse_set_code(raw: Any) -> Optional[Tuple[str, int, Any, str]]:
    text = collapse_ws(raw).upper()
    if not text:
        return None

    variant = 1
    number_token = text
    match = SET_NUM_RE.match(text)
    if match:
        number_token = collapse_ws(match.group(1)).upper()
        variant = parse_int(match.group(2)) or 1

    if not number_token:
        return None

    if number_token.isdigit():
        number_out: Any = int(number_token)
        number_key = str(int(number_token))
    else:
        number_out = number_token
        number_key = number_token

    set_code = f"{number_key}-{variant}"
    return (number_key, variant, number_out, set_code)


def normalize_set_key(number: Any, variant: Any) -> str:
    parsed = parse_set_code(f"{collapse_ws(number)}-{parse_int(variant) or 1}")
    if parsed is None:
        token = collapse_ws(number).upper()
        v = parse_int(variant) or 1
        return f"{token}-{v}"
    return parsed[3]


def minifig_key(value: Any) -> str:
    return collapse_ws(value).lower()


def derive_character_name(value: str) -> str:
    text = collapse_ws(value)
    if not text:
        return ""
    if " - " in text:
        left = collapse_ws(text.split(" - ", 1)[0])
        if left:
            return left
    return text


def fetch_feed(session: requests.Session, cfg: FetchConfig, urls: List[str], label: str) -> Tuple[bytes, str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
    }
    if cfg.cookie:
        headers["Cookie"] = cfg.cookie

    errors: List[str] = []
    for url in urls:
        attempts = max(1, cfg.retries + 1)
        for attempt in range(1, attempts + 1):
            try:
                response = session.get(url, timeout=cfg.timeout, headers=headers)
            except requests.RequestException as exc:
                if attempt == attempts:
                    errors.append(f"{url}: request error: {exc}")
                else:
                    time.sleep(min(10.0, attempt * 1.5))
                continue

            if response.status_code >= 500:
                if attempt == attempts:
                    errors.append(f"{url}: HTTP {response.status_code}")
                else:
                    time.sleep(min(15.0, attempt * 2.0))
                continue

            if response.status_code >= 400:
                errors.append(f"{url}: HTTP {response.status_code}")
                break

            body = response.content
            if not body:
                errors.append(f"{url}: empty response")
                break

            text_head = decode_bytes(body[:4000]).lower()
            if "<html" in text_head and "bricklink" in text_head and ("login" in text_head or "sign in" in text_head):
                errors.append(f"{url}: received login HTML; set --cookie or BRICKLINK_CATALOG_COOKIE")
                break

            log(f"[{label}] downloaded bytes={len(body)} from {url}", enabled=cfg.verbose)
            return (body, url)

    detail = " | ".join(errors[-6:]) if errors else "unknown failure"
    raise RuntimeError(f"{label}: failed to fetch feed ({detail})")


def parse_xml_records(raw: bytes) -> List[Dict[str, str]]:
    text = decode_bytes(raw).lstrip("\ufeff\n\r\t ")
    if not text.startswith("<"):
        return []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []

    item_nodes = root.findall(".//ITEM")
    if not item_nodes:
        # fallback for alternate wrappers
        for node in root.iter():
            if len(list(node)) < 2:
                continue
            child_tags = {clean_tag(child.tag) for child in list(node)}
            if {"itemid", "item_no", "number", "no"} & child_tags:
                item_nodes.append(node)

    records: List[Dict[str, str]] = []
    for node in item_nodes:
        row: Dict[str, str] = {}
        for child in list(node):
            key = clean_tag(child.tag)
            if not key:
                continue
            value = collapse_ws(child.text)
            if value:
                row[key] = value
        if row:
            records.append(row)
    return records


def parse_csv_records(raw: bytes) -> List[Dict[str, str]]:
    text = decode_bytes(raw)
    lines = [line for line in text.splitlines() if collapse_ws(line)]
    if len(lines) < 2:
        return []

    sample = "\n".join(lines[:4])
    if "," not in sample and "\t" not in sample and ";" not in sample:
        return []

    # CSV sniffing can fail on odd samples; fallback to comma.
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except Exception:
        dialect = csv.excel

    reader = csv.DictReader(io.StringIO("\n".join(lines)), dialect=dialect)
    records: List[Dict[str, str]] = []
    for row in reader:
        if not isinstance(row, dict):
            continue
        normalized: Dict[str, str] = {}
        for key, value in row.items():
            k = clean_tag(str(key))
            if not k:
                continue
            v = collapse_ws(value)
            if v:
                normalized[k] = v
        if normalized:
            records.append(normalized)
    return records


def parse_feed_records(raw: bytes, *, label: str, verbose: bool) -> List[Dict[str, str]]:
    xml_records = parse_xml_records(raw)
    if xml_records:
        log(f"[{label}] parsed {len(xml_records)} XML rows", enabled=verbose)
        return xml_records

    csv_records = parse_csv_records(raw)
    if csv_records:
        log(f"[{label}] parsed {len(csv_records)} CSV rows", enabled=verbose)
        return csv_records

    raise RuntimeError(f"{label}: could not parse feed as XML or CSV")


def set_sort_key(row: Dict[str, Any]) -> Tuple[int, int, str, int]:
    year = parse_int(row.get("YearFrom"))
    if year is None:
        year = 9999

    raw_number = collapse_ws(row.get("Number")).upper()
    numeric_flag = 1 if raw_number.isdigit() else 2
    numeric_sort = str(int(raw_number)) if raw_number.isdigit() else raw_number
    variant = parse_int(row.get("Variant")) or 1
    return (year, numeric_flag, numeric_sort, variant)


def upsert_sets(existing_rows: List[Dict[str, Any]], source_records: List[Dict[str, str]], *, prune_missing: bool) -> Tuple[List[Dict[str, Any]], SourceStats]:
    stats = SourceStats(parsed=len(source_records))
    existing_by_key: Dict[str, Dict[str, Any]] = {}
    for row in existing_rows:
        key = normalize_set_key(row.get("Number"), row.get("Variant"))
        existing_by_key[key] = row

    columns = ordered_columns(existing_rows)
    template: Dict[str, Any] = {key: None for key in columns}
    template.update(
        {
            "Category": "Normal",
            "Theme": "Unknown",
            "ThemeGroup": "Unknown",
            "SetName": "",
            "link": "",
            "instructionsLink": "",
            "productImage": "",
            "type": "Set",
        }
    )

    next_set_id = 1
    for row in existing_rows:
        sid = parse_int(row.get("SetID"))
        if sid is not None:
            next_set_id = max(next_set_id, sid + 1)

    seen: set[str] = set()

    for rec in source_records:
        code_raw = first_of(rec, ["itemid", "item_no", "number", "no", "setnumber", "set_num"])
        parsed = parse_set_code(code_raw)
        if parsed is None:
            stats.skipped += 1
            continue

        number_key, variant, number_out, set_code = parsed
        seen.add(set_code)

        name = first_of(rec, ["itemname", "name", "description", "setname"]) or set_code
        year = parse_int(first_of(rec, ["year", "yearfrom", "released", "releaseyear"]))
        category = first_of(rec, ["category", "catname", "categoryname", "theme", "themename"]) or "Unknown"
        image = first_of(rec, ["image", "imageurl", "image_url", "img", "imgurl", "thumbnail", "thumbnailurl"])
        link = f"https://www.bricklink.com/v2/catalog/catalogitem.page?S={set_code}#T=P"

        existing = existing_by_key.get(set_code)
        if existing is None:
            row = dict(template)
            row.update(
                {
                    "SetID": next_set_id,
                    "Number": number_out,
                    "Variant": variant,
                    "SetName": name,
                    "YearFrom": year,
                    "Theme": category,
                    "ThemeGroup": category,
                    "Category": collapse_ws(row.get("Category")) or "Normal",
                    "link": link,
                    "instructionsLink": collapse_ws(row.get("instructionsLink")) or "",
                    "productImage": image,
                    "type": "Set",
                }
            )
            next_set_id += 1
            existing_rows.append(row)
            existing_by_key[set_code] = row
            stats.created += 1
            continue

        changed = False
        if existing.get("Number") != number_out:
            existing["Number"] = number_out
            changed = True
        if parse_int(existing.get("Variant")) != variant:
            existing["Variant"] = variant
            changed = True
        if collapse_ws(existing.get("SetName")) != name:
            existing["SetName"] = name
            changed = True
        if year is not None and parse_int(existing.get("YearFrom")) != year:
            existing["YearFrom"] = year
            changed = True
        if category and collapse_ws(existing.get("Theme")) != category:
            existing["Theme"] = category
            changed = True
        if category and collapse_ws(existing.get("ThemeGroup")) != category:
            existing["ThemeGroup"] = category
            changed = True
        if collapse_ws(existing.get("link")) != link:
            existing["link"] = link
            changed = True
        if image and collapse_ws(existing.get("productImage")) != image:
            existing["productImage"] = image
            changed = True
        if collapse_ws(existing.get("type")) != "Set":
            existing["type"] = "Set"
            changed = True
        if not collapse_ws(existing.get("Category")):
            existing["Category"] = "Normal"
            changed = True
        if collapse_ws(existing.get("instructionsLink")) == "":
            existing["instructionsLink"] = ""

        if changed:
            stats.patched += 1
        else:
            stats.skipped += 1

    if prune_missing:
        kept: List[Dict[str, Any]] = []
        for row in existing_rows:
            key = normalize_set_key(row.get("Number"), row.get("Variant"))
            if key in seen:
                kept.append(row)
        existing_rows = kept

    existing_rows.sort(key=set_sort_key)
    return existing_rows, stats


def upsert_minifigs(existing_rows: List[Dict[str, Any]], source_records: List[Dict[str, str]], *, prune_missing: bool) -> Tuple[List[Dict[str, Any]], SourceStats]:
    stats = SourceStats(parsed=len(source_records))
    existing_by_key: Dict[str, Dict[str, Any]] = {}
    for row in existing_rows:
        key = minifig_key(row.get("Number"))
        if key:
            existing_by_key[key] = row

    columns = ordered_columns(existing_rows)
    template: Dict[str, Any] = {key: None for key in columns}
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

    seen: set[str] = set()

    for rec in source_records:
        code_raw = first_of(rec, ["itemid", "item_no", "number", "no", "minifignumber", "fig_num"])
        code = minifig_key(code_raw)
        if not code:
            stats.skipped += 1
            continue

        seen.add(code)
        name = first_of(rec, ["itemname", "name", "description", "minifigname"]) or code
        year = parse_int(first_of(rec, ["year", "releaseyear", "yearfrom"]))
        category = first_of(rec, ["category", "catname", "categoryname", "theme", "themename"]) or "Unknown"
        image = first_of(rec, ["image", "imageurl", "image_url", "img", "imgurl", "thumbnail", "thumbnailurl"])

        link = f"https://www.bricklink.com/v2/catalog/catalogitem.page?M={code}#T=P"
        character_name = derive_character_name(name)
        year_text = str(year) if year is not None else ""

        existing = existing_by_key.get(code)
        if existing is None:
            row = dict(template)
            row.update(
                {
                    "Number": code,
                    "Minifig name": name,
                    "Character name": character_name,
                    "Category": category,
                    "Theme": category,
                    "Year": year_text,
                    "In sets": collapse_ws(row.get("In sets")) or "0",
                    "New": collapse_ws(row.get("New")) or "",
                    "Used": collapse_ws(row.get("Used")) or "",
                    "link": link,
                    "instructionsLink": "",
                    "productImage": image,
                    "type": "Minifigure",
                    "AppearsInSetNumbers": collapse_ws(row.get("AppearsInSetNumbers")) or "",
                }
            )
            existing_rows.append(row)
            existing_by_key[code] = row
            stats.created += 1
            continue

        changed = False
        if collapse_ws(existing.get("Minifig name")) != name:
            existing["Minifig name"] = name
            changed = True
        if collapse_ws(existing.get("Character name")) != character_name:
            existing["Character name"] = character_name
            changed = True
        if category and collapse_ws(existing.get("Category")) != category:
            existing["Category"] = category
            changed = True
        if category and collapse_ws(existing.get("Theme")) != category:
            existing["Theme"] = category
            changed = True
        if year_text and collapse_ws(existing.get("Year")) != year_text:
            existing["Year"] = year_text
            changed = True
        if collapse_ws(existing.get("link")) != link:
            existing["link"] = link
            changed = True
        if image and collapse_ws(existing.get("productImage")) != image:
            existing["productImage"] = image
            changed = True
        if collapse_ws(existing.get("type")) != "Minifigure":
            existing["type"] = "Minifigure"
            changed = True
        if collapse_ws(existing.get("instructionsLink")) == "":
            existing["instructionsLink"] = ""

        if changed:
            stats.patched += 1
        else:
            stats.skipped += 1

    if prune_missing:
        existing_rows = [row for row in existing_rows if minifig_key(row.get("Number")) in seen]

    existing_rows.sort(key=lambda row: minifig_key(row.get("Number")))
    return existing_rows, stats


def build_theme_index(set_rows: List[Dict[str, Any]], minifig_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    set_counts: Dict[str, int] = {}
    for row in set_rows:
        theme = collapse_ws(row.get("Theme")) or "Unknown"
        set_counts[theme] = set_counts.get(theme, 0) + 1

    minifig_counts: Dict[str, int] = {}
    for row in minifig_rows:
        theme = collapse_ws(row.get("Theme") or row.get("Category")) or "Unknown"
        minifig_counts[theme] = minifig_counts.get(theme, 0) + 1

    names = sorted(set(set_counts) | set(minifig_counts), key=lambda value: value.lower())
    return [
        {
            "Theme": name,
            "SetCount": set_counts.get(name, 0),
            "MinifigCount": minifig_counts.get(name, 0),
        }
        for name in names
    ]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap catalogs from BrickLink catalog downloads (XML/CSV).")
    parser.add_argument("--sets-json", default="dist/Lego Star Wars Database.json", help="Sets JSON path.")
    parser.add_argument("--minifigs-json", default="dist/Lego-Star-Wars-Minifigure-Database.json", help="Minifigs JSON path.")
    parser.add_argument("--themes-json", default="dist/Themes.json", help="Themes JSON path.")
    parser.add_argument("--sets-xml-url", default=DEFAULT_SETS_XML_URL, help="BrickLink sets XML feed URL.")
    parser.add_argument("--sets-csv-url", default=DEFAULT_SETS_CSV_URL, help="BrickLink sets CSV feed URL fallback.")
    parser.add_argument("--minifigs-xml-url", default=DEFAULT_MINIFIGS_XML_URL, help="BrickLink minifigs XML feed URL.")
    parser.add_argument("--minifigs-csv-url", default=DEFAULT_MINIFIGS_CSV_URL, help="BrickLink minifigs CSV feed URL fallback.")
    parser.add_argument("--cookie", default="", help="Optional Cookie header value for BrickLink catalog download endpoints.")
    parser.add_argument("--timeout", type=float, default=45.0, help="HTTP timeout seconds.")
    parser.add_argument("--retries", type=int, default=4, help="Retry count.")
    parser.add_argument("--prune-missing", action="store_true", help="Remove rows not present in current BrickLink feed.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    sets_path = Path(args.sets_json)
    minifigs_path = Path(args.minifigs_json)
    themes_path = Path(args.themes_json)

    if not sets_path.exists():
        print(f"Missing sets JSON: {sets_path}", file=sys.stderr)
        return 1

    existing_sets = load_json_array(sets_path)
    existing_minifigs = load_json_array(minifigs_path) if minifigs_path.exists() else []

    cfg = FetchConfig(
        timeout=max(5.0, args.timeout),
        retries=max(0, args.retries),
        verbose=bool(args.verbose),
        cookie=collapse_ws(args.cookie) or collapse_ws(__import__("os").environ.get("BRICKLINK_CATALOG_COOKIE")),
    )

    log(f"[Start] existing sets={len(existing_sets)} minifigs={len(existing_minifigs)}", enabled=cfg.verbose)

    session = requests.Session()

    sets_blob, sets_url = fetch_feed(
        session,
        cfg,
        [args.sets_xml_url, args.sets_csv_url],
        "Sets",
    )
    minifigs_blob, minifigs_url = fetch_feed(
        session,
        cfg,
        [args.minifigs_xml_url, args.minifigs_csv_url],
        "Minifigs",
    )

    set_records = parse_feed_records(sets_blob, label="Sets", verbose=cfg.verbose)
    minifig_records = parse_feed_records(minifigs_blob, label="Minifigs", verbose=cfg.verbose)

    merged_sets, set_stats = upsert_sets(existing_sets, set_records, prune_missing=bool(args.prune_missing))
    merged_minifigs, minifig_stats = upsert_minifigs(existing_minifigs, minifig_records, prune_missing=bool(args.prune_missing))
    themes = build_theme_index(merged_sets, merged_minifigs)

    print(
        (
            f"[Sets] source={sets_url} parsed={set_stats.parsed} created={set_stats.created} "
            f"patched={set_stats.patched} skipped={set_stats.skipped} total={len(merged_sets)}"
        ),
        flush=True,
    )
    print(
        (
            f"[Minifigs] source={minifigs_url} parsed={minifig_stats.parsed} created={minifig_stats.created} "
            f"patched={minifig_stats.patched} skipped={minifig_stats.skipped} total={len(merged_minifigs)}"
        ),
        flush=True,
    )
    print(f"[Themes] total={len(themes)}", flush=True)

    if args.dry_run:
        print("[Dry run] no files written", flush=True)
        return 0

    write_json_array(sets_path, merged_sets)
    write_json_array(minifigs_path, merged_minifigs)
    write_json_array(themes_path, themes)
    print("[Write] sets/minifigs/themes updated", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
