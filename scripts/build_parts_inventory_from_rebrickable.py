#!/usr/bin/env python3
"""Build app-ready parts inventory artifacts from Rebrickable CSV dumps.

Expected Rebrickable files in --rebrickable-dir:
  - parts.csv(.gz)
  - part_categories.csv(.gz)
  - colors.csv(.gz)
  - sets.csv(.gz)
  - inventories.csv(.gz)
  - inventory_parts.csv(.gz)
Optional (recommended to include minifigure component parts):
  - inventory_minifigs.csv(.gz)

Outputs in --output-dir (default: dist/parts):
  - parts-catalog.json
  - set-parts-index.json
  - set-parts/<prefix>/<set-number>.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


CSV_CANDIDATES = {
    "parts": ["parts.csv.gz", "parts.csv"],
    "part_categories": ["part_categories.csv.gz", "part_categories.csv"],
    "colors": ["colors.csv.gz", "colors.csv"],
    "sets": ["sets.csv.gz", "sets.csv"],
    "inventories": ["inventories.csv.gz", "inventories.csv"],
    "inventory_parts": ["inventory_parts.csv.gz", "inventory_parts.csv"],
    "inventory_minifigs": ["inventory_minifigs.csv.gz", "inventory_minifigs.csv"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build parts inventory JSON files from Rebrickable dumps.")
    parser.add_argument(
        "--rebrickable-dir",
        required=True,
        help="Directory containing Rebrickable CSV/CSV.GZ dump files.",
    )
    parser.add_argument(
        "--sets-json",
        default="dist/Lego Star Wars Database.json",
        help="Set catalog JSON used to scope per-set part exports.",
    )
    parser.add_argument(
        "--output-dir",
        default="dist/parts",
        help="Output directory for parts artifacts.",
    )
    parser.add_argument(
        "--limit-sets",
        type=int,
        default=0,
        help="Optional cap for exported sets (0 = no cap).",
    )
    return parser.parse_args()


def resolve_csv_path(base_dir: Path, key: str) -> Path:
    for name in CSV_CANDIDATES[key]:
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    expected = ", ".join(CSV_CANDIDATES[key])
    raise FileNotFoundError(f"Missing {key} CSV. Expected one of: {expected}")


def resolve_optional_csv_path(base_dir: Path, key: str) -> Optional[Path]:
    for name in CSV_CANDIDATES[key]:
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    return None


def open_csv_rows(path: Path) -> Iterator[Dict[str, str]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield {k: (v or "") for k, v in row.items()}
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield {k: (v or "") for k, v in row.items()}


def as_int(raw: str, default: int = 0) -> int:
    value = (raw or "").strip()
    if not value:
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def as_bool(raw: str) -> bool:
    value = (raw or "").strip().lower()
    return value in {"1", "true", "yes", "y"}


def normalized_key(raw: str) -> str:
    return (raw or "").strip().lower()


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def first_non_empty(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def canonical_target_set_number(number: object, variant: object) -> str:
    raw = str(number or "").strip()
    if not raw:
        return ""
    if re.search(r"-[0-9]+$", raw):
        return raw
    variant_value = as_int(str(variant or ""), default=1)
    if variant_value <= 0:
        variant_value = 1
    return f"{raw}-{variant_value}"


def load_target_set_numbers(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    target: List[str] = []
    for item in data:
        canonical = canonical_target_set_number(
            item.get("Number") or item.get("number"),
            item.get("Variant") or item.get("variant"),
        )
        if canonical:
            target.append(canonical)
    # Preserve order, de-dup.
    seen = set()
    ordered: List[str] = []
    for number in target:
        key = normalized_key(number)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(number)
    return ordered


def sanitize_prefix(raw_set_number: str) -> str:
    token = re.sub(r"[^a-z0-9]", "", normalized_key(raw_set_number))
    if not token:
        return "zz"
    return token[:2]


def build_lookup_tables(rebrickable_dir: Path) -> Tuple[Dict[str, Dict[str, str]], Dict[int, str], Dict[int, str]]:
    part_categories_path = resolve_csv_path(rebrickable_dir, "part_categories")
    parts_path = resolve_csv_path(rebrickable_dir, "parts")
    colors_path = resolve_csv_path(rebrickable_dir, "colors")

    category_by_id: Dict[int, str] = {}
    for row in open_csv_rows(part_categories_path):
        category_id = as_int(row.get("id", ""))
        if category_id <= 0:
            continue
        category_name = row.get("name", "").strip()
        if category_name:
            category_by_id[category_id] = category_name

    parts_by_num: Dict[str, Dict[str, str]] = {}
    for row in open_csv_rows(parts_path):
        part_num = row.get("part_num", "").strip()
        if not part_num:
            continue
        part_cat_id = as_int(row.get("part_cat_id", ""))
        category_name = category_by_id.get(part_cat_id, "")
        parts_by_num[part_num] = {
            "part_num": part_num,
            "name": row.get("name", "").strip(),
            "category": category_name,
            "part_img_url": row.get("part_img_url", "").strip(),
            "part_url": row.get("part_url", "").strip(),
        }

    colors_by_id: Dict[int, str] = {}
    for row in open_csv_rows(colors_path):
        color_id = as_int(row.get("id", ""))
        if color_id < 0:
            continue
        color_name = row.get("name", "").strip()
        if color_name:
            colors_by_id[color_id] = color_name

    return parts_by_num, category_by_id, colors_by_id


def build_inventory_selection(
    rebrickable_dir: Path,
    target_sets_ordered: List[str],
    limit_sets: int,
) -> Dict[str, int]:
    inventories_path = resolve_csv_path(rebrickable_dir, "inventories")
    target_lookup = {normalized_key(value): value for value in target_sets_ordered}

    selected_inventory_id_by_set_key: Dict[str, Tuple[int, int]] = {}
    for row in open_csv_rows(inventories_path):
        set_num = row.get("set_num", "").strip()
        set_key = normalized_key(set_num)
        if set_key not in target_lookup:
            continue

        inventory_id = as_int(row.get("id", ""))
        if inventory_id <= 0:
            continue
        version = as_int(row.get("version", ""), default=1)

        existing = selected_inventory_id_by_set_key.get(set_key)
        if existing is None or version > existing[0]:
            selected_inventory_id_by_set_key[set_key] = (version, inventory_id)

    ordered_keys = [normalized_key(value) for value in target_sets_ordered if normalized_key(value) in selected_inventory_id_by_set_key]
    if limit_sets > 0:
        ordered_keys = ordered_keys[:limit_sets]

    return {
        key: selected_inventory_id_by_set_key[key][1]
        for key in ordered_keys
    }


def build_latest_inventory_ids_by_set_key(rebrickable_dir: Path) -> Dict[str, int]:
    inventories_path = resolve_csv_path(rebrickable_dir, "inventories")
    latest_by_key: Dict[str, Tuple[int, int]] = {}
    for row in open_csv_rows(inventories_path):
        set_key = normalized_key(row.get("set_num", ""))
        if not set_key:
            continue
        inventory_id = as_int(row.get("id", ""))
        if inventory_id <= 0:
            continue
        version = as_int(row.get("version", ""), default=1)
        existing = latest_by_key.get(set_key)
        if existing is None or version > existing[0]:
            latest_by_key[set_key] = (version, inventory_id)
    return {set_key: payload[1] for set_key, payload in latest_by_key.items()}


def collapse_part_entries(entries: List[Dict[str, object]]) -> List[Dict[str, object]]:
    collapsed: Dict[Tuple[str, str, int], Dict[str, object]] = {}

    for entry in entries:
        part_num = str(entry.get("part_num", "")).strip()
        if not part_num:
            continue
        color_name = str(entry.get("color_name") or "").strip()
        is_spare = bool(entry.get("is_spare"))
        try:
            quantity = int(entry.get("quantity", 0) or 0)
        except (TypeError, ValueError):
            quantity = 0
        if quantity <= 0:
            continue

        key = (part_num.lower(), color_name.lower(), 1 if is_spare else 0)
        existing = collapsed.get(key)
        if existing is None:
            row = dict(entry)
            row["part_num"] = part_num
            row["color_name"] = color_name or None
            row["is_spare"] = is_spare
            row["quantity"] = quantity
            collapsed[key] = row
            continue

        existing["quantity"] = int(existing.get("quantity", 0) or 0) + quantity
        if not str(existing.get("image_url") or "").strip() and str(entry.get("image_url") or "").strip():
            existing["image_url"] = entry.get("image_url")
        if not str(existing.get("name") or "").strip() and str(entry.get("name") or "").strip():
            existing["name"] = entry.get("name")

        existing_from_minifigure = bool(existing.get("from_minifigure"))
        incoming_from_minifigure = bool(entry.get("from_minifigure"))
        if incoming_from_minifigure and not existing_from_minifigure:
            existing["from_minifigure"] = True
            existing["minifigure_number"] = entry.get("minifigure_number")
        elif incoming_from_minifigure and existing_from_minifigure:
            existing_minifigure = str(existing.get("minifigure_number") or "").strip()
            incoming_minifigure = str(entry.get("minifigure_number") or "").strip()
            if incoming_minifigure and existing_minifigure and incoming_minifigure != existing_minifigure:
                existing["minifigure_number"] = "Multiple"
            elif incoming_minifigure and not existing_minifigure:
                existing["minifigure_number"] = incoming_minifigure

    result = list(collapsed.values())
    result.sort(
        key=lambda item: (
            -int(item.get("quantity", 0) or 0),
            str(item.get("part_num", "")).lower(),
            str(item.get("color_name") or "").lower(),
            1 if bool(item.get("is_spare")) else 0,
        )
    )
    return result


def build_set_parts(
    rebrickable_dir: Path,
    selected_inventory_ids_by_set_key: Dict[str, int],
    parts_by_num: Dict[str, Dict[str, str]],
    colors_by_id: Dict[int, str],
) -> Dict[str, List[Dict[str, object]]]:
    inventory_parts_path = resolve_csv_path(rebrickable_dir, "inventory_parts")
    inventory_minifigs_path = resolve_optional_csv_path(rebrickable_dir, "inventory_minifigs")
    inventory_to_set = {inventory_id: set_key for set_key, inventory_id in selected_inventory_ids_by_set_key.items()}
    selected_set_inventory_ids = set(inventory_to_set.keys())
    per_set_entries: Dict[str, List[Dict[str, object]]] = {set_key: [] for set_key in selected_inventory_ids_by_set_key}

    minifig_links_by_set_key: Dict[str, List[Tuple[str, int, int]]] = {}
    if inventory_minifigs_path is None:
        print("[parts] inventory_minifigs.csv(.gz) not found; minifigure component parts are not included.")
    else:
        latest_inventory_by_set_key = build_latest_inventory_ids_by_set_key(rebrickable_dir)
        linked_count = 0
        unresolved_count = 0

        set_minifig_qty: Dict[str, Dict[str, int]] = {}
        for row in open_csv_rows(inventory_minifigs_path):
            inventory_id = as_int(row.get("inventory_id", ""))
            set_key = inventory_to_set.get(inventory_id)
            if not set_key:
                continue

            minifigure_number = (
                row.get("fig_num", "").strip()
                or row.get("set_num", "").strip()
                or row.get("minifig_num", "").strip()
            )
            if not minifigure_number:
                continue

            quantity = as_int(row.get("quantity", ""), default=1)
            if quantity <= 0:
                quantity = 1

            if set_key not in set_minifig_qty:
                set_minifig_qty[set_key] = {}
            set_minifig_qty[set_key][minifigure_number] = (
                set_minifig_qty[set_key].get(minifigure_number, 0) + quantity
            )

        for set_key, mapping in set_minifig_qty.items():
            links: List[Tuple[str, int, int]] = []
            for minifigure_number, quantity in mapping.items():
                minifigure_key = normalized_key(minifigure_number)
                minifigure_inventory_id = latest_inventory_by_set_key.get(minifigure_key, 0)
                if minifigure_inventory_id <= 0:
                    unresolved_count += 1
                    continue
                links.append((minifigure_number, quantity, minifigure_inventory_id))
                linked_count += 1
            if links:
                minifig_links_by_set_key[set_key] = links

        print(
            f"[parts] Linked {linked_count} set->minifig inventory relations"
            + (f" ({unresolved_count} unresolved)" if unresolved_count > 0 else "")
        )

    tracked_inventory_ids = set(selected_set_inventory_ids)
    for links in minifig_links_by_set_key.values():
        for _, _, inventory_id in links:
            tracked_inventory_ids.add(inventory_id)

    inventory_part_rows: Dict[int, List[Dict[str, object]]] = {inventory_id: [] for inventory_id in tracked_inventory_ids}

    for row in open_csv_rows(inventory_parts_path):
        inventory_id = as_int(row.get("inventory_id", ""))
        if inventory_id not in tracked_inventory_ids:
            continue

        quantity = as_int(row.get("quantity", ""), default=0)
        if quantity <= 0:
            continue

        part_num = row.get("part_num", "").strip()
        if not part_num:
            continue

        color_id = as_int(row.get("color_id", ""))
        color_name = colors_by_id.get(color_id, "")
        part_info = parts_by_num.get(part_num, {})
        image_url = row.get("img_url", "").strip() or part_info.get("part_img_url", "").strip()
        part_name = part_info.get("name", "").strip() or part_num

        entry = {
            "part_num": part_num,
            "name": part_name,
            "color_name": color_name or None,
            "quantity": quantity,
            "is_spare": as_bool(row.get("is_spare", "")),
            "image_url": image_url or None,
            "bricklink_part_num": part_num,
        }
        inventory_part_rows[inventory_id].append(entry)

    for set_key, set_inventory_id in selected_inventory_ids_by_set_key.items():
        combined_entries: List[Dict[str, object]] = list(inventory_part_rows.get(set_inventory_id, []))
        for minifigure_number, minifigure_quantity, minifigure_inventory_id in minifig_links_by_set_key.get(set_key, []):
            for row in inventory_part_rows.get(minifigure_inventory_id, []):
                qty = int(row.get("quantity", 0) or 0) * minifigure_quantity
                if qty <= 0:
                    continue
                entry = dict(row)
                entry["quantity"] = qty
                entry["from_minifigure"] = True
                entry["minifigure_number"] = minifigure_number
                combined_entries.append(entry)

        per_set_entries[set_key] = collapse_part_entries(combined_entries)

    return per_set_entries


def write_outputs(
    output_dir: Path,
    parts_by_num: Dict[str, Dict[str, str]],
    selected_inventory_ids_by_set_key: Dict[str, int],
    per_set_entries: Dict[str, List[Dict[str, object]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_parts_root = output_dir / "set-parts"
    set_parts_root.mkdir(parents=True, exist_ok=True)

    fallback_image_by_part: Dict[str, str] = {}
    for entries in per_set_entries.values():
        for entry in entries:
            part_num = str(entry.get("part_num", "")).strip()
            if not part_num or part_num in fallback_image_by_part:
                continue
            image_url = str(entry.get("image_url") or "").strip()
            if image_url:
                fallback_image_by_part[part_num] = image_url

    existing_catalog_path = output_dir / "parts-catalog.json"
    existing_by_part_num: Dict[str, Dict[str, object]] = {}
    if existing_catalog_path.exists():
        try:
            existing_rows = json.loads(existing_catalog_path.read_text(encoding="utf-8"))
            if isinstance(existing_rows, list):
                for row in existing_rows:
                    if not isinstance(row, dict):
                        continue
                    part_num = str(row.get("part_num", "")).strip()
                    if part_num:
                        existing_by_part_num[part_num.lower()] = row
        except Exception:
            existing_by_part_num = {}

    timestamp_now = now_iso_utc()

    catalog_rows: List[Dict[str, object]] = []
    for part_num in sorted(parts_by_num.keys(), key=lambda value: value.lower()):
        part = parts_by_num[part_num]
        image_url = part.get("part_img_url", "").strip() or fallback_image_by_part.get(part_num, "").strip()
        existing = existing_by_part_num.get(part_num.lower(), {})
        base_row = {
            "part_num": part_num,
            "name": part.get("name", "").strip() or part_num,
            "category": part.get("category", "").strip() or None,
            "image_url": image_url or None,
            "bricklink_part_num": part_num,
        }
        existing_base_row = {
            "part_num": str(existing.get("part_num", "")).strip(),
            "name": str(existing.get("name", "")).strip() or part_num,
            "category": (str(existing.get("category", "")).strip() or None),
            "image_url": (str(existing.get("image_url", "")).strip() or None),
            "bricklink_part_num": str(existing.get("bricklink_part_num", "")).strip() or part_num,
        }
        changed = not existing or base_row != existing_base_row
        added_at = first_non_empty(existing.get("CatalogDateAddedUTC"), timestamp_now)
        updated_at = timestamp_now if changed else first_non_empty(existing.get("CatalogLastUpdatedUTC"), added_at, timestamp_now)
        catalog_rows.append(
            {
                **base_row,
                "CatalogDateAddedUTC": added_at,
                "CatalogLastUpdatedUTC": updated_at,
            }
        )

    set_parts_index: Dict[str, str] = {}
    for set_key in selected_inventory_ids_by_set_key.keys():
        entries = per_set_entries.get(set_key, [])
        if not entries:
            continue

        prefix = sanitize_prefix(set_key)
        filename = f"{set_key}.json"
        relative_path = f"{prefix}/{filename}"
        destination = set_parts_root / prefix / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(entries, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        set_parts_index[set_key] = relative_path

    (output_dir / "parts-catalog.json").write_text(
        json.dumps(catalog_rows, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (output_dir / "set-parts-index.json").write_text(
        json.dumps(set_parts_index, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    print(f"[parts] Wrote {len(catalog_rows)} catalog parts")
    print(f"[parts] Wrote set parts index for {len(set_parts_index)} sets")


def main() -> None:
    args = parse_args()
    rebrickable_dir = Path(args.rebrickable_dir).expanduser().resolve()
    sets_json_path = Path(args.sets_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not rebrickable_dir.exists():
        raise FileNotFoundError(f"Rebrickable directory not found: {rebrickable_dir}")
    if not sets_json_path.exists():
        raise FileNotFoundError(f"Sets JSON not found: {sets_json_path}")

    target_sets_ordered = load_target_set_numbers(sets_json_path)
    print(f"[parts] Loaded {len(target_sets_ordered)} target sets from {sets_json_path}")

    parts_by_num, _, colors_by_id = build_lookup_tables(rebrickable_dir)
    print(f"[parts] Loaded {len(parts_by_num)} parts from Rebrickable dump")

    selected_inventory_ids_by_set_key = build_inventory_selection(
        rebrickable_dir=rebrickable_dir,
        target_sets_ordered=target_sets_ordered,
        limit_sets=max(0, int(args.limit_sets)),
    )
    print(f"[parts] Found inventories for {len(selected_inventory_ids_by_set_key)} target sets")

    per_set_entries = build_set_parts(
        rebrickable_dir=rebrickable_dir,
        selected_inventory_ids_by_set_key=selected_inventory_ids_by_set_key,
        parts_by_num=parts_by_num,
        colors_by_id=colors_by_id,
    )
    write_outputs(
        output_dir=output_dir,
        parts_by_num=parts_by_num,
        selected_inventory_ids_by_set_key=selected_inventory_ids_by_set_key,
        per_set_entries=per_set_entries,
    )


if __name__ == "__main__":
    main()
