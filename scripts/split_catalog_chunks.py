#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split LEGO catalog JSON into smaller theme/category chunks and emit a "
            "catalog-index.json manifest for app chunk loading."
        )
    )
    parser.add_argument(
        "--sets-json",
        type=Path,
        help="Path to sets JSON (array of objects).",
    )
    parser.add_argument(
        "--minifigs-json",
        type=Path,
        help="Path to minifigures JSON (array of objects).",
    )
    parser.add_argument(
        "--merged-json",
        type=Path,
        help=(
            "Path to merged JSON containing both sets and minifigs with a 'type' field. "
            "Used when --sets-json/--minifigs-json are not provided."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write chunk files into (for example: dist/chunks).",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Output manifest path (for example: dist/catalog-index.json).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help=(
            "Optional base URL prepended to manifest chunk URLs. "
            "Example: https://raw.githubusercontent.com/<owner>/<repo>/refs/heads/main"
        ),
    )
    parser.add_argument(
        "--set-group-key",
        type=str,
        default="Theme",
        help="Grouping key for sets (default: Theme).",
    )
    parser.add_argument(
        "--minifig-group-key",
        type=str,
        default="Category",
        help="Grouping key for minifigures (default: Category).",
    )
    parser.add_argument(
        "--max-items-per-chunk",
        type=int,
        default=800,
        help="Maximum records per emitted chunk file (default: 800).",
    )
    return parser.parse_args()


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array")
    return [row for row in data if isinstance(row, dict)]


def normalize_type(value: Any) -> str:
    return str(value or "").strip().lower()


def split_merged_catalog(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sets: List[Dict[str, Any]] = []
    minifigs: List[Dict[str, Any]] = []
    for row in rows:
        kind = normalize_type(row.get("type"))
        if kind == "minifigure":
            minifigs.append(row)
        else:
            sets.append(row)
    return sets, minifigs


def slugify(value: str) -> str:
    v = value.strip().lower()
    v = re.sub(r"[^a-z0-9]+", "-", v)
    v = re.sub(r"-{2,}", "-", v).strip("-")
    return v or "unknown"


def text_value(row: Dict[str, Any], key: str, fallback: str) -> str:
    raw = row.get(key)
    if raw is None:
        return fallback
    s = str(raw).strip()
    return s if s else fallback


def number_sort_key(row: Dict[str, Any]) -> Tuple[str, str]:
    number = str(row.get("Number") or "").strip()
    name = str(row.get("SetName") or row.get("Minifig name") or "").strip()
    return (number, name)


def chunk_iter(rows: List[Dict[str, Any]], max_items: int) -> Iterable[List[Dict[str, Any]]]:
    if max_items <= 0:
        yield rows
        return
    for start in range(0, len(rows), max_items):
        yield rows[start : start + max_items]


def write_chunk(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def make_manifest_url(base_url: str, manifest_path: Path, chunk_path: Path) -> str:
    # Always prefer manifest-relative paths for stable URL generation.
    try:
        relative = chunk_path.relative_to(manifest_path.parent).as_posix()
    except ValueError:
        relative = chunk_path.name
    if base_url:
        return f"{base_url.rstrip('/')}/{relative.lstrip('/')}"
    return relative


def emit_group_chunks(
    rows: List[Dict[str, Any]],
    group_key: str,
    chunk_root: Path,
    manifest: Path,
    base_url: str,
    max_items_per_chunk: int,
    label_field_name: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        label = text_value(row, group_key, "Unknown")
        grouped.setdefault(label, []).append(row)

    entries: List[Dict[str, Any]] = []
    slug_occurrences: Dict[str, int] = {}
    for label in sorted(grouped.keys(), key=lambda s: s.lower()):
        records = sorted(grouped[label], key=number_sort_key)
        total = len(records)
        parts = list(chunk_iter(records, max_items_per_chunk))
        total_parts = max(1, len(parts))
        base_slug = slugify(label)
        occurrence = slug_occurrences.get(base_slug, 0) + 1
        slug_occurrences[base_slug] = occurrence
        label_slug = base_slug if occurrence == 1 else f"{base_slug}-{occurrence}"
        for index, part_rows in enumerate(parts, start=1):
            suffix = f"-p{index}" if total_parts > 1 else ""
            filename = f"{label_slug}{suffix}.json"
            chunk_path = chunk_root / filename
            write_chunk(chunk_path, part_rows)
            manifest_entry: Dict[str, Any] = {
                "label": f"{label} ({index}/{total_parts})" if total_parts > 1 else label,
                "url": make_manifest_url(base_url, manifest, chunk_path),
                "count": len(part_rows),
                "part": index,
                "parts": total_parts,
            }
            manifest_entry[label_field_name] = label
            entries.append(manifest_entry)
        print(
            f"[Chunk] {label_field_name}={label!r} total={total} "
            f"parts={total_parts} -> {chunk_root.as_posix()}"
        )

    return entries


def main() -> int:
    args = parse_args()

    if args.sets_json is None and args.minifigs_json is None and args.merged_json is None:
        raise SystemExit("Provide --sets-json/--minifigs-json or --merged-json.")

    sets: List[Dict[str, Any]] = []
    minifigs: List[Dict[str, Any]] = []

    if args.merged_json is not None:
        merged_rows = load_json_array(args.merged_json)
        merged_sets, merged_minifigs = split_merged_catalog(merged_rows)
        sets.extend(merged_sets)
        minifigs.extend(merged_minifigs)

    if args.sets_json is not None:
        sets = load_json_array(args.sets_json)

    if args.minifigs_json is not None:
        minifigs = load_json_array(args.minifigs_json)

    if not sets and not minifigs:
        raise SystemExit("No sets or minifigs found in provided input files.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    set_chunk_root = args.output_dir / "sets"
    minifig_chunk_root = args.output_dir / "minifigures"

    set_entries: List[Dict[str, Any]] = []
    minifig_entries: List[Dict[str, Any]] = []

    if sets:
        set_entries = emit_group_chunks(
            rows=sets,
            group_key=args.set_group_key,
            chunk_root=set_chunk_root,
            manifest=args.manifest_path,
            base_url=args.base_url,
            max_items_per_chunk=args.max_items_per_chunk,
            label_field_name="theme",
        )

    if minifigs:
        minifig_entries = emit_group_chunks(
            rows=minifigs,
            group_key=args.minifig_group_key,
            chunk_root=minifig_chunk_root,
            manifest=args.manifest_path,
            base_url=args.base_url,
            max_items_per_chunk=args.max_items_per_chunk,
            label_field_name="category",
        )

    distinct_themes = sorted(
        {
            str(entry.get("theme")).strip()
            for entry in set_entries
            if str(entry.get("theme")).strip()
        },
        key=lambda s: s.lower(),
    )

    manifest: Dict[str, Any] = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sets": set_entries,
        "minifigures": minifig_entries,
        "themes": distinct_themes,
        "summary": {
            "setCount": len(sets),
            "minifigureCount": len(minifigs),
            "setChunkCount": len(set_entries),
            "minifigureChunkCount": len(minifig_entries),
            "maxItemsPerChunk": args.max_items_per_chunk,
        },
    }

    args.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    approx_total_chunks = len(set_entries) + len(minifig_entries)
    print(
        "[Done] "
        f"sets={len(sets)} minifigs={len(minifigs)} chunks={approx_total_chunks} "
        f"manifest={args.manifest_path.as_posix()}"
    )
    if args.base_url:
        print(f"[Done] base_url={args.base_url.rstrip('/')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
