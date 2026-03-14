#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

PROVIDER_ORDER = [
    "bricklink",
    "brickowl",
    "amazon",
    "lego",
    "johnlewis",
    "very",
    "vinted",
    "ebay",
]
REGIONS = ["UK", "US", "EU"]
OUTPUT_FILENAME_BY_REGION = {"UK": "uk.json", "US": "us.json", "EU": "eu.json"}


def write_json(path: Path, data: Any, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        if pretty
        else json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    path.write_text(payload, encoding="utf-8")


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def dedupe_and_sort(deals: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for deal in deals:
        deal_id = str(deal.get("id") or "").strip().lower()
        if not deal_id:
            continue
        merged[deal_id] = deal
    return sorted(
        merged.values(),
        key=lambda row: (
            str(row.get("number") or "").strip().casefold(),
            float(row.get("priceValue") or 0.0),
            str(row.get("source") or "").strip().casefold(),
            str(row.get("title") or "").strip().casefold(),
        ),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge provider-scoped marketplace deal artifacts into app-facing JSON outputs.")
    parser.add_argument("--provider-root", default="dist/provider-deals")
    parser.add_argument("--output-dir", default="dist/deals")
    parser.add_argument("--fallback-output", default="dist/marketplace-deals.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    provider_root = Path(args.provider_root)
    output_dir = Path(args.output_dir)
    fallback_output = Path(args.fallback_output)

    merged_by_region: Dict[str, List[Dict[str, Any]]] = {region: [] for region in REGIONS}
    provider_counts: Dict[str, Dict[str, int]] = {}

    for provider in PROVIDER_ORDER:
        provider_dir = provider_root / provider
        if not provider_dir.exists():
            continue
        counts: Dict[str, int] = {}
        for region in REGIONS:
            deals = load_json_array(provider_dir / OUTPUT_FILENAME_BY_REGION[region])
            if not deals:
                continue
            merged_by_region[region].extend(deals)
            counts[region] = len(deals)
        metadata_path = provider_dir / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
            if isinstance(metadata, dict):
                counts.update({f"meta:{k}": v for k, v in metadata.items() if k in {"generatedAt", "searchErrors", "accessDenied"}})
        if counts:
            provider_counts[provider] = counts

    all_deals: List[Dict[str, Any]] = []
    for region in REGIONS:
        normalized = dedupe_and_sort(merged_by_region[region])
        write_json(output_dir / OUTPUT_FILENAME_BY_REGION[region], normalized)
        all_deals.extend(normalized)
    write_json(output_dir / "all.json", dedupe_and_sort(all_deals))
    write_json(
        fallback_output,
        {
            "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "regions": {region: dedupe_and_sort(merged_by_region[region]) for region in REGIONS if merged_by_region[region]},
            "providers": provider_counts,
        },
        pretty=True,
    )

    for region in REGIONS:
        print(f"[merge:{region}] deals={len(dedupe_and_sort(merged_by_region[region]))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
