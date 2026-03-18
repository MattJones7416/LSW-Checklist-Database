#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import update_market_values as market


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def normalize_key(value: Any) -> str:
    return market.collapse_ws(value).lower()


def parse_float(value: Any) -> Optional[float]:
    return market._parse_float(value)


def parse_int(value: Any) -> Optional[int]:
    return market._parse_int(value)


def price_value(row: Dict[str, Any]) -> Optional[float]:
    return market.first_non_none([
        parse_float(row.get("qty_avg_price")),
        parse_float(row.get("avg_price")),
        parse_float(row.get("min_price")),
    ])


def row_weight(row: Dict[str, Any]) -> int:
    return (
        parse_int(row.get("total_quantity"))
        or parse_int(row.get("unit_quantity"))
        or 1
    )


def aggregate_listing_summary(
    stock_new: Optional[Dict[str, Any]],
    stock_used: Optional[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[int]]:
    rows = [row for row in [stock_new, stock_used] if isinstance(row, dict)]
    if not rows:
        return (None, None, None, None)

    weighted_total = 0.0
    total_weight = 0
    fallback_values: List[float] = []
    listing_count = 0
    min_price: Optional[float] = None
    max_price: Optional[float] = None

    for row in rows:
        value = price_value(row)
        if value is not None and value > 0:
            weight = row_weight(row)
            weighted_total += value * weight
            total_weight += weight
            fallback_values.append(value)

        lots = parse_int(row.get("unit_quantity"))
        if lots and lots > 0:
            listing_count += lots

        row_min = parse_float(row.get("min_price"))
        if row_min is not None and row_min > 0:
            min_price = row_min if min_price is None else min(min_price, row_min)

        row_max = parse_float(row.get("max_price"))
        if row_max is not None and row_max > 0:
            max_price = row_max if max_price is None else max(max_price, row_max)

    average = None
    if total_weight > 0:
        average = round(weighted_total / total_weight, 4)
    elif fallback_values:
        average = round(sum(fallback_values) / len(fallback_values), 4)

    return (
        average,
        round(min_price, 4) if min_price is not None else None,
        round(max_price, 4) if max_price is not None else None,
        listing_count if listing_count > 0 else None,
    )


def extract_record(
    row: Dict[str, Any],
    fetcher: market.HTMLPriceGuideFetcher,
    throttle: market.RuntimeThrottle,
    now_iso: str,
    quiet_no_data: bool,
) -> Tuple[str, Optional[Dict[str, Any]], str]:
    part_num = market.collapse_ws(row.get("part_num"))
    bricklink_part_num = market.collapse_ws(row.get("bricklink_part_num"))
    candidates = market.build_part_item_candidates(
        row.get("part_num"),
        row.get("Link"),
        row.get("BrickLinkPriceGuideURL"),
        row.get("bricklink_part_num"),
    )
    if not candidates:
        return (part_num, None, "no_candidate")

    for candidate in candidates:
        fetched = fetcher.fetch_price_guide_html("PART", candidate, throttle, quiet_no_data=quiet_no_data)
        if not fetched:
            continue
        matrix, _month_new, _month_used, _tx_new, _tx_used, _listings_new, _listings_used, currency = fetched
        stock_new = matrix.get(("stock", "N"))
        stock_used = matrix.get(("stock", "U"))
        average, min_price, max_price, listing_count = aggregate_listing_summary(stock_new, stock_used)
        if average is None and min_price is None and max_price is None and listing_count is None:
            continue
        resolved_part_num = part_num or candidate
        resolved_bricklink_part_num = bricklink_part_num or candidate
        record = {
            "part_num": resolved_part_num,
            "bricklink_part_num": resolved_bricklink_part_num or None,
            "average_listing_price": average,
            "min_listing_price": min_price,
            "max_listing_price": max_price,
            "listing_count": listing_count,
            "currency_code": (currency or "GBP").strip().upper() or "GBP",
            "price_guide_url": market.build_html_price_guide_url("PART", candidate),
            "last_updated_utc": now_iso,
        }
        return (resolved_part_num, record, "updated")

    return (part_num, None, fetcher.last_error_kind or "no_data")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch BrickLink HTML live piece pricing into a dedicated artifact.")
    parser.add_argument("--parts-json", default="dist/parts/parts-catalog.json")
    parser.add_argument("--output", default="dist/parts/piece-live-pricing.json")
    parser.add_argument("--state-path", default="dist/parts/piece-live-pricing-state.json")
    parser.add_argument("--limit", type=int, default=240)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--html-delay-seconds", type=float, default=0.35)
    parser.add_argument("--html-jitter-seconds", type=float, default=0.15)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--only-part", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    parts_path = Path(args.parts_json)
    output_path = Path(args.output)
    state_path = Path(args.state_path)

    rows = load_json(parts_path, [])
    if not isinstance(rows, list):
        raise SystemExit(f"Parts catalog is not a JSON list: {parts_path}")

    existing_rows = load_json(output_path, [])
    existing_lookup: Dict[str, Dict[str, Any]] = {}
    if isinstance(existing_rows, list):
        for entry in existing_rows:
            if not isinstance(entry, dict):
                continue
            key = normalize_key(entry.get("part_num"))
            if key:
                existing_lookup[key] = entry

    state = load_json(state_path, {})
    cursor = int(state.get("cursor", 0) or 0)

    unique_rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = normalize_key(row.get("part_num"))
        if not key or key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)

    if args.only_part.strip():
        target = normalize_key(args.only_part)
        unique_rows = [row for row in unique_rows if normalize_key(row.get("part_num")) == target]
        cursor = 0

    total = len(unique_rows)
    if total == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("[]\n", encoding="utf-8")
        state_path.write_text(json.dumps({"cursor": 0, "total": 0}, indent=2) + "\n", encoding="utf-8")
        print("[PiecePricing] no rows to process", flush=True)
        return 0

    start = max(0, min(cursor, total))
    limit = max(1, args.limit)
    end = min(total, start + limit)
    batch = unique_rows[start:end]

    throttle = market.RuntimeThrottle(args.html_delay_seconds, args.html_jitter_seconds)
    budget = market.ApiRequestBudget(max_calls=None)
    fetcher = market.HTMLPriceGuideFetcher(
        timeout=args.timeout_seconds,
        retries=args.retries,
        verbose=args.verbose,
        request_budget=budget,
    )
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    print(f"[PiecePricing] processing {start}:{end} of {total} with workers={max(1, args.workers)}", flush=True)

    results: List[Tuple[str, Optional[Dict[str, Any]], str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(extract_record, row, fetcher, throttle, now_iso, not args.verbose)
            for row in batch
        ]
        for offset, future in enumerate(futures, start=1):
            key, record, status = future.result()
            results.append((key, record, status))
            label = key or "?"
            print(f"[PiecePricing] {start + offset}/{total}: {label} {status}", flush=True)

    changed = 0
    updated = 0
    missed = 0
    for key, record, status in results:
        normalized = normalize_key(key)
        if not normalized:
            continue
        if record is None:
            missed += 1
            continue
        previous = existing_lookup.get(normalized)
        if previous != record:
            existing_lookup[normalized] = record
            changed += 1
        updated += 1

    ordered = sorted(existing_lookup.values(), key=lambda entry: normalize_key(entry.get("part_num")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    next_cursor = 0 if end >= total else end
    touched_part_numbers = []
    seen_part_numbers = set()
    for key, _record, _status in results:
        normalized = normalize_key(key)
        if not normalized or normalized in seen_part_numbers:
            continue
        seen_part_numbers.add(normalized)
        touched_part_numbers.append(normalized)

    state_payload = {
        "cursor": next_cursor,
        "total": total,
        "last_batch_start": start,
        "last_batch_end": end,
        "updated": updated,
        "changed": changed,
        "missed": missed,
        "last_updated_utc": now_iso,
        "lastUpdatedPartNumbers": touched_part_numbers,
    }
    state_path.write_text(json.dumps(state_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"[PiecePricing] done updated={updated} changed={changed} missed={missed} next_cursor={next_cursor}/{total}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
