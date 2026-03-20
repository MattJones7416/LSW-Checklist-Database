#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def collapse_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_set_code(number: Any) -> str:
    raw = collapse_ws(number)
    if not raw:
        return ""
    if re.search(r"-[0-9]+$", raw):
        return raw.lower()
    return f"{raw}-1".lower()


def normalize_piece_key(part_num: Any) -> str:
    return collapse_ws(part_num).lower()


def normalize_scraper_item_type(value: Any) -> str:
    raw = collapse_ws(value).lower()
    mapping = {
        "s": "set",
        "set": "set",
        "m": "minifig",
        "minifig": "minifig",
        "minifigure": "minifig",
        "p": "part",
        "part": "part",
        "piece": "part",
    }
    return mapping.get(raw, "")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import normalized market.json files from the standalone scraper output.")
    parser.add_argument("--scraper-dist-dir", required=True, help="Path to the scraper dist directory.")
    parser.add_argument("--catalog-dir", default="dist/catalog", help="Path to the per-item catalog directory.")
    parser.add_argument("--item-type", choices=["all", "set", "minifig", "piece"], default="all")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def month_to_iso(heading: str) -> Optional[str]:
    text = collapse_ws(heading)
    if not text:
        return None
    for fmt in ["%B %Y", "%b %Y"]:
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.strftime("%Y-%m")
        except ValueError:
            continue
    return None


def amount_value(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        raw = value.get("amount")
        if isinstance(raw, (int, float)):
            return float(raw)
        raw = collapse_ws(raw)
        if raw:
            try:
                return float(raw)
            except ValueError:
                return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = collapse_ws(value)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def infer_currency(*values: Any) -> Optional[str]:
    for value in values:
        if isinstance(value, dict):
            currency = collapse_ws(value.get("currency"))
            if currency:
                return currency.upper()
        elif isinstance(value, list):
            inferred = infer_currency(*value)
            if inferred:
                return inferred
    return None


def normalize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "minPrice": amount_value(metrics.get("minPrice")),
        "avgPrice": amount_value(metrics.get("avgPrice")),
        "maxPrice": amount_value(metrics.get("maxPrice")),
        "totalLots": metrics.get("totalLots") if isinstance(metrics.get("totalLots"), int) else None,
        "totalQty": metrics.get("totalQty") if isinstance(metrics.get("totalQty"), int) else None,
        "timesSold": metrics.get("timesSold") if isinstance(metrics.get("timesSold"), int) else None,
    }


def normalize_current_section(section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(section, dict) or not section.get("available"):
        return None
    metrics = section.get("metrics")
    if not isinstance(metrics, dict):
        return None
    normalized = normalize_metrics(metrics)
    if normalized["minPrice"] is None and normalized["avgPrice"] is None and normalized["maxPrice"] is None:
        return None
    return normalized


def normalize_history(blocks: Any) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    if not isinstance(blocks, list):
        return output
    for block in blocks:
        if not isinstance(block, dict):
            continue
        month = month_to_iso(block.get("heading"))
        metrics = block.get("metrics")
        if not month or not isinstance(metrics, dict):
            continue
        normalized = normalize_metrics(metrics)
        avg_price = normalized.get("avgPrice")
        if avg_price is None:
            continue
        entry: Dict[str, Any] = {
            "month": month,
            "monthLabel": collapse_ws(block.get("heading")),
            "avgPrice": avg_price,
        }
        if normalized.get("totalLots") is not None:
            entry["totalLots"] = normalized["totalLots"]
        if normalized.get("totalQty") is not None:
            entry["totalQty"] = normalized["totalQty"]
        if normalized.get("timesSold") is not None:
            entry["timesSold"] = normalized["timesSold"]
        output.append(entry)
    return output


def normalize_transactions(blocks: Any) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    if not isinstance(blocks, list):
        return output
    for block in blocks:
        if not isinstance(block, dict):
            continue
        month = month_to_iso(block.get("heading"))
        month_label = collapse_ws(block.get("heading"))
        entries = block.get("entries")
        if not month or not isinstance(entries, list):
            continue
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                continue
            each = entry.get("each")
            price = amount_value(each)
            if price is None:
                continue
            row: Dict[str, Any] = {
                "month": month,
                "monthLabel": month_label,
                "sequence": index,
                "quantity": entry.get("qty") if isinstance(entry.get("qty"), int) else 1,
                "price": price,
            }
            currency = infer_currency(each)
            if currency:
                row["currencyCode"] = currency
            store_url = collapse_ws(entry.get("storeUrl"))
            if store_url:
                row["storeUrl"] = store_url
            output.append(row)
    return output


def latest_sale_from_history(history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not history:
        return None
    latest = max(history, key=lambda row: (row.get("month") or "", row.get("avgPrice") or 0))
    price = latest.get("avgPrice")
    month = latest.get("month")
    if price is None or not month:
        return None
    return {"month": month, "price": price}


def normalize_set_or_minifig_payload(data: Dict[str, Any], item_type: str) -> Optional[Dict[str, Any]]:
    price_guide = data.get("priceGuide")
    if not isinstance(price_guide, dict):
        return None
    summary = price_guide.get("summary") or {}
    detail = price_guide.get("detail") or {}
    sold_summary = summary.get("sold") if isinstance(summary, dict) else {}
    current_summary = summary.get("current") if isinstance(summary, dict) else {}
    sold_detail = detail.get("sold") if isinstance(detail, dict) else {}
    current_detail = detail.get("current") if isinstance(detail, dict) else {}

    history_new = normalize_history((sold_detail.get("new") or {}).get("blocks"))
    history_used = normalize_history((sold_detail.get("used") or {}).get("blocks"))
    transactions_new = normalize_transactions((sold_detail.get("new") or {}).get("blocks"))
    transactions_used = normalize_transactions((sold_detail.get("used") or {}).get("blocks"))
    current_new = normalize_current_section(current_summary.get("new") if isinstance(current_summary, dict) else None)
    current_used = normalize_current_section(current_summary.get("used") if isinstance(current_summary, dict) else None)

    payload: Dict[str, Any] = {
        "version": 1,
        "itemType": item_type,
        "number": collapse_ws(data.get("itemNo")),
        "updatedAtUTC": collapse_ws(data.get("scrapedAt") or data.get("updatedAt")),
        "sourceURL": collapse_ws((data.get("source") or {}).get("itemUrl")),
        "priceGuideURL": collapse_ws((data.get("source") or {}).get("priceGuideUrl")),
        "currencyCode": infer_currency(
            (current_detail.get("new") or {}).get("blocks", []),
            (current_detail.get("used") or {}).get("blocks", []),
            (sold_detail.get("new") or {}).get("blocks", []),
            (sold_detail.get("used") or {}).get("blocks", []),
        ),
    }
    if current_new or current_used:
        payload["current"] = {}
        if current_new:
            payload["current"]["new"] = {
                key: current_new[key]
                for key in ["minPrice", "avgPrice", "maxPrice"]
                if current_new.get(key) is not None
            }
        if current_used:
            payload["current"]["used"] = {
                key: current_used[key]
                for key in ["minPrice", "avgPrice", "maxPrice"]
                if current_used.get(key) is not None
            }
    if history_new or history_used:
        payload["history"] = {}
        if history_new:
            payload["history"]["new"] = history_new
        if history_used:
            payload["history"]["used"] = history_used
    if transactions_new or transactions_used:
        payload["transactions"] = {}
        if transactions_new:
            payload["transactions"]["new"] = transactions_new
        if transactions_used:
            payload["transactions"]["used"] = transactions_used
    latest_sales: Dict[str, Any] = {}
    latest_new = latest_sale_from_history(history_new)
    latest_used = latest_sale_from_history(history_used)
    if latest_new:
        latest_sales["new"] = latest_new
    if latest_used:
        latest_sales["used"] = latest_used
    if latest_sales:
        payload["latestSales"] = latest_sales
    if not payload.get("current") and not payload.get("history") and not payload.get("transactions") and not payload.get("latestSales"):
        return None
    return payload


def normalize_piece_payload(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    colors = data.get("colors")
    if not isinstance(colors, list) or not colors:
        return None
    weighted_sum = 0.0
    weight_total = 0
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    listing_count = 0
    currency_code: Optional[str] = None
    source_url = collapse_ws((data.get("source") or {}).get("itemUrl"))
    price_guide_url = ""

    for color in colors:
        if not isinstance(color, dict):
            continue
        pg = color.get("priceGuide")
        if not isinstance(pg, dict):
            continue
        current = ((pg.get("summary") or {}).get("current") or {}).get("new")
        if not isinstance(current, dict) or not current.get("available"):
            continue
        metrics = current.get("metrics")
        if not isinstance(metrics, dict):
            continue
        avg_price = amount_value(metrics.get("avgPrice"))
        min_value = amount_value(metrics.get("minPrice"))
        max_value = amount_value(metrics.get("maxPrice"))
        total_lots = metrics.get("totalLots") if isinstance(metrics.get("totalLots"), int) else 0
        total_qty = metrics.get("totalQty") if isinstance(metrics.get("totalQty"), int) else 0
        weight = total_qty or total_lots
        if avg_price is not None and weight > 0:
            weighted_sum += avg_price * weight
            weight_total += weight
        if min_value is not None:
            min_price = min(min_price, min_value) if min_price is not None else min_value
        if max_value is not None:
            max_price = max(max_price, max_value) if max_price is not None else max_value
        listing_count += total_lots
        if currency_code is None:
            currency_code = infer_currency(metrics.get("avgPrice"), metrics.get("minPrice"), metrics.get("maxPrice"))
        if not price_guide_url:
            price_guide_url = collapse_ws((color.get("source") or {}).get("priceGuideUrl"))

    if weight_total <= 0 and min_price is None and max_price is None and listing_count <= 0:
        return None

    payload: Dict[str, Any] = {
        "version": 1,
        "itemType": "piece",
        "partNum": collapse_ws(data.get("itemNo")),
        "brickLinkPartNum": collapse_ws(data.get("itemNo")),
        "updatedAtUTC": collapse_ws(data.get("updatedAt")),
        "sourceURL": source_url,
        "priceGuideURL": price_guide_url,
        "currencyCode": currency_code,
        "averageBuyPrice": (weighted_sum / weight_total) if weight_total > 0 else None,
        "minBuyPrice": min_price,
        "maxBuyPrice": max_price,
        "listingCount": listing_count or None,
    }
    if payload["averageBuyPrice"] is None and payload["minBuyPrice"] is None and payload["maxBuyPrice"] is None:
        return None
    return payload


def iter_scraper_files(root: Path, item_type: str) -> Iterable[Path]:
    if item_type in {"all", "set"}:
        yield from sorted((root / "sets").rglob("*.json"))
    if item_type in {"all", "minifig"}:
        yield from sorted((root / "minifigures").rglob("*.json"))
    if item_type in {"all", "piece"}:
        yield from sorted((root / "parts").rglob("*.json"))


def build_catalog_lookup(catalog_dir: Path, item_type: str) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    base_map = {
        "set": catalog_dir / "sets",
        "minifig": catalog_dir / "minifigs",
        "piece": catalog_dir / "pieces",
    }
    for kind, base in base_map.items():
        if item_type not in {"all", kind} or not base.exists():
            continue
        for item_json in base.rglob("item.json"):
            try:
                data = load_json(item_json)
            except Exception:
                continue
            if kind == "set":
                key = normalize_set_code(data.get("Number") or data.get("number"))
            elif kind == "minifig":
                key = collapse_ws(data.get("Number") or data.get("number")).lower()
            else:
                key = normalize_piece_key(data.get("part_num") or data.get("bricklink_part_num"))
            if key:
                lookup[key] = item_json.parent
    return lookup


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    scraper_root = Path(args.scraper_dist_dir)
    catalog_dir = Path(args.catalog_dir)
    lookup = build_catalog_lookup(catalog_dir, args.item_type)
    written = 0
    skipped = 0

    for path in iter_scraper_files(scraper_root, args.item_type):
        try:
            data = load_json(path)
        except Exception:
            skipped += 1
            continue
        item_type = normalize_scraper_item_type(data.get("itemType") or data.get("itemTypeLabel"))
        if item_type == "set":
            key = normalize_set_code(data.get("itemNo"))
            payload = normalize_set_or_minifig_payload(data, "set")
        elif item_type == "minifig":
            key = collapse_ws(data.get("itemNo")).lower()
            payload = normalize_set_or_minifig_payload(data, "minifig")
        elif item_type == "part":
            key = normalize_piece_key(data.get("itemNo"))
            payload = normalize_piece_payload(data)
        else:
            skipped += 1
            continue

        if not key or payload is None:
            skipped += 1
            continue
        item_dir = lookup.get(key)
        if item_dir is None:
            skipped += 1
            continue
        market_path = item_dir / "market.json"
        market_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written += 1
        if args.verbose:
            print(f"[ImportMarket] wrote {market_path}", flush=True)

    print(f"[ImportMarket] written={written} skipped={skipped}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
