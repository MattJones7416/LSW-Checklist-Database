#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


LEGACY_MARKET_FIELDS = {
    "New",
    "Used",
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
    "CurrentNewVsRRPAmount",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strip legacy market fields from release monolith JSON files.")
    parser.add_argument("--sets-json", default="dist/Lego Star Wars Database.json")
    parser.add_argument("--minifigs-json", default="dist/Lego-Star-Wars-Minifigure-Database.json")
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [row for row in data if isinstance(row, dict)]


def compact_rows(rows: List[Dict[str, Any]]) -> int:
    removed = 0
    for row in rows:
        for field in LEGACY_MARKET_FIELDS:
            if field in row:
                row.pop(field, None)
                removed += 1
    return removed


def maybe_write(path: Path, rows: List[Dict[str, Any]]) -> bool:
    updated = json.dumps(rows, ensure_ascii=False, separators=(",", ":")) + "\n"
    original = path.read_text(encoding="utf-8")
    if original == updated:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def process(path: Path) -> None:
    if not path.exists():
        print(f"[Compact] skip missing {path}")
        return
    before = path.stat().st_size
    rows = load_rows(path)
    removed = compact_rows(rows)
    changed = maybe_write(path, rows)
    after = path.stat().st_size
    print(
        f"[Compact] {path} removed_fields={removed} changed={changed} bytes_before={before} bytes_after={after}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    process(Path(args.sets_json))
    process(Path(args.minifigs_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
