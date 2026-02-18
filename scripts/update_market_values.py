#!/usr/bin/env python3
"""Update LEGO market values in catalog JSON files using BrickLink API only.

This replaces HTML scraping with authenticated BrickLink API requests.

Per item, the script attempts a single call to:
  GET /items/{type}/{no}/price?currency_code={code}
and only falls back to targeted guide_type/new_or_used calls when needed.

Updated fields include:
- New / Used display values
- BrickLink sold/current summary stats
- BrickLink monthly series (snapshot-per-run, API-only)
- Derived latest sale, RRP delta, basic 2Y/5Y forecast
- Cross-catalog exclusivity and appears-in mappings
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote

import requests
from requests_oauthlib import OAuth1


BRICKLINK_API_BASE_URL = "https://api.bricklink.com/api/store/v1"

MONTH_LABELS = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


@dataclass
class FetchConfig:
    timeout: float
    retries: int
    delay: float
    jitter: float
    verbose: bool
    currency_code: str


@dataclass
class FileUpdateStats:
    total_rows: int = 0
    rows_considered: int = 0
    rows_changed: int = 0
    fetch_failures: int = 0
    parse_misses: int = 0
    cross_rows_changed: int = 0
    last_index_processed: Optional[int] = None


@dataclass
class ApiRequestBudget:
    max_calls: Optional[int]
    used_calls: int = 0
    exhausted: bool = False

    def consume(self) -> bool:
        if self.max_calls is not None and self.used_calls >= self.max_calls:
            self.exhausted = True
            return False
        self.used_calls += 1
        return True


class RuntimeThrottle:
    def __init__(self, min_delay: float, jitter: float) -> None:
        self.min_delay = max(0.0, min_delay)
        self.jitter = max(0.0, jitter)
        self.current_delay = self.min_delay
        self.max_delay = 6.0

    def sleep_between_requests(self) -> None:
        delay = self.current_delay
        if self.jitter > 0:
            delay += random.uniform(0.0, self.jitter)
        if delay > 0:
            time.sleep(delay)

    def apply_success(self) -> None:
        self.current_delay = max(self.min_delay, self.current_delay * 0.96)

    def apply_backoff(self, retry_after: Optional[float] = None) -> None:
        candidate = max(self.current_delay * 1.6, self.min_delay * 1.5)
        if retry_after is not None:
            candidate = max(candidate, retry_after)
        self.current_delay = min(self.max_delay, candidate)


class BrickLinkClient:
    def __init__(
        self,
        consumer_key: str,
        consumer_secret: str,
        token_value: str,
        token_secret: str,
        timeout: float,
        retries: int,
        verbose: bool,
        request_budget: Optional[ApiRequestBudget] = None,
        base_url: str = BRICKLINK_API_BASE_URL,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = max(1.0, timeout)
        self.retries = max(0, retries)
        self.verbose = verbose
        self.session = requests.Session()
        self.oauth = OAuth1(
            consumer_key,
            consumer_secret,
            token_value,
            token_secret,
            signature_method="HMAC-SHA1",
            signature_type="AUTH_HEADER",
        )
        self.auth_failed = False
        self.auth_error_message = ""
        self.request_budget = request_budget or ApiRequestBudget(max_calls=None)

    @property
    def budget_exhausted(self) -> bool:
        return bool(self.request_budget.exhausted)

    def fetch_price_matrix(
        self,
        item_type: str,
        item_no: str,
        currency_code: str,
        throttle: RuntimeThrottle,
    ) -> Optional[Dict[Tuple[str, str], Dict[str, Any]]]:
        # Fast path: one request, attempt to receive all guide_type/new_or_used combinations.
        matrix = self._fetch_matrix_once(
            item_type=item_type,
            item_no=item_no,
            params={"currency_code": currency_code},
            throttle=throttle,
        )
        if matrix is None:
            return None

        needed: List[Tuple[str, str]] = [
            ("sold", "N"),
            ("sold", "U"),
            ("stock", "N"),
            ("stock", "U"),
        ]

        missing = [pair for pair in needed if pair not in matrix]
        for guide_type, condition in missing:
            sub = self._fetch_matrix_once(
                item_type=item_type,
                item_no=item_no,
                params={
                    "currency_code": currency_code,
                    "guide_type": guide_type,
                    "new_or_used": condition,
                },
                throttle=throttle,
            )
            if not sub:
                continue
            matrix.update(sub)

        return matrix

    def _fetch_matrix_once(
        self,
        item_type: str,
        item_no: str,
        params: Dict[str, str],
        throttle: RuntimeThrottle,
    ) -> Optional[Dict[Tuple[str, str], Dict[str, Any]]]:
        safe_item_type = quote(item_type.upper(), safe="")
        safe_item_no = quote(item_no, safe="")
        url = f"{self.base_url}/items/{safe_item_type}/{safe_item_no}/price"

        attempt = 0
        while True:
            attempt += 1
            if not self.request_budget.consume():
                if self.verbose:
                    print(
                        f"[API] request budget exhausted before {item_type}:{item_no}",
                        flush=True,
                    )
                return None
            throttle.sleep_between_requests()
            try:
                response = self.session.get(
                    url,
                    params=params,
                    auth=self.oauth,
                    timeout=self.timeout,
                    headers={"Accept": "application/json"},
                )
            except requests.RequestException as exc:
                if attempt > self.retries + 1:
                    if self.verbose:
                        print(f"[API] request failed {item_type}:{item_no}: {exc}", flush=True)
                    return None
                throttle.apply_backoff()
                continue

            if response.status_code == 429:
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                throttle.apply_backoff(retry_after)
                if attempt > self.retries + 1:
                    if self.verbose:
                        print(f"[API] HTTP 429 {item_type}:{item_no}", flush=True)
                    return None
                continue

            if response.status_code >= 500:
                throttle.apply_backoff()
                if attempt > self.retries + 1:
                    if self.verbose:
                        print(
                            f"[API] HTTP {response.status_code} {item_type}:{item_no}",
                            flush=True,
                        )
                    return None
                continue

            if response.status_code >= 400:
                if response.status_code == 401:
                    self.auth_failed = True
                    self.auth_error_message = (
                        "BrickLink API authentication failed (HTTP 401). "
                        "Check BRICKLINK_CONSUMER_KEY/SECRET and BRICKLINK_TOKEN_VALUE/SECRET."
                    )
                if self.verbose:
                    print(
                        f"[API] HTTP {response.status_code} {item_type}:{item_no} params={params}",
                        flush=True,
                    )
                return None

            try:
                payload = response.json()
            except ValueError:
                if self.verbose:
                    print(f"[API] invalid JSON {item_type}:{item_no}", flush=True)
                return None

            meta = payload.get("meta") if isinstance(payload, dict) else None
            if isinstance(meta, dict):
                code = _parse_int(meta.get("code"))
                if code is not None and code >= 400:
                    if code == 401:
                        self.auth_failed = True
                        message = collapse_ws(meta.get("message"))
                        if message:
                            self.auth_error_message = (
                                f"BrickLink API authentication failed ({message}). "
                                "Check BRICKLINK_CONSUMER_KEY/SECRET and BRICKLINK_TOKEN_VALUE/SECRET."
                            )
                        else:
                            self.auth_error_message = (
                                "BrickLink API authentication failed (meta code 401). "
                                "Check BRICKLINK_CONSUMER_KEY/SECRET and BRICKLINK_TOKEN_VALUE/SECRET."
                            )
                    if self.verbose:
                        print(
                            f"[API] meta code={code} {item_type}:{item_no} msg={meta.get('message')}",
                            flush=True,
                        )
                    return None

            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, list):
                if self.verbose:
                    print(f"[API] missing data array {item_type}:{item_no}", flush=True)
                return None

            matrix: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for row in data:
                if not isinstance(row, dict):
                    continue
                guide_type = _normalize_guide_type(row.get("guide_type"))
                condition = _normalize_condition(row.get("new_or_used"))
                if not guide_type or not condition:
                    continue
                matrix[(guide_type, condition)] = row

            throttle.apply_success()
            return matrix


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return parsed


def _normalize_guide_type(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"stock", "sold"}:
        return text
    return None


def _normalize_condition(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"N", "U"}:
        return text
    return None


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"-?[0-9][0-9,]*", text)
    if not match:
        return None
    try:
        return int(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"-?[0-9][0-9,]*(?:\.[0-9]+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def collapse_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def currency_symbol(code: Optional[str]) -> str:
    if code == "GBP":
        return "£"
    if code == "USD":
        return "$"
    if code == "EUR":
        return "€"
    return "£"


def format_display_price(amount: Optional[float], currency_code: Optional[str]) -> Optional[str]:
    if amount is None:
        return None
    cc = (currency_code or "GBP").upper()
    return f"~{currency_symbol(cc)}{amount:.2f}"


def normalize_set_code(number: Any, variant: Any) -> str:
    raw = collapse_ws(number)
    if not raw:
        return ""
    if re.search(r"-[0-9]+$", raw):
        return raw
    var = _parse_int(variant) or 1
    return f"{raw}-{var}"


def parse_minifig_numbers(raw: Any) -> List[str]:
    if raw is None:
        return []
    tokens = [
        token.strip().lower()
        for token in re.split(r"[,;\n\r]+", str(raw))
        if token.strip()
    ]
    seen: set[str] = set()
    ordered: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def month_label(month_key: str) -> str:
    try:
        year_s, month_s = month_key.split("-")
        month_no = int(month_s)
        year_no = int(year_s)
    except Exception:
        return month_key
    return f"{MONTH_LABELS.get(month_no, month_key)} {year_no}"


def upsert_monthly_point(
    series: Any,
    *,
    month: str,
    avg_price: Optional[float],
    total_lots: Optional[int],
    total_qty: Optional[int],
    cap: int = 84,
) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    if isinstance(series, list):
        for row in series:
            if isinstance(row, dict):
                m = collapse_ws(row.get("month"))
                if not m:
                    continue
                points.append(
                    {
                        "month": m,
                        "monthLabel": collapse_ws(row.get("monthLabel")) or month_label(m),
                        "avgPrice": _parse_float(row.get("avgPrice")),
                        "totalLots": _parse_int(row.get("totalLots")),
                        "totalQty": _parse_int(row.get("totalQty")),
                    }
                )

    updated = False
    for row in points:
        if row["month"] == month:
            row["monthLabel"] = month_label(month)
            row["avgPrice"] = round(avg_price, 2) if avg_price is not None else None
            row["totalLots"] = total_lots
            row["totalQty"] = total_qty
            updated = True
            break

    if not updated:
        points.append(
            {
                "month": month,
                "monthLabel": month_label(month),
                "avgPrice": round(avg_price, 2) if avg_price is not None else None,
                "totalLots": total_lots,
                "totalQty": total_qty,
            }
        )

    points.sort(key=lambda row: row.get("month") or "")
    if cap > 0 and len(points) > cap:
        points = points[-cap:]
    return points


def monthly_series_to_transactions(series: List[Dict[str, Any]], currency_code: Optional[str]) -> List[Dict[str, Any]]:
    cc = (currency_code or "GBP").upper()
    tx: List[Dict[str, Any]] = []
    for point in series:
        avg = _parse_float(point.get("avgPrice"))
        if avg is None:
            continue
        qty = max(1, _parse_int(point.get("totalQty")) or 1)
        tx.append(
            {
                "month": point.get("month"),
                "monthLabel": point.get("monthLabel") or month_label(collapse_ws(point.get("month"))),
                "sequence": 1,
                "qty": qty,
                "eachPrice": round(avg, 2),
                "currency": cc,
            }
        )
    return tx


def first_non_none(values: Iterable[Optional[float]]) -> Optional[float]:
    for value in values:
        if value is not None:
            return value
    return None


def compute_forecast_from_series(series: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    samples: List[Tuple[datetime, float]] = []
    for row in series:
        m = collapse_ws(row.get("month"))
        avg = _parse_float(row.get("avgPrice"))
        if not m or avg is None or avg <= 0:
            continue
        try:
            dt = datetime.strptime(m, "%Y-%m").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        samples.append((dt, avg))

    if len(samples) < 2:
        return (None, None, None)

    samples.sort(key=lambda pair: pair[0])
    start_dt, start_val = samples[0]
    end_dt, end_val = samples[-1]
    if start_val <= 0 or end_val <= 0:
        return (None, None, None)

    month_span = max(1, (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month))
    years = month_span / 12.0

    growth = (end_val / start_val) ** (1.0 / years) - 1.0
    growth = max(-0.85, min(2.0, growth))

    f2 = end_val * ((1.0 + growth) ** 2)
    f5 = end_val * ((1.0 + growth) ** 5)
    return (round(f2, 2), round(f5, 2), round(growth * 100.0, 2))


def to_price_guide_url(item_type: str, item_no: str) -> str:
    if item_type == "SET":
        return f"https://www.bricklink.com/catalogPG.asp?S={item_no}&ColorID=0"
    return f"https://www.bricklink.com/catalogPG.asp?M={item_no}&ColorID=0"


def apply_market_to_row(
    row: Dict[str, Any],
    *,
    item_type: str,
    item_no: str,
    currency_code: str,
    client: BrickLinkClient,
    throttle: RuntimeThrottle,
    month_key: str,
) -> bool:
    matrix = client.fetch_price_matrix(item_type=item_type, item_no=item_no, currency_code=currency_code, throttle=throttle)
    if matrix is None:
        return False

    sold_new = matrix.get(("sold", "N"))
    sold_used = matrix.get(("sold", "U"))
    stock_new = matrix.get(("stock", "N"))
    stock_used = matrix.get(("stock", "U"))

    sold_new_avg = _parse_float(sold_new.get("avg_price") if sold_new else None)
    sold_used_avg = _parse_float(sold_used.get("avg_price") if sold_used else None)
    stock_new_avg = _parse_float(stock_new.get("avg_price") if stock_new else None)
    stock_used_avg = _parse_float(stock_used.get("avg_price") if stock_used else None)

    currency = collapse_ws(
        (sold_new or sold_used or stock_new or stock_used or {}).get("currency_code")
    ).upper() or currency_code

    # Summary prices and canonical display values.
    row["BrickLinkPriceGuideURL"] = to_price_guide_url(item_type, item_no)
    row["BrickLinkPriceGuideCurrency"] = currency

    row["BrickLinkSoldPriceNew"] = round(sold_new_avg, 2) if sold_new_avg is not None else None
    row["BrickLinkSoldPriceUsed"] = round(sold_used_avg, 2) if sold_used_avg is not None else None

    row["BrickLink6MSoldNewTimesSold"] = _parse_int(sold_new.get("unit_quantity") if sold_new else None)
    row["BrickLink6MSoldNewTotalQty"] = _parse_int(sold_new.get("total_quantity") if sold_new else None)
    row["BrickLink6MSoldNewMinPrice"] = round(_parse_float(sold_new.get("min_price") if sold_new else None), 2) if _parse_float(sold_new.get("min_price") if sold_new else None) is not None else None
    row["BrickLink6MSoldNewAvgPrice"] = round(sold_new_avg, 2) if sold_new_avg is not None else None
    row["BrickLink6MSoldNewQtyAvgPrice"] = round(_parse_float(sold_new.get("qty_avg_price") if sold_new else None), 2) if _parse_float(sold_new.get("qty_avg_price") if sold_new else None) is not None else None
    row["BrickLink6MSoldNewMaxPrice"] = round(_parse_float(sold_new.get("max_price") if sold_new else None), 2) if _parse_float(sold_new.get("max_price") if sold_new else None) is not None else None

    row["BrickLink6MSoldUsedTimesSold"] = _parse_int(sold_used.get("unit_quantity") if sold_used else None)
    row["BrickLink6MSoldUsedTotalQty"] = _parse_int(sold_used.get("total_quantity") if sold_used else None)
    row["BrickLink6MSoldUsedMinPrice"] = round(_parse_float(sold_used.get("min_price") if sold_used else None), 2) if _parse_float(sold_used.get("min_price") if sold_used else None) is not None else None
    row["BrickLink6MSoldUsedAvgPrice"] = round(sold_used_avg, 2) if sold_used_avg is not None else None
    row["BrickLink6MSoldUsedQtyAvgPrice"] = round(_parse_float(sold_used.get("qty_avg_price") if sold_used else None), 2) if _parse_float(sold_used.get("qty_avg_price") if sold_used else None) is not None else None
    row["BrickLink6MSoldUsedMaxPrice"] = round(_parse_float(sold_used.get("max_price") if sold_used else None), 2) if _parse_float(sold_used.get("max_price") if sold_used else None) is not None else None

    row["BrickLinkCurrentNewTotalLots"] = _parse_int(stock_new.get("unit_quantity") if stock_new else None)
    row["BrickLinkCurrentNewTotalQty"] = _parse_int(stock_new.get("total_quantity") if stock_new else None)
    row["BrickLinkCurrentNewMinPrice"] = round(_parse_float(stock_new.get("min_price") if stock_new else None), 2) if _parse_float(stock_new.get("min_price") if stock_new else None) is not None else None
    row["BrickLinkCurrentNewAvgPrice"] = round(stock_new_avg, 2) if stock_new_avg is not None else None
    row["BrickLinkCurrentNewQtyAvgPrice"] = round(_parse_float(stock_new.get("qty_avg_price") if stock_new else None), 2) if _parse_float(stock_new.get("qty_avg_price") if stock_new else None) is not None else None
    row["BrickLinkCurrentNewMaxPrice"] = round(_parse_float(stock_new.get("max_price") if stock_new else None), 2) if _parse_float(stock_new.get("max_price") if stock_new else None) is not None else None

    row["BrickLinkCurrentUsedTotalLots"] = _parse_int(stock_used.get("unit_quantity") if stock_used else None)
    row["BrickLinkCurrentUsedTotalQty"] = _parse_int(stock_used.get("total_quantity") if stock_used else None)
    row["BrickLinkCurrentUsedMinPrice"] = round(_parse_float(stock_used.get("min_price") if stock_used else None), 2) if _parse_float(stock_used.get("min_price") if stock_used else None) is not None else None
    row["BrickLinkCurrentUsedAvgPrice"] = round(stock_used_avg, 2) if stock_used_avg is not None else None
    row["BrickLinkCurrentUsedQtyAvgPrice"] = round(_parse_float(stock_used.get("qty_avg_price") if stock_used else None), 2) if _parse_float(stock_used.get("qty_avg_price") if stock_used else None) is not None else None
    row["BrickLinkCurrentUsedMaxPrice"] = round(_parse_float(stock_used.get("max_price") if stock_used else None), 2) if _parse_float(stock_used.get("max_price") if stock_used else None) is not None else None

    display_new = first_non_none([sold_new_avg, stock_new_avg])
    display_used = first_non_none([sold_used_avg, stock_used_avg])
    row["New"] = format_display_price(display_new, currency)
    row["Used"] = format_display_price(display_used, currency)

    # Keep latest-sale fields aligned to the most recent API snapshot (month-level granularity).
    row["BrickLinkLatestSaleNewMonth"] = month_key if sold_new_avg is not None else None
    row["BrickLinkLatestSaleNewPrice"] = round(sold_new_avg, 2) if sold_new_avg is not None else None
    row["BrickLinkLatestSaleUsedMonth"] = month_key if sold_used_avg is not None else None
    row["BrickLinkLatestSaleUsedPrice"] = round(sold_used_avg, 2) if sold_used_avg is not None else None

    used_min_candidates = [
        _parse_float(sold_used.get("min_price") if sold_used else None),
        _parse_float(stock_used.get("min_price") if stock_used else None),
    ]
    used_max_candidates = [
        _parse_float(sold_used.get("max_price") if sold_used else None),
        _parse_float(stock_used.get("max_price") if stock_used else None),
    ]
    used_min = min((v for v in used_min_candidates if v is not None), default=None)
    used_max = max((v for v in used_max_candidates if v is not None), default=None)
    row["BrickLinkUsedPriceRangeMin"] = round(used_min, 2) if used_min is not None else None
    row["BrickLinkUsedPriceRangeMax"] = round(used_max, 2) if used_max is not None else None

    # Monthly series = one API snapshot per run (stable, API-only, no scraping).
    month_new = upsert_monthly_point(
        row.get("BrickLinkMonthlySalesNew"),
        month=month_key,
        avg_price=sold_new_avg,
        total_lots=_parse_int(sold_new.get("unit_quantity") if sold_new else None),
        total_qty=_parse_int(sold_new.get("total_quantity") if sold_new else None),
    )
    month_used = upsert_monthly_point(
        row.get("BrickLinkMonthlySalesUsed"),
        month=month_key,
        avg_price=sold_used_avg,
        total_lots=_parse_int(sold_used.get("unit_quantity") if sold_used else None),
        total_qty=_parse_int(sold_used.get("total_quantity") if sold_used else None),
    )
    row["BrickLinkMonthlySalesNew"] = month_new
    row["BrickLinkMonthlySalesUsed"] = month_used

    tx_new = monthly_series_to_transactions(month_new, currency)
    tx_used = monthly_series_to_transactions(month_used, currency)
    row["BrickLinkTransactionsNew"] = tx_new
    row["BrickLinkTransactionsUsed"] = tx_used
    row["BrickLinkTransactionsNewCount"] = len(tx_new)
    row["BrickLinkTransactionsUsedCount"] = len(tx_used)

    # RRP and forecast helpers.
    rrp = _parse_float(row.get("UKRetailPrice"))
    current_new_for_compare = first_non_none([stock_new_avg, sold_new_avg])
    if currency == "GBP" and rrp is not None and rrp > 0 and current_new_for_compare is not None:
        delta = current_new_for_compare - rrp
        row["CurrentNewVsRRPAmount"] = round(delta, 2)
        row["CurrentNewVsRRPPercent"] = round((delta / rrp) * 100.0, 2)
        row["CurrentRRPBaseline"] = round(rrp, 2)
    else:
        row["CurrentNewVsRRPAmount"] = None
        row["CurrentNewVsRRPPercent"] = None
        row["CurrentRRPBaseline"] = round(rrp, 2) if rrp is not None else None

    f2n, f5n, gn = compute_forecast_from_series(month_new)
    f2u, f5u, gu = compute_forecast_from_series(month_used)
    row["PriceForecast2YNew"] = f2n
    row["PriceForecast5YNew"] = f5n
    row["PriceForecast2YUsed"] = f2u
    row["PriceForecast5YUsed"] = f5u
    row["PriceTrendAnnualizedNewPercent"] = gn
    row["PriceTrendAnnualizedUsedPercent"] = gu
    row["PriceForecastMethod"] = "bricklink_api_monthly_trend"
    row["MarketLastUpdatedUTC"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return True


def apply_cross_catalog_enrichment(
    set_rows: List[Dict[str, Any]],
    minifig_rows: List[Dict[str, Any]],
) -> Tuple[int, int]:
    minifig_to_sets: Dict[str, set[str]] = {}

    for row in set_rows:
        set_code = normalize_set_code(row.get("Number"), row.get("Variant"))
        if not set_code:
            continue
        for minifig_number in parse_minifig_numbers(row.get("MinifigNumbers")):
            minifig_to_sets.setdefault(minifig_number, set()).add(set_code)

    set_changed_count = 0
    for row in set_rows:
        unique_minifigs = parse_minifig_numbers(row.get("MinifigNumbers"))
        exclusive = sorted([code for code in unique_minifigs if len(minifig_to_sets.get(code, set())) == 1])

        changed = False
        exclusive_numbers = ",".join(exclusive)
        if row.get("ExclusiveMinifigNumbers") != exclusive_numbers:
            row["ExclusiveMinifigNumbers"] = exclusive_numbers
            changed = True

        if row.get("ExclusiveMinifigCount") != len(exclusive):
            row["ExclusiveMinifigCount"] = len(exclusive)
            changed = True

        if changed:
            set_changed_count += 1

    minifig_changed_count = 0
    for row in minifig_rows:
        number = collapse_ws(row.get("Number")).lower()
        sets = sorted(minifig_to_sets.get(number, set()))
        appears = ",".join(sets)

        changed = False
        if row.get("AppearsInSetNumbers") != appears:
            row["AppearsInSetNumbers"] = appears
            changed = True

        is_exclusive = len(sets) == 1
        if row.get("IsSetExclusive") != is_exclusive:
            row["IsSetExclusive"] = is_exclusive
            changed = True

        exclusive_set = sets[0] if is_exclusive else ""
        if row.get("ExclusiveToSetNumber") != exclusive_set:
            row["ExclusiveToSetNumber"] = exclusive_set
            changed = True

        if changed:
            minifig_changed_count += 1

    return (set_changed_count, minifig_changed_count)


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    rows: List[Dict[str, Any]] = []
    for row in data:
        if isinstance(row, dict):
            rows.append(row)
    return rows


def maybe_write_json(path: Path, rows: List[Dict[str, Any]]) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    if original == updated:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def load_json_object(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def write_json_object(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rotating_indices(total: int, start_index: int, excluded: set[int]) -> List[int]:
    if total <= 0:
        return []
    start = start_index % total
    ordered: List[int] = []
    for idx in range(start, total):
        if idx not in excluded:
            ordered.append(idx)
    for idx in range(0, start):
        if idx not in excluded:
            ordered.append(idx)
    return ordered


def print_summary(label: str, stats: FileUpdateStats) -> None:
    print(
        (
            f"[{label}] total={stats.total_rows} "
            f"considered={stats.rows_considered} "
            f"changed={stats.rows_changed} "
            f"fetch_failures={stats.fetch_failures} "
            f"parse_misses={stats.parse_misses} "
            f"cross_changed={stats.cross_rows_changed}"
        ),
        flush=True,
    )


def merge_update_stats(base: FileUpdateStats, add: FileUpdateStats) -> FileUpdateStats:
    base.rows_considered += add.rows_considered
    base.rows_changed += add.rows_changed
    base.fetch_failures += add.fetch_failures
    base.parse_misses += add.parse_misses
    base.cross_rows_changed += add.cross_rows_changed
    if add.last_index_processed is not None:
        base.last_index_processed = add.last_index_processed
    return base


def update_rows(
    rows: List[Dict[str, Any]],
    *,
    item_type: str,
    cfg: FetchConfig,
    client: BrickLinkClient,
    throttle: RuntimeThrottle,
    month_key: str,
    start_index: int,
    limit: Optional[int],
    indexes: Optional[Sequence[int]],
    label: str,
) -> FileUpdateStats:
    stats = FileUpdateStats(total_rows=len(rows))
    if indexes is not None:
        iter_indices = [i for i in indexes if 0 <= i < len(rows)]
    else:
        iter_indices = list(range(len(rows)))

    for idx in iter_indices:
        if client.auth_failed:
            break
        if client.budget_exhausted:
            break
        if indexes is None and idx < start_index:
            continue
        if limit is not None and stats.rows_considered >= limit:
            break

        row = rows[idx]
        stats.rows_considered += 1
        stats.last_index_processed = idx

        if item_type == "SET":
            item_no = normalize_set_code(row.get("Number"), row.get("Variant"))
        else:
            item_no = collapse_ws(row.get("Number"))

        if not item_no:
            stats.parse_misses += 1
            continue

        before = json.dumps(row, sort_keys=True, ensure_ascii=False)
        ok = apply_market_to_row(
            row,
            item_type=item_type,
            item_no=item_no,
            currency_code=cfg.currency_code,
            client=client,
            throttle=throttle,
            month_key=month_key,
        )
        if not ok:
            stats.fetch_failures += 1
            if cfg.verbose:
                print(f"[{label}] failed: {item_no}", flush=True)
            if client.auth_failed:
                break
            if client.budget_exhausted:
                break
            continue

        after = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if before != after:
            stats.rows_changed += 1

        if cfg.verbose:
            print(
                f"[{label}] {stats.rows_considered}/{stats.total_rows}: {item_no} updated",
                flush=True,
            )

    return stats


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update LEGO market values via BrickLink API.")
    parser.add_argument("--sets-json", default="dist/Lego Star Wars Database.json", help="Path to sets JSON file.")
    parser.add_argument("--minifigs-json", default="dist/Lego-Star-Wars-Minifigure-Database.json", help="Path to minifig JSON file.")

    parser.add_argument("--bricklink-base-url", default=BRICKLINK_API_BASE_URL, help="BrickLink API base URL.")
    parser.add_argument("--currency-code", default=os.getenv("BRICKLINK_CURRENCY", "GBP"), help="Price currency code (default GBP).")

    parser.add_argument("--consumer-key", default=os.getenv("BRICKLINK_CONSUMER_KEY", ""), help="BrickLink consumer key.")
    parser.add_argument("--consumer-secret", default=os.getenv("BRICKLINK_CONSUMER_SECRET", ""), help="BrickLink consumer secret.")
    parser.add_argument("--token-value", default=os.getenv("BRICKLINK_TOKEN_VALUE", ""), help="BrickLink token value.")
    parser.add_argument("--token-secret", default=os.getenv("BRICKLINK_TOKEN_SECRET", ""), help="BrickLink token secret.")

    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retries after first attempt.")
    parser.add_argument("--delay", type=float, default=0.15, help="Minimum delay between API requests.")
    parser.add_argument("--jitter", type=float, default=0.05, help="Random jitter added to delay.")
    parser.add_argument(
        "--max-api-calls",
        type=int,
        default=4800,
        help="Hard cap on BrickLink API requests for this run.",
    )
    parser.add_argument(
        "--market-state-json",
        default="dist/market-sync-state.json",
        help="State file for rotating market refresh cursor.",
    )
    parser.add_argument(
        "--catalog-sync-state-json",
        default="dist/sync-state.json",
        help="Optional sync state file used to prioritize recently changed set entries.",
    )
    parser.add_argument(
        "--priority-updated-limit",
        type=int,
        default=1200,
        help="Maximum recently changed set numbers to prioritize from catalog sync state.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional per-file row limit for testing.")
    parser.add_argument("--start-index", type=int, default=0, help="Optional per-file start index.")
    parser.add_argument("--skip-cross-enrichment", action="store_true", help="Skip exclusivity/appears-in enrichment.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    required = {
        "BRICKLINK_CONSUMER_KEY": collapse_ws(args.consumer_key),
        "BRICKLINK_CONSUMER_SECRET": collapse_ws(args.consumer_secret),
        "BRICKLINK_TOKEN_VALUE": collapse_ws(args.token_value),
        "BRICKLINK_TOKEN_SECRET": collapse_ws(args.token_secret),
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        print("Missing BrickLink credentials: " + ", ".join(missing), file=sys.stderr)
        return 1

    sets_path = Path(args.sets_json)
    minifigs_path = Path(args.minifigs_json)
    if not sets_path.exists():
        print(f"Missing sets JSON: {sets_path}", file=sys.stderr)
        return 1
    if not minifigs_path.exists():
        print(f"Missing minifig JSON: {minifigs_path}", file=sys.stderr)
        return 1

    cfg = FetchConfig(
        timeout=max(1.0, args.timeout),
        retries=max(0, args.retries),
        delay=max(0.0, args.delay),
        jitter=max(0.0, args.jitter),
        verbose=bool(args.verbose),
        currency_code=collapse_ws(args.currency_code).upper() or "GBP",
    )

    request_budget = ApiRequestBudget(
        max_calls=max(0, args.max_api_calls) if args.max_api_calls is not None else None
    )
    client = BrickLinkClient(
        consumer_key=required["BRICKLINK_CONSUMER_KEY"],
        consumer_secret=required["BRICKLINK_CONSUMER_SECRET"],
        token_value=required["BRICKLINK_TOKEN_VALUE"],
        token_secret=required["BRICKLINK_TOKEN_SECRET"],
        timeout=cfg.timeout,
        retries=cfg.retries,
        verbose=cfg.verbose,
        request_budget=request_budget,
        base_url=args.bricklink_base_url,
    )

    now = datetime.now(timezone.utc)
    month_key = now.strftime("%Y-%m")

    throttle = RuntimeThrottle(min_delay=cfg.delay, jitter=cfg.jitter)

    # Fail fast on invalid BrickLink OAuth credentials.
    _ = client.fetch_price_matrix(
        item_type="SET",
        item_no="7101-1",
        currency_code=cfg.currency_code,
        throttle=throttle,
    )
    if client.auth_failed:
        print(client.auth_error_message or "BrickLink API authentication failed.", file=sys.stderr)
        return 1

    sets_rows = load_json_array(sets_path)
    minifigs_rows = load_json_array(minifigs_path)
    market_state_path = Path(args.market_state_json)
    market_state = load_json_object(market_state_path)
    catalog_state = load_json_object(Path(args.catalog_sync_state_json))

    raw_changed_set_codes = catalog_state.get("lastUpdatedSetCodes")
    changed_set_codes: List[str] = []
    if isinstance(raw_changed_set_codes, list):
        for value in raw_changed_set_codes:
            code = collapse_ws(value).lower()
            if code:
                changed_set_codes.append(code)
    if args.priority_updated_limit is not None and args.priority_updated_limit >= 0:
        changed_set_codes = changed_set_codes[: args.priority_updated_limit]
    changed_set_lookup = set(changed_set_codes)

    def build_unique_plan(priority: List[int], rotating: List[int]) -> List[int]:
        out: List[int] = []
        seen: set[int] = set()
        for idx in priority + rotating:
            if idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
        return out

    set_priority_indices: List[int] = []
    for idx, row in enumerate(sets_rows):
        code = normalize_set_code(row.get("Number"), row.get("Variant")).lower()
        has_new = bool(collapse_ws(row.get("New")))
        has_used = bool(collapse_ws(row.get("Used")))
        if (not has_new) or (not has_used) or (code in changed_set_lookup):
            set_priority_indices.append(idx)

    minifig_priority_indices: List[int] = []
    for idx, row in enumerate(minifigs_rows):
        has_new = bool(collapse_ws(row.get("New")))
        has_used = bool(collapse_ws(row.get("Used")))
        if (not has_new) or (not has_used):
            minifig_priority_indices.append(idx)

    stored_set_cursor = _parse_int(market_state.get("nextSetIndex"))
    stored_minifig_cursor = _parse_int(market_state.get("nextMinifigIndex"))
    set_cursor = stored_set_cursor if stored_set_cursor is not None else max(0, args.start_index)
    minifig_cursor = stored_minifig_cursor if stored_minifig_cursor is not None else max(0, args.start_index)

    set_rotating_indices = rotating_indices(len(sets_rows), set_cursor, set(set_priority_indices))
    minifig_rotating_indices = rotating_indices(len(minifigs_rows), minifig_cursor, set(minifig_priority_indices))
    set_plan = build_unique_plan(set_priority_indices, set_rotating_indices)
    minifig_plan = build_unique_plan(minifig_priority_indices, minifig_rotating_indices)

    if cfg.verbose:
        print(
            (
                f"[Plan] set_priority={len(set_priority_indices)} set_rotating={len(set_rotating_indices)} "
                f"minifig_priority={len(minifig_priority_indices)} minifig_rotating={len(minifig_rotating_indices)}"
            ),
            flush=True,
        )

    sets_stats = update_rows(
        sets_rows,
        item_type="SET",
        cfg=cfg,
        client=client,
        throttle=throttle,
        month_key=month_key,
        start_index=max(0, args.start_index),
        limit=args.limit,
        indexes=set_plan,
        label="Sets",
    )

    minifigs_stats = update_rows(
        minifigs_rows,
        item_type="MINIFIG",
        cfg=cfg,
        client=client,
        throttle=throttle,
        month_key=month_key,
        start_index=max(0, args.start_index),
        limit=args.limit,
        indexes=minifig_plan,
        label="Minifigs",
    )

    next_set_cursor = set_cursor
    if sets_rows and sets_stats.last_index_processed is not None and sets_stats.last_index_processed in set(set_rotating_indices):
        next_set_cursor = (sets_stats.last_index_processed + 1) % len(sets_rows)
    next_minifig_cursor = minifig_cursor
    if minifigs_rows and minifigs_stats.last_index_processed is not None and minifigs_stats.last_index_processed in set(minifig_rotating_indices):
        next_minifig_cursor = (minifigs_stats.last_index_processed + 1) % len(minifigs_rows)

    if client.auth_failed:
        print(client.auth_error_message or "BrickLink API authentication failed.", file=sys.stderr)
        return 1

    if not args.skip_cross_enrichment:
        set_cross, minifig_cross = apply_cross_catalog_enrichment(sets_rows, minifigs_rows)
        sets_stats.cross_rows_changed = set_cross
        minifigs_stats.cross_rows_changed = minifig_cross

    sets_written = maybe_write_json(sets_path, sets_rows)
    minifigs_written = maybe_write_json(minifigs_path, minifigs_rows)

    market_state.update(
        {
            "nextSetIndex": next_set_cursor,
            "nextMinifigIndex": next_minifig_cursor,
            "lastRunUTC": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lastMonthKey": month_key,
            "lastApiRequestsUsed": client.request_budget.used_calls,
            "lastApiRequestCap": client.request_budget.max_calls,
            "lastApiBudgetExhausted": client.budget_exhausted,
            "lastSetRowsConsidered": sets_stats.rows_considered,
            "lastMinifigRowsConsidered": minifigs_stats.rows_considered,
            "lastSetPriorityCount": len(set_priority_indices),
            "lastMinifigPriorityCount": len(minifig_priority_indices),
            "lastChangedSetPriorityCount": len(changed_set_codes),
        }
    )
    write_json_object(market_state_path, market_state)

    if cfg.verbose:
        print(f"[Write] sets_written={sets_written} minifigs_written={minifigs_written}", flush=True)

    cap_text = "unlimited" if client.request_budget.max_calls is None else str(client.request_budget.max_calls)
    print(
        (
            f"[API] requests_used={client.request_budget.used_calls} "
            f"cap={cap_text} exhausted={client.budget_exhausted}"
        ),
        flush=True,
    )
    print_summary("Sets", sets_stats)
    print_summary("Minifigs", minifigs_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
