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
import html
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
from urllib.parse import parse_qs, quote, urlparse

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

MONTH_NAME_TO_NUMBER = {name.lower(): idx for idx, name in MONTH_LABELS.items()}

HTML_PRICE_GUIDE_BASE_URL = "https://www.bricklink.com/catalogPG.asp"
HTML_MONTH_RE = re.compile(
    r"<B>\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s*(?:&nbsp;|\s)+(\d{4})\s*</B>",
    re.IGNORECASE | re.DOTALL,
)

MARKET_PRESERVE_FIELDS = {
    "BrickLinkSoldPriceNew",
    "BrickLinkSoldPriceUsed",
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
    "CurrentNewVsRRPAmount",
    "CurrentNewVsRRPPercent",
    "CurrentRRPBaseline",
    "PriceForecast2YNew",
    "PriceForecast5YNew",
    "PriceForecast2YUsed",
    "PriceForecast5YUsed",
    "PriceTrendAnnualizedNewPercent",
    "PriceTrendAnnualizedUsedPercent",
    "PriceForecastMethod",
    "BrickLinkNewPriceRangeMin",
    "BrickLinkNewPriceRangeMax",
    "BrickLinkUsedPriceRangeMin",
    "BrickLinkUsedPriceRangeMax",
    "BrickLinkCurrentListingsNew",
    "BrickLinkCurrentListingsUsed",
}

BRICKLINK_MINIFIG_CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")



@dataclass
class FetchConfig:
    timeout: float
    retries: int
    delay: float
    jitter: float
    verbose: bool
    currency_code: str
    fallback_currency_codes: Tuple[str, ...] = ()
    allow_html_fallback: bool = True


@dataclass
class FileUpdateStats:
    total_rows: int = 0
    rows_considered: int = 0
    rows_succeeded: int = 0
    rows_changed: int = 0
    fetch_failures: int = 0
    no_price_data_skips: int = 0
    parse_misses: int = 0
    cross_rows_changed: int = 0
    cooldown_skips: int = 0
    last_index_processed: Optional[int] = None
    processed_indices: List[int] = None

    def __post_init__(self) -> None:
        if self.processed_indices is None:
            self.processed_indices = []


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
        self.last_error_kind = ""
        self.last_http_status: Optional[int] = None
        self.request_budget = request_budget or ApiRequestBudget(max_calls=None)
        self.html_headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.9",
            "User-Agent": "Mozilla/5.0 (compatible; lsw-checklist-market-sync/1.0)",
        }

    def _mark_failure(self, kind: str, http_status: Optional[int] = None) -> None:
        self.last_error_kind = kind
        self.last_http_status = http_status

    def _mark_success(self, http_status: Optional[int] = None) -> None:
        self.last_error_kind = ""
        self.last_http_status = http_status

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

        # Some BrickLink responses only become parseable when explicitly scoped
        # to guide_type + condition. Probe those combinations when the aggregate
        # response is empty or structurally ambiguous.
        if matrix is None:
            # Hard parameter/API failures should not fan out into 4 extra probes.
            # That burns budget and repeatedly fails for malformed/nonexistent IDs.
            if self.last_error_kind in {
                "auth",
                "budget",
                "not_found",
                "rate_limited",
                "server_error",
                "http_error",
                "api_error",
                "invalid_json",
            }:
                return None
            matrix = {}

        needed: List[Tuple[str, str]] = [
            ("sold", "N"),
            ("sold", "U"),
            ("stock", "N"),
            ("stock", "U"),
        ]
        missing = [pair for pair in needed if pair not in matrix]
        if not missing:
            return matrix

        max_probes = len(missing) if not matrix else min(2, len(missing))
        terminal_probe_errors = {
            "auth",
            "budget",
            "not_found",
            "rate_limited",
            "server_error",
            "http_error",
            "api_error",
            "invalid_json",
        }
        for guide_type, condition in missing:
            if max_probes <= 0:
                break
            max_probes -= 1
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
                if self.last_error_kind in terminal_probe_errors:
                    break
                continue
            matrix.update(sub)
            if all(pair in matrix for pair in needed):
                break

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
                self._mark_failure("budget")
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
                    self._mark_failure("request_error")
                    if self.verbose:
                        print(f"[API] request failed {item_type}:{item_no}: {exc}", flush=True)
                    return None
                throttle.apply_backoff()
                continue

            if response.status_code == 429:
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                throttle.apply_backoff(retry_after)
                if attempt > self.retries + 1:
                    self._mark_failure("rate_limited", 429)
                    if self.verbose:
                        print(f"[API] HTTP 429 {item_type}:{item_no}", flush=True)
                    return None
                continue

            if response.status_code >= 500:
                throttle.apply_backoff()
                if attempt > self.retries + 1:
                    self._mark_failure("server_error", response.status_code)
                    if self.verbose:
                        print(
                            f"[API] HTTP {response.status_code} {item_type}:{item_no}",
                            flush=True,
                        )
                    return None
                continue

            if response.status_code >= 400:
                if response.status_code == 401:
                    body_text = ""
                    try:
                        body_json = response.json()
                        if isinstance(body_json, dict):
                            meta_body = body_json.get("meta") if isinstance(body_json.get("meta"), dict) else {}
                            body_text = collapse_ws(meta_body.get("description") or meta_body.get("message"))
                    except Exception:
                        body_text = ""
                    self.auth_failed = True
                    if "TOKEN_IP_MISMATCHED" in body_text.upper():
                        self.auth_error_message = (
                            "BrickLink API authentication failed (TOKEN_IP_MISMATCHED). "
                            "Update the BrickLink access token allowed IP/mask to include GitHub runner IPs."
                        )
                    else:
                        self.auth_error_message = (
                            "BrickLink API authentication failed (HTTP 401). "
                            "Check BRICKLINK_CONSUMER_KEY/SECRET and BRICKLINK_TOKEN_VALUE/SECRET."
                        )
                    self._mark_failure("auth", 401)
                elif response.status_code == 404:
                    self._mark_failure("not_found", 404)
                else:
                    self._mark_failure("http_error", response.status_code)
                if self.verbose:
                    print(
                        f"[API] HTTP {response.status_code} {item_type}:{item_no} params={params}",
                        flush=True,
                    )
                return None

            try:
                payload = response.json()
            except ValueError:
                self._mark_failure("invalid_json")
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
                        description = collapse_ws(meta.get("description"))
                        if "TOKEN_IP_MISMATCHED" in description.upper():
                            self.auth_error_message = (
                                "BrickLink API authentication failed (TOKEN_IP_MISMATCHED). "
                                "Update the BrickLink access token allowed IP/mask to include GitHub runner IPs."
                            )
                        elif "BAD_OAUTH_REQUEST" in message.upper() and "TOKEN_IP_MISMATCHED" in description.upper():
                            self.auth_error_message = (
                                "BrickLink API authentication failed (BAD_OAUTH_REQUEST / TOKEN_IP_MISMATCHED). "
                                "Update the BrickLink access token allowed IP/mask to include GitHub runner IPs."
                            )
                        elif message:
                            self.auth_error_message = (
                                f"BrickLink API authentication failed ({message}). "
                                "Check BRICKLINK_CONSUMER_KEY/SECRET and BRICKLINK_TOKEN_VALUE/SECRET."
                            )
                        else:
                            self.auth_error_message = (
                                "BrickLink API authentication failed (meta code 401). "
                                "Check BRICKLINK_CONSUMER_KEY/SECRET and BRICKLINK_TOKEN_VALUE/SECRET."
                            )
                        self._mark_failure("auth", 401)
                    elif code == 404:
                        self._mark_failure("not_found", 404)
                    else:
                        self._mark_failure("api_error", code)
                    if self.verbose:
                        print(
                            f"[API] meta code={code} {item_type}:{item_no} msg={meta.get('message')}",
                            flush=True,
                        )
                    return None

            data = payload.get("data") if isinstance(payload, dict) else None
            data_rows = _extract_matrix_rows(data)
            if data_rows is None:
                self._mark_failure("no_data", response.status_code)
                if self.verbose:
                    preview = ""
                    if isinstance(data, dict):
                        keys = [str(key) for key in list(data.keys())[:8]]
                        preview = f" keys={keys}"
                    elif data is not None:
                        preview = f" type={type(data).__name__}"
                    print(f"[API] missing data array {item_type}:{item_no}{preview}", flush=True)
                return None

            matrix: Dict[Tuple[str, str], Dict[str, Any]] = {}
            default_guide_type = _normalize_guide_type(params.get("guide_type"))
            default_condition = _normalize_condition(params.get("new_or_used"))

            for row in data_rows:
                if not isinstance(row, dict):
                    continue
                guide_type = _normalize_guide_type(row.get("guide_type")) or default_guide_type
                condition = (
                    _normalize_condition(row.get("new_or_used"))
                    or _normalize_condition(row.get("condition"))
                    or default_condition
                )
                if condition is None:
                    is_new = row.get("is_new")
                    if isinstance(is_new, bool):
                        condition = "N" if is_new else "U"

                if not guide_type or not condition:
                    continue
                matrix[(guide_type, condition)] = row

            throttle.apply_success()
            self._mark_success(response.status_code)
            return matrix

    def fetch_price_guide_html(
        self,
        item_type: str,
        item_no: str,
        throttle: RuntimeThrottle,
        quiet_no_data: bool = False,
    ) -> Optional[
        Tuple[
            Dict[Tuple[str, str], Dict[str, Any]],
            List[Dict[str, Any]],
            List[Dict[str, Any]],
            List[Dict[str, Any]],
            List[Dict[str, Any]],
            List[Dict[str, Any]],
            List[Dict[str, Any]],
            str,
        ]
    ]:
        code = collapse_ws(item_no)
        if not code:
            self._mark_failure("no_data")
            return None

        item_type_upper = collapse_ws(item_type).upper()
        if item_type_upper not in {"SET", "MINIFIG"}:
            self._mark_failure("parse_error")
            return None

        url = build_html_price_guide_url(item_type_upper, code)

        attempt = 0
        while True:
            attempt += 1
            if not self.request_budget.consume():
                self._mark_failure("budget")
                if self.verbose:
                    print(
                        f"[HTML] request budget exhausted before {item_type}:{item_no}",
                        flush=True,
                    )
                return None

            throttle.sleep_between_requests()
            try:
                response = self.session.get(
                    url,
                    timeout=self.timeout,
                    headers=self.html_headers,
                )
            except requests.RequestException as exc:
                if attempt > self.retries + 1:
                    self._mark_failure("request_error")
                    if self.verbose:
                        print(f"[HTML] request failed {item_type}:{item_no}: {exc}", flush=True)
                    return None
                throttle.apply_backoff()
                continue

            if response.status_code == 429:
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                throttle.apply_backoff(retry_after)
                if attempt > self.retries + 1:
                    self._mark_failure("rate_limited", 429)
                    if self.verbose:
                        print(f"[HTML] HTTP 429 {item_type}:{item_no}", flush=True)
                    return None
                continue

            if response.status_code >= 500:
                throttle.apply_backoff()
                if attempt > self.retries + 1:
                    self._mark_failure("server_error", response.status_code)
                    if self.verbose:
                        print(f"[HTML] HTTP {response.status_code} {item_type}:{item_no}", flush=True)
                    return None
                continue

            if response.status_code >= 400:
                if response.status_code == 404:
                    self._mark_failure("not_found", 404)
                else:
                    self._mark_failure("http_error", response.status_code)
                if self.verbose:
                    print(
                        f"[HTML] HTTP {response.status_code} {item_type}:{item_no}",
                        flush=True,
                    )
                return None

            parsed = _parse_price_guide_html(response.text)
            if parsed is None:
                self._mark_failure("no_data", response.status_code)
                if self.verbose and not quiet_no_data:
                    print(f"[HTML] no parseable price data {item_type}:{item_no}", flush=True)
                return None

            matrix, month_new, month_used, tx_new, tx_used, listings_new, listings_used, currency = parsed
            throttle.apply_success()
            self._mark_success(response.status_code)
            return (matrix, month_new, month_used, tx_new, tx_used, listings_new, listings_used, currency)

    def fetch_set_alias_from_brickset(
        self,
        brickset_url: Any,
        throttle: RuntimeThrottle,
    ) -> Optional[str]:
        raw = collapse_ws(brickset_url)
        if not raw:
            return None
        try:
            parsed = urlparse(raw)
        except Exception:
            return None
        host = (parsed.netloc or "").lower()
        if "brickset.com" not in host:
            return None

        attempt = 0
        while True:
            attempt += 1
            throttle.sleep_between_requests()
            try:
                response = self.session.get(
                    raw,
                    timeout=self.timeout,
                    headers=self.html_headers,
                )
            except requests.RequestException:
                if attempt > self.retries + 1:
                    return None
                throttle.apply_backoff()
                continue

            if response.status_code == 429:
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                throttle.apply_backoff(retry_after)
                if attempt > self.retries + 1:
                    return None
                continue
            if response.status_code >= 500:
                throttle.apply_backoff()
                if attempt > self.retries + 1:
                    return None
                continue
            if response.status_code >= 400:
                return None

            html_text = response.text or ""
            match = re.search(
                r"https?://(?:www\.)?bricklink\.com/v2/catalog/catalogitem\.page\?S=([^#\"'&<>\s]+)",
                html_text,
                re.IGNORECASE,
            )
            if not match:
                return None
            alias = canonicalize_set_item_no(html.unescape(match.group(1)))
            return alias or None


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


def _extract_matrix_rows(data: Any) -> Optional[List[Dict[str, Any]]]:
    if data is None:
        return []
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if not isinstance(data, dict):
        return None

    # Some responses return a single row object instead of an array.
    if data.get("guide_type") is not None and (
        data.get("new_or_used") is not None or data.get("condition") is not None
    ):
        row = dict(data)
        if row.get("new_or_used") is None and row.get("condition") is not None:
            row["new_or_used"] = row.get("condition")
        return [row]

    # Single-row payloads may omit guide/condition fields when those values
    # are implied by request parameters.
    if any(
        key in data
        for key in (
            "min_price",
            "avg_price",
            "max_price",
            "qty_avg_price",
            "unit_quantity",
            "total_quantity",
        )
    ):
        return [dict(data)]

    # Common nested containers seen in API wrappers / gateway transforms.
    for key in (
        "price_detail",
        "price_details",
        "rows",
        "items",
        "guides",
        "price_guides",
        "data",
    ):
        nested = data.get(key)
        if isinstance(nested, list):
            return [row for row in nested if isinstance(row, dict)]
        if isinstance(nested, dict):
            inner = _extract_matrix_rows(nested)
            if inner is not None:
                return inner

    # Flatten shaped keys like stock_new / sold_used.
    synthesized: List[Dict[str, Any]] = []
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        normalized_key = str(key).strip().lower().replace("-", "_")
        guide_type: Optional[str] = None
        condition: Optional[str] = None
        if "stock" in normalized_key or "current" in normalized_key:
            guide_type = "stock"
        elif "sold" in normalized_key:
            guide_type = "sold"
        if normalized_key.endswith("_n") or "new" in normalized_key:
            condition = "N"
        elif normalized_key.endswith("_u") or "used" in normalized_key:
            condition = "U"
        if not guide_type or not condition:
            continue
        row = dict(value)
        row.setdefault("guide_type", guide_type)
        row.setdefault("new_or_used", condition)
        synthesized.append(row)

    if synthesized:
        return synthesized
    return None


def _html_to_text(raw: Any) -> str:
    if raw is None:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(raw))
    text = html.unescape(text).replace("\xa0", " ")
    return collapse_ws(text)


def _extract_block_metric_text(block_html: str, label_pattern: str) -> str:
    match = re.search(label_pattern + r".*?<B>(.*?)</B>", block_html, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return _html_to_text(match.group(1))


def build_html_price_guide_url(item_type: str, item_no: str) -> str:
    if collapse_ws(item_type).upper() == "SET":
        key = "S"
    else:
        key = "M"
    # Keep parity with BrickLink's "Exclude incomplete sets" and
    # grouped-by-currency price-guide mode used in manual verification.
    return (
        f"{HTML_PRICE_GUIDE_BASE_URL}?{key}={quote(item_no, safe='')}"
        "&ColorID=0&v=D&viewExclude=Y&cID=Y"
    )


def _parse_price_and_currency(raw: Any) -> Tuple[Optional[float], Optional[str]]:
    text = _html_to_text(raw).replace("~", " ").strip()
    if not text:
        return (None, None)

    code_match = re.search(r"\b([A-Z]{3})\b\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if code_match:
        amount = _parse_float(code_match.group(2))
        currency = collapse_ws(code_match.group(1)).upper() or None
        return (amount, currency)

    symbol_match = re.search(r"([£$€])\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)", text)
    if symbol_match:
        symbol = symbol_match.group(1)
        amount = _parse_float(symbol_match.group(2))
        symbol_map = {"£": "GBP", "$": "USD", "€": "EUR"}
        return (amount, symbol_map.get(symbol))

    return (_parse_float(text), None)


def _parse_price_guide_summary_block(
    block_html: str,
    *,
    guide_type: str,
    condition: str,
    fallback_currency: Optional[str],
) -> Optional[Dict[str, Any]]:
    qty_value = _extract_block_metric_text(block_html, r"(?:Times Sold|Total Lots):")
    total_qty_value = _extract_block_metric_text(block_html, r"Total Qty:")
    min_price_raw = _extract_block_metric_text(block_html, r"Min Price:")
    avg_price_raw = _extract_block_metric_text(block_html, r"Avg Price:")
    qty_avg_raw = _extract_block_metric_text(block_html, r"Qty Avg Price:")
    max_price_raw = _extract_block_metric_text(block_html, r"Max Price:")

    min_price, min_currency = _parse_price_and_currency(min_price_raw)
    avg_price, avg_currency = _parse_price_and_currency(avg_price_raw)
    qty_avg_price, qty_avg_currency = _parse_price_and_currency(qty_avg_raw)
    max_price, max_currency = _parse_price_and_currency(max_price_raw)
    currency = (
        min_currency
        or avg_currency
        or qty_avg_currency
        or max_currency
        or (collapse_ws(fallback_currency).upper() or None)
    )

    row: Dict[str, Any] = {
        "guide_type": guide_type,
        "new_or_used": condition,
        "unit_quantity": _parse_int(qty_value),
        "total_quantity": _parse_int(total_qty_value),
        "min_price": min_price,
        "avg_price": avg_price,
        "qty_avg_price": qty_avg_price,
        "max_price": max_price,
        "currency_code": currency,
    }

    if not any(
        value is not None
        for value in (
            row.get("avg_price"),
            row.get("min_price"),
            row.get("max_price"),
            row.get("qty_avg_price"),
            row.get("unit_quantity"),
            row.get("total_quantity"),
        )
    ):
        return None
    return row


def _parse_monthly_sales_column_html(
    column_html: str,
    *,
    fallback_currency: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    month_matches = list(HTML_MONTH_RE.finditer(column_html))
    if not month_matches:
        return ([], [])

    rows: List[Dict[str, Any]] = []
    transactions: List[Dict[str, Any]] = []
    seen_months: set[str] = set()

    for idx, match in enumerate(month_matches):
        month_name = collapse_ws(match.group(1))
        year_text = collapse_ws(match.group(2))
        month_num = MONTH_NAME_TO_NUMBER.get(month_name.lower())
        year_num = _parse_int(year_text)
        if month_num is None or year_num is None:
            continue

        start = match.start()
        end = month_matches[idx + 1].start() if idx + 1 < len(month_matches) else len(column_html)
        block = column_html[start:end]

        avg_price_raw = _extract_block_metric_text(block, r"Avg Price:")
        total_qty_raw = _extract_block_metric_text(block, r"Total Qty:")
        times_sold_raw = _extract_block_metric_text(block, r"Times Sold:")

        month_key = f"{year_num:04d}-{month_num:02d}"
        if month_key in seen_months:
            continue
        seen_months.add(month_key)

        monthly_tx: List[Dict[str, Any]] = []
        for tx_match in re.finditer(
            r"<TR\s+ALIGN=\"RIGHT\"[^>]*>\s*"
            r"<TD[^>]*>.*?</TD>\s*"
            r"<TD[^>]*>(.*?)</TD>\s*"
            r"<TD[^>]*>(.*?)</TD>\s*"
            r"</TR>",
            block,
            re.IGNORECASE | re.DOTALL,
        ):
            qty_raw = _html_to_text(tx_match.group(1))
            each_raw = _html_to_text(tx_match.group(2))
            # Guard against summary rows (e.g. "Total Qty", "Avg Price") that
            # match the generic row regex but are not actual transaction lines.
            if not re.search(r"([£$€]|~|\b[A-Z]{3}\b)", each_raw):
                continue
            qty_text = collapse_ws(qty_raw)
            if not re.fullmatch(r"\d{1,3}", qty_text):
                continue
            qty = _parse_int(qty_text)
            each_price, tx_currency = _parse_price_and_currency(each_raw)
            if qty is None or each_price is None:
                continue
            monthly_tx.append(
                {
                    "month": month_key,
                    "monthLabel": month_label(month_key),
                    "sequence": len(monthly_tx) + 1,
                    "qty": max(1, qty),
                    "eachPrice": round(each_price, 2),
                    "currency": collapse_ws(tx_currency or fallback_currency).upper() or None,
                }
            )

        avg_price, _currency = _parse_price_and_currency(avg_price_raw)
        if avg_price is None and monthly_tx:
            qty_weighted_total = 0.0
            qty_total = 0
            for tx in monthly_tx:
                each = _parse_float(tx.get("eachPrice"))
                qty = _parse_int(tx.get("qty")) or 0
                if each is None or qty <= 0:
                    continue
                qty_weighted_total += each * qty
                qty_total += qty
            if qty_total > 0:
                avg_price = qty_weighted_total / qty_total
        if avg_price is None:
            continue

        total_qty = _parse_int(total_qty_raw)
        if total_qty is None and monthly_tx:
            total_qty = sum(_parse_int(tx.get("qty")) or 0 for tx in monthly_tx)
        total_lots = _parse_int(times_sold_raw)
        if total_lots is None and monthly_tx:
            total_lots = len(monthly_tx)

        rows.append(
            {
                "month": month_key,
                "monthLabel": month_label(month_key),
                "avgPrice": round(avg_price, 2),
                "totalLots": total_lots,
                "totalQty": total_qty,
                "currency": collapse_ws(fallback_currency).upper() or None,
            }
        )
        transactions.extend(monthly_tx)

    rows.sort(key=lambda row: row.get("month") or "")
    return (rows, transactions)


def _parse_current_listings_column_html(
    column_html: str,
    *,
    fallback_currency: Optional[str],
) -> List[Dict[str, Any]]:
    listings: List[Dict[str, Any]] = []
    fallback_cc = collapse_ws(fallback_currency).upper() or None

    for tx_match in re.finditer(
        r"<TR\s+ALIGN=\"RIGHT\"[^>]*>\s*"
        r"<TD[^>]*>(.*?)</TD>\s*"
        r"<TD[^>]*>(.*?)</TD>\s*"
        r"<TD[^>]*>(.*?)</TD>\s*"
        r"</TR>",
        column_html,
        re.IGNORECASE | re.DOTALL,
    ):
        link_cell = tx_match.group(1)
        qty_raw = _html_to_text(tx_match.group(2))
        each_raw = _html_to_text(tx_match.group(3))

        if not re.search(r"([£$€]|~|\b[A-Z]{3}\b)", each_raw):
            continue
        qty_text = collapse_ws(qty_raw)
        if not re.fullmatch(r"\d{1,3}", qty_text):
            continue

        qty = _parse_int(qty_text)
        each_price, tx_currency = _parse_price_and_currency(each_raw)
        if qty is None or each_price is None:
            continue

        href_match = re.search(r"HREF=['\"]([^'\"]+)['\"]", link_cell, re.IGNORECASE)
        listing_url = ""
        store_id: Optional[int] = None
        if href_match:
            href = collapse_ws(html.unescape(href_match.group(1)))
            if href.startswith("/"):
                listing_url = f"https://www.bricklink.com{href}"
            elif href.startswith("http://") or href.startswith("https://"):
                listing_url = href
            if href:
                parsed_qs = parse_qs(urlparse(href).query or "")
                store_id = _parse_int((parsed_qs.get("sID") or [""])[0])

        listings.append(
            {
                "sequence": len(listings) + 1,
                "qty": max(1, qty),
                "eachPrice": round(each_price, 2),
                "currency": collapse_ws(tx_currency or fallback_cc).upper() or None,
                "listingURL": listing_url or None,
                "storeId": store_id,
                "region": None,
            }
        )

    return listings


def _parse_price_guide_html(
    html_text: str,
) -> Optional[
    Tuple[
        Dict[Tuple[str, str], Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        str,
    ]
]:
    if not html_text:
        return None
    lower = html_text.lower()
    if "internal server error" in lower and "catalogpg" not in lower:
        return None

    currency_match = re.search(r"Showing prices in.*?\(([A-Z]{3})\)", html_text, re.IGNORECASE | re.DOTALL)
    page_currency = collapse_ws(currency_match.group(1)).upper() if currency_match else ""

    summary_match = re.search(
        r"<TR\b[^>]*BGCOLOR=['\"]#C0C0C0['\"][^>]*>(.*?)<TR\b[^>]*VALIGN=['\"]TOP['\"]",
        html_text,
        re.IGNORECASE | re.DOTALL,
    )
    if not summary_match:
        return None
    summary_segment = summary_match.group(1)
    # Keep positional columns even when sold blocks are "(Unavailable)" and do
    # not contain Avg/Max rows. Requiring all four parseable blocks causes valid
    # pages to be dropped entirely (e.g. stock exists, sold unavailable).
    split_blocks = re.split(
        r"<TD\b[^>]*VALIGN=['\"]TOP['\"][^>]*>",
        summary_segment,
        flags=re.IGNORECASE | re.DOTALL,
    )
    summary_blocks = [block for block in split_blocks[1:] if collapse_ws(block)]
    if len(summary_blocks) < 2:
        return None

    sold_new = _parse_price_guide_summary_block(
        summary_blocks[0],
        guide_type="sold",
        condition="N",
        fallback_currency=page_currency,
    ) if len(summary_blocks) >= 1 else None
    sold_used = _parse_price_guide_summary_block(
        summary_blocks[1],
        guide_type="sold",
        condition="U",
        fallback_currency=page_currency,
    ) if len(summary_blocks) >= 2 else None
    stock_new = _parse_price_guide_summary_block(
        summary_blocks[2],
        guide_type="stock",
        condition="N",
        fallback_currency=page_currency,
    ) if len(summary_blocks) >= 3 else None
    stock_used = _parse_price_guide_summary_block(
        summary_blocks[3],
        guide_type="stock",
        condition="U",
        fallback_currency=page_currency,
    ) if len(summary_blocks) >= 4 else None

    matrix: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if sold_new is not None:
        matrix[("sold", "N")] = sold_new
    if sold_used is not None:
        matrix[("sold", "U")] = sold_used
    if stock_new is not None:
        matrix[("stock", "N")] = stock_new
    if stock_used is not None:
        matrix[("stock", "U")] = stock_used
    if not matrix:
        return None

    currency = ""
    for row in matrix.values():
        cc = collapse_ws(row.get("currency_code")).upper()
        if cc:
            currency = cc
            break
    if not currency:
        currency = page_currency or "GBP"
    for row in matrix.values():
        if not collapse_ws(row.get("currency_code")):
            row["currency_code"] = currency

    column_positions = [
        m.start()
        for m in re.finditer(r"<TD\b[^>]*\bWIDTH\s*=\s*['\"]?25%['\"]?[^>]*>", html_text, re.IGNORECASE)
    ]
    month_new: List[Dict[str, Any]] = []
    month_used: List[Dict[str, Any]] = []
    tx_new: List[Dict[str, Any]] = []
    tx_used: List[Dict[str, Any]] = []
    listings_new: List[Dict[str, Any]] = []
    listings_used: List[Dict[str, Any]] = []
    if len(column_positions) >= 2:
        sold_new_col = html_text[column_positions[0] : column_positions[1]]
        sold_used_col = html_text[
            column_positions[1] : column_positions[2] if len(column_positions) > 2 else len(html_text)
        ]
        month_new, tx_new = _parse_monthly_sales_column_html(sold_new_col, fallback_currency=currency)
        month_used, tx_used = _parse_monthly_sales_column_html(sold_used_col, fallback_currency=currency)

        if len(column_positions) >= 4:
            stock_new_col = html_text[column_positions[2] : column_positions[3]]
            stock_used_col = html_text[column_positions[3] : len(html_text)]
            listings_new = _parse_current_listings_column_html(stock_new_col, fallback_currency=currency)
            listings_used = _parse_current_listings_column_html(stock_used_col, fallback_currency=currency)
        else:
            listings_new = _parse_current_listings_column_html(sold_new_col, fallback_currency=currency)
            listings_used = _parse_current_listings_column_html(sold_used_col, fallback_currency=currency)

    return (matrix, month_new, month_used, tx_new, tx_used, listings_new, listings_used, currency)


def _normalize_guide_type(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"stock", "sold"}:
        return text
    if text in {"s", "current", "available"}:
        return "stock"
    if text in {"c", "history", "sales"}:
        return "sold"
    if "stock" in text or "current" in text:
        return "stock"
    if "sold" in text:
        return "sold"
    return None


def _normalize_condition(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"N", "U"}:
        return text
    if text in {"NEW", "SEALED", "MISB"}:
        return "N"
    if text in {"USED", "COMPLETE"}:
        return "U"
    if text.startswith("N"):
        return "N"
    if text.startswith("U"):
        return "U"
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


def parse_iso_utc(value: Any) -> Optional[datetime]:
    text = collapse_ws(value)
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def infer_fetch_status(last_error_kind: str, http_status: Optional[int]) -> str:
    if last_error_kind == "not_found":
        return "not_found"
    if last_error_kind == "no_data":
        return "no_data"
    if last_error_kind == "budget":
        return "budget_exhausted"
    if last_error_kind == "auth":
        return "auth_error"
    if last_error_kind == "rate_limited":
        return "rate_limited"
    if last_error_kind == "server_error":
        return "server_error"
    if last_error_kind == "request_error":
        return "request_error"
    if last_error_kind in {"http_error", "api_error"} and http_status is not None:
        return f"http_{http_status}"
    if last_error_kind:
        return last_error_kind
    return "failed"


def parse_status_retry_at(row: Dict[str, Any]) -> Optional[datetime]:
    return parse_iso_utc(row.get("MarketNoDataRetryAfterUTC"))


def sanitize_secret(value: Any) -> str:
    text = collapse_ws(value)
    # Common copy/paste issues from UI fields / env files.
    for ch in ("\ufeff", "\u200b", "\u200e", "\u200f"):
        text = text.replace(ch, "")
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text


def bricklink_auth_help_hint() -> str:
    return (
        "Use GitHub repository Actions secrets (Settings > Secrets and variables > Actions), "
        "not Codespaces secrets, and paste the four BrickLink values without quotes."
    )


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


def normalize_currency_code(value: Any) -> str:
    code = collapse_ws(value).upper()
    if not code:
        return ""
    return re.sub(r"[^A-Z]", "", code)


def build_currency_try_order(primary: str, fallbacks: Sequence[str]) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()

    first = normalize_currency_code(primary)
    if first:
        ordered.append(first)
        seen.add(first)

    for value in fallbacks:
        code = normalize_currency_code(value)
        if not code or code in seen:
            continue
        ordered.append(code)
        seen.add(code)
    return ordered


def normalize_set_code(number: Any, variant: Any) -> str:
    raw = collapse_ws(number)
    if not raw:
        return ""
    if re.search(r"-[0-9]+$", raw):
        return raw
    var = _parse_int(variant) or 1
    return f"{raw}-{var}"


def resolve_rrp_from_row(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    existing_rrp = _parse_float(row.get("RRP"))
    existing_currency = normalize_currency_code(row.get("RRPCurrency"))
    if existing_rrp is not None and existing_rrp > 0 and existing_currency:
        return (round(existing_rrp, 2), existing_currency)

    # Prefer region-specific retail price fields in deterministic order.
    candidate_fields: List[Tuple[str, str]] = [
        ("GBP", "UKRetailPrice"),
        ("USD", "USRetailPrice"),
        ("CAD", "CARetailPrice"),
        ("EUR", "DERetailPrice"),
    ]
    for currency_code, field_name in candidate_fields:
        value = _parse_float(row.get(field_name))
        if value is not None and value > 0:
            return (round(value, 2), currency_code)

    if existing_rrp is not None and existing_rrp > 0:
        return (round(existing_rrp, 2), existing_currency or None)
    return (None, None)


def parse_bricklink_item_reference(link: Any) -> Optional[Tuple[str, str]]:
    raw = collapse_ws(link)
    if not raw:
        return None
    try:
        parsed = urlparse(raw)
    except Exception:
        return None
    host = (parsed.netloc or "").lower()
    if "bricklink.com" not in host:
        return None
    query = parse_qs(parsed.query or "")
    set_code = canonicalize_set_item_no((query.get("S") or [""])[0])
    if set_code:
        return ("SET", set_code)
    minifig_code = canonicalize_minifig_item_no((query.get("M") or [""])[0])
    if minifig_code:
        return ("MINIFIG", minifig_code)
    return None


def canonicalize_set_item_no(value: Any) -> str:
    raw = collapse_ws(value)
    if not raw:
        return ""
    raw = raw.split("#", 1)[0]
    raw = re.sub(r"-+$", "", raw)
    raw = re.sub(r"\s+", "", raw)
    if not raw:
        return ""

    match = re.match(r"^(.+)-([0-9]+)$", raw)
    if match:
        base = match.group(1)
        variant = _parse_int(match.group(2))
        if variant is None:
            return ""
        return f"{base}-{variant}"

    if re.fullmatch(r"[A-Za-z0-9._]+", raw):
        return f"{raw}-1"
    return raw


def canonicalize_minifig_item_no(value: Any) -> str:
    raw = collapse_ws(value)
    if not raw:
        return ""
    raw = raw.split("#", 1)[0]
    raw = re.sub(r"\s+", "", raw)
    return raw


def is_probable_bricklink_minifig_code(value: Any) -> bool:
    code = collapse_ws(value)
    if not code:
        return False
    lower = code.lower()
    if lower.startswith("fig-"):
        return False
    return bool(BRICKLINK_MINIFIG_CODE_RE.match(code))


def build_set_item_candidates(
    number: Any,
    variant: Any,
    link: Any = None,
    price_guide_url: Any = None,
    alias_set_code: Any = None,
) -> List[str]:
    primary = normalize_set_code(number, variant)
    candidates: List[str] = []
    seen: set[str] = set()

    def add_candidate(value: Any) -> None:
        code = collapse_ws(value)
        if not code:
            return
        key = code.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(code)

    if not primary:
        ref = parse_bricklink_item_reference(link)
        if ref and ref[0] == "SET":
            add_candidate(ref[1])
        guide_ref = parse_bricklink_item_reference(price_guide_url)
        if guide_ref and guide_ref[0] == "SET":
            add_candidate(guide_ref[1])
        return candidates

    alias_code = canonicalize_set_item_no(alias_set_code)
    if alias_code:
        add_candidate(alias_code)

    # Prefer canonical Number+Variant if alias is absent or fails later.
    add_candidate(primary)

    ref = parse_bricklink_item_reference(link)
    if ref and ref[0] == "SET":
        add_candidate(ref[1])
    guide_ref = parse_bricklink_item_reference(price_guide_url)
    if guide_ref and guide_ref[0] == "SET":
        add_candidate(guide_ref[1])

    match = re.match(r"^(.+)-([0-9]+)$", primary)
    if not match:
        return candidates

    base = match.group(1)
    var = _parse_int(match.group(2)) or 1

    # BrickLink often only exposes -1 for legacy/re-release variants.
    if var != 1:
        add_candidate(f"{base}-1")

    # Some imported set numbers contain punctuation not present in BrickLink item_no.
    compact_base = re.sub(r"[^A-Za-z0-9]", "", base)
    if compact_base and compact_base.lower() != base.lower():
        add_candidate(f"{compact_base}-{var}")
        add_candidate(f"{compact_base}-1")

    # Do not probe bare-number/base codes (e.g. "MAZKANATA") by default.
    # These frequently return HTTP 400 and waste request budget.

    return candidates


def build_minifig_item_candidates(
    number: Any,
    link: Any = None,
    price_guide_url: Any = None,
) -> List[str]:
    candidates: List[str] = []
    seen: set[str] = set()

    ref = parse_bricklink_item_reference(link)
    if ref and ref[0] == "MINIFIG":
        ref_code = collapse_ws(ref[1])
        if ref_code:
            candidates.append(ref_code)
            seen.add(ref_code.lower())
    guide_ref = parse_bricklink_item_reference(price_guide_url)
    if guide_ref and guide_ref[0] == "MINIFIG":
        ref_code = collapse_ws(guide_ref[1])
        if ref_code and ref_code.lower() not in seen:
            candidates.append(ref_code)
            seen.add(ref_code.lower())

    code = canonicalize_minifig_item_no(number)
    if is_probable_bricklink_minifig_code(code):
        lowered = code.lower()
        if lowered not in seen:
            candidates.append(code)
            seen.add(lowered)
    return candidates


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


def latest_sale_from_transactions(transactions: Any) -> Tuple[Optional[str], Optional[float]]:
    if not isinstance(transactions, list):
        return (None, None)
    best_month = ""
    best_sequence = 10**9
    best_price: Optional[float] = None
    for tx in transactions:
        if not isinstance(tx, dict):
            continue
        month = collapse_ws(tx.get("month"))
        if not re.match(r"^\d{4}-\d{2}$", month):
            continue
        price = _parse_float(tx.get("eachPrice"))
        if price is None:
            continue
        seq = _parse_int(tx.get("sequence")) or 10**9
        if month > best_month or (month == best_month and seq < best_sequence):
            best_month = month
            best_sequence = seq
            best_price = round(price, 2)
    if not best_month or best_price is None:
        return (None, None)
    return (best_month, best_price)


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
    return build_html_price_guide_url(item_type, item_no)


def apply_market_to_row(
    row: Dict[str, Any],
    *,
    item_type: str,
    item_no: str,
    currency_code: str,
    fallback_currency_codes: Sequence[str],
    allow_html_fallback: bool,
    client: BrickLinkClient,
    throttle: RuntimeThrottle,
    month_key: str,
) -> bool:
    previous_values = {key: row.get(key) for key in MARKET_PRESERVE_FIELDS}

    matrix: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None
    sold_new: Optional[Dict[str, Any]] = None
    sold_used: Optional[Dict[str, Any]] = None
    stock_new: Optional[Dict[str, Any]] = None
    stock_used: Optional[Dict[str, Any]] = None
    sold_new_avg: Optional[float] = None
    sold_used_avg: Optional[float] = None
    stock_new_avg: Optional[float] = None
    stock_used_avg: Optional[float] = None
    resolved_request_currency = normalize_currency_code(currency_code) or "USD"
    fallback_month_new: Optional[List[Dict[str, Any]]] = None
    fallback_month_used: Optional[List[Dict[str, Any]]] = None
    fallback_transactions_new: Optional[List[Dict[str, Any]]] = None
    fallback_transactions_used: Optional[List[Dict[str, Any]]] = None
    fallback_current_listings_new: Optional[List[Dict[str, Any]]] = None
    fallback_current_listings_used: Optional[List[Dict[str, Any]]] = None
    html_probe_attempted = False
    market_source = "api"

    def fetch_html_once(*, quiet_no_data: bool = False) -> Optional[
        Tuple[
            Dict[Tuple[str, str], Dict[str, Any]],
            List[Dict[str, Any]],
            List[Dict[str, Any]],
            List[Dict[str, Any]],
            List[Dict[str, Any]],
            List[Dict[str, Any]],
            List[Dict[str, Any]],
            str,
        ]
    ]:
        nonlocal html_probe_attempted
        if html_probe_attempted:
            return None
        html_probe_attempted = True
        return client.fetch_price_guide_html(
            item_type=item_type,
            item_no=item_no,
            throttle=throttle,
            quiet_no_data=quiet_no_data,
        )

    currency_try_order = build_currency_try_order(currency_code, fallback_currency_codes)
    if not currency_try_order:
        currency_try_order = [normalize_currency_code(currency_code) or "GBP"]

    if not client.auth_failed:
        for cc in currency_try_order:
            candidate = client.fetch_price_matrix(
                item_type=item_type,
                item_no=item_no,
                currency_code=cc,
                throttle=throttle,
            )
            if candidate is None:
                if client.last_error_kind == "not_found":
                    break
                if client.last_error_kind == "auth":
                    break
                continue

            c_sold_new = candidate.get(("sold", "N"))
            c_sold_used = candidate.get(("sold", "U"))
            c_stock_new = candidate.get(("stock", "N"))
            c_stock_used = candidate.get(("stock", "U"))

            c_sold_new_avg = _parse_float(c_sold_new.get("avg_price") if c_sold_new else None)
            c_sold_used_avg = _parse_float(c_sold_used.get("avg_price") if c_sold_used else None)
            c_stock_new_avg = _parse_float(c_stock_new.get("avg_price") if c_stock_new else None)
            c_stock_used_avg = _parse_float(c_stock_used.get("avg_price") if c_stock_used else None)

            any_numeric = any(
                value is not None
                for value in (
                    c_sold_new_avg,
                    c_sold_used_avg,
                    c_stock_new_avg,
                    c_stock_used_avg,
                    _parse_float(c_sold_new.get("min_price") if c_sold_new else None),
                    _parse_float(c_sold_used.get("min_price") if c_sold_used else None),
                    _parse_float(c_stock_new.get("min_price") if c_stock_new else None),
                    _parse_float(c_stock_used.get("min_price") if c_stock_used else None),
                )
            )
            if not any_numeric:
                continue

            matrix = candidate
            sold_new = c_sold_new
            sold_used = c_sold_used
            stock_new = c_stock_new
            stock_used = c_stock_used
            sold_new_avg = c_sold_new_avg
            sold_used_avg = c_sold_used_avg
            stock_new_avg = c_stock_new_avg
            stock_used_avg = c_stock_used_avg
            resolved_request_currency = cc
            break

    if matrix is None:
        if allow_html_fallback:
            fallback_result = fetch_html_once(quiet_no_data=False)
            if fallback_result is not None:
                (
                    candidate,
                    html_month_new,
                    html_month_used,
                    html_tx_new,
                    html_tx_used,
                    html_listings_new,
                    html_listings_used,
                    html_currency,
                ) = fallback_result
                c_sold_new = candidate.get(("sold", "N"))
                c_sold_used = candidate.get(("sold", "U"))
                c_stock_new = candidate.get(("stock", "N"))
                c_stock_used = candidate.get(("stock", "U"))

                c_sold_new_avg = _parse_float(c_sold_new.get("avg_price") if c_sold_new else None)
                c_sold_used_avg = _parse_float(c_sold_used.get("avg_price") if c_sold_used else None)
                c_stock_new_avg = _parse_float(c_stock_new.get("avg_price") if c_stock_new else None)
                c_stock_used_avg = _parse_float(c_stock_used.get("avg_price") if c_stock_used else None)

                any_numeric = any(
                    value is not None
                    for value in (
                        c_sold_new_avg,
                        c_sold_used_avg,
                        c_stock_new_avg,
                        c_stock_used_avg,
                        _parse_float(c_sold_new.get("min_price") if c_sold_new else None),
                        _parse_float(c_sold_used.get("min_price") if c_sold_used else None),
                        _parse_float(c_stock_new.get("min_price") if c_stock_new else None),
                        _parse_float(c_stock_used.get("min_price") if c_stock_used else None),
                    )
                )
                if any_numeric:
                    market_source = "html"
                    matrix = candidate
                    sold_new = c_sold_new
                    sold_used = c_sold_used
                    stock_new = c_stock_new
                    stock_used = c_stock_used
                    sold_new_avg = c_sold_new_avg
                    sold_used_avg = c_sold_used_avg
                    stock_new_avg = c_stock_new_avg
                    stock_used_avg = c_stock_used_avg
                    fallback_month_new = html_month_new
                    fallback_month_used = html_month_used
                    fallback_transactions_new = html_tx_new
                    fallback_transactions_used = html_tx_used
                    fallback_current_listings_new = html_listings_new
                    fallback_current_listings_used = html_listings_used
                    if collapse_ws(html_currency):
                        resolved_request_currency = collapse_ws(html_currency).upper()

    if matrix is None:
        client._mark_failure("no_data", client.last_http_status)
        return False

    if allow_html_fallback and (
        not isinstance(row.get("BrickLinkMonthlySalesNew"), list)
        or not row.get("BrickLinkMonthlySalesNew")
        or not isinstance(row.get("BrickLinkMonthlySalesUsed"), list)
        or not row.get("BrickLinkMonthlySalesUsed")
        or not isinstance(row.get("BrickLinkTransactionsNew"), list)
        or not row.get("BrickLinkTransactionsNew")
        or not isinstance(row.get("BrickLinkTransactionsUsed"), list)
        or not row.get("BrickLinkTransactionsUsed")
    ):
        history_result = fetch_html_once(quiet_no_data=True)
        if history_result is not None:
            (
                history_matrix,
                html_month_new,
                html_month_used,
                html_tx_new,
                html_tx_used,
                html_listings_new,
                html_listings_used,
                html_currency,
            ) = history_result
            h_sold_new = history_matrix.get(("sold", "N"))
            h_sold_used = history_matrix.get(("sold", "U"))
            h_stock_new = history_matrix.get(("stock", "N"))
            h_stock_used = history_matrix.get(("stock", "U"))
            history_has_numeric = any(
                value is not None
                for value in (
                    _parse_float(h_sold_new.get("avg_price") if h_sold_new else None),
                    _parse_float(h_sold_used.get("avg_price") if h_sold_used else None),
                    _parse_float(h_stock_new.get("avg_price") if h_stock_new else None),
                    _parse_float(h_stock_used.get("avg_price") if h_stock_used else None),
                )
            )
            if history_has_numeric:
                market_source = "html"
                matrix = history_matrix
                sold_new = h_sold_new
                sold_used = h_sold_used
                stock_new = h_stock_new
                stock_used = h_stock_used
                sold_new_avg = _parse_float(sold_new.get("avg_price") if sold_new else None)
                sold_used_avg = _parse_float(sold_used.get("avg_price") if sold_used else None)
                stock_new_avg = _parse_float(stock_new.get("avg_price") if stock_new else None)
                stock_used_avg = _parse_float(stock_used.get("avg_price") if stock_used else None)
                if collapse_ws(html_currency):
                    resolved_request_currency = collapse_ws(html_currency).upper()
            if html_month_new:
                fallback_month_new = html_month_new
            if html_month_used:
                fallback_month_used = html_month_used
            if html_tx_new:
                fallback_transactions_new = html_tx_new
            if html_tx_used:
                fallback_transactions_used = html_tx_used
            if html_listings_new:
                fallback_current_listings_new = html_listings_new
            if html_listings_used:
                fallback_current_listings_used = html_listings_used

    currency = collapse_ws(
        (sold_new or sold_used or stock_new or stock_used or {}).get("currency_code")
    ).upper() or resolved_request_currency

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

    # Top-level New/Used should align with "current value" semantics (sold avg),
    # while still falling back to current listing summary when needed.
    display_new = first_non_none([
        sold_new_avg,
        stock_new_avg,
        _parse_float(stock_new.get("min_price") if stock_new else None),
        _parse_float(sold_new.get("min_price") if sold_new else None),
    ])
    display_used = first_non_none([
        sold_used_avg,
        stock_used_avg,
        _parse_float(stock_used.get("min_price") if stock_used else None),
        _parse_float(sold_used.get("min_price") if sold_used else None),
    ])
    existing_new = collapse_ws(row.get("New"))
    existing_used = collapse_ws(row.get("Used"))
    row["New"] = format_display_price(display_new, currency) if display_new is not None else (existing_new or None)
    row["Used"] = format_display_price(display_used, currency) if display_used is not None else (existing_used or None)

    new_min_candidates = [_parse_float(stock_new.get("min_price") if stock_new else None)]
    new_max_candidates = [_parse_float(stock_new.get("max_price") if stock_new else None)]
    new_min = min((v for v in new_min_candidates if v is not None), default=None)
    new_max = max((v for v in new_max_candidates if v is not None), default=None)
    if new_min is None:
        new_min = _parse_float(sold_new.get("min_price") if sold_new else None)
    if new_max is None:
        new_max = _parse_float(sold_new.get("max_price") if sold_new else None)
    row["BrickLinkNewPriceRangeMin"] = round(new_min, 2) if new_min is not None else None
    row["BrickLinkNewPriceRangeMax"] = round(new_max, 2) if new_max is not None else None

    used_min_candidates = [_parse_float(stock_used.get("min_price") if stock_used else None)]
    used_max_candidates = [_parse_float(stock_used.get("max_price") if stock_used else None)]
    used_min = min((v for v in used_min_candidates if v is not None), default=None)
    used_max = max((v for v in used_max_candidates if v is not None), default=None)
    if used_min is None:
        used_min = _parse_float(sold_used.get("min_price") if sold_used else None)
    if used_max is None:
        used_max = _parse_float(sold_used.get("max_price") if sold_used else None)
    row["BrickLinkUsedPriceRangeMin"] = round(used_min, 2) if used_min is not None else None
    row["BrickLinkUsedPriceRangeMax"] = round(used_max, 2) if used_max is not None else None

    existing_current_listings_new = (
        row.get("BrickLinkCurrentListingsNew")
        if isinstance(row.get("BrickLinkCurrentListingsNew"), list)
        else []
    )
    existing_current_listings_used = (
        row.get("BrickLinkCurrentListingsUsed")
        if isinstance(row.get("BrickLinkCurrentListingsUsed"), list)
        else []
    )
    if isinstance(fallback_current_listings_new, list) and fallback_current_listings_new:
        row["BrickLinkCurrentListingsNew"] = fallback_current_listings_new
    else:
        row["BrickLinkCurrentListingsNew"] = existing_current_listings_new
    if isinstance(fallback_current_listings_used, list) and fallback_current_listings_used:
        row["BrickLinkCurrentListingsUsed"] = fallback_current_listings_used
    else:
        row["BrickLinkCurrentListingsUsed"] = existing_current_listings_used

    # Prefer true historical rows from HTML when available. Keep existing values
    # otherwise instead of synthesizing artificial month points every run.
    existing_month_new = row.get("BrickLinkMonthlySalesNew") if isinstance(row.get("BrickLinkMonthlySalesNew"), list) else []
    existing_month_used = row.get("BrickLinkMonthlySalesUsed") if isinstance(row.get("BrickLinkMonthlySalesUsed"), list) else []
    existing_tx_new = row.get("BrickLinkTransactionsNew") if isinstance(row.get("BrickLinkTransactionsNew"), list) else []
    existing_tx_used = row.get("BrickLinkTransactionsUsed") if isinstance(row.get("BrickLinkTransactionsUsed"), list) else []
    if fallback_month_new is not None:
        month_new = fallback_month_new
    else:
        month_new = existing_month_new
    if fallback_month_used is not None:
        month_used = fallback_month_used
    else:
        month_used = existing_month_used
    row["BrickLinkMonthlySalesNew"] = month_new
    row["BrickLinkMonthlySalesUsed"] = month_used

    if isinstance(fallback_transactions_new, list) and fallback_transactions_new:
        tx_new = fallback_transactions_new
    elif existing_tx_new:
        tx_new = existing_tx_new
    else:
        tx_new = monthly_series_to_transactions(month_new, currency)

    if isinstance(fallback_transactions_used, list) and fallback_transactions_used:
        tx_used = fallback_transactions_used
    elif existing_tx_used:
        tx_used = existing_tx_used
    else:
        tx_used = monthly_series_to_transactions(month_used, currency)
    row["BrickLinkTransactionsNew"] = tx_new
    row["BrickLinkTransactionsUsed"] = tx_used
    row["BrickLinkTransactionsNewCount"] = len(tx_new)
    row["BrickLinkTransactionsUsedCount"] = len(tx_used)

    latest_new_month, latest_new_price = latest_sale_from_transactions(tx_new)
    latest_used_month, latest_used_price = latest_sale_from_transactions(tx_used)
    row["BrickLinkLatestSaleNewMonth"] = latest_new_month or (month_key if sold_new_avg is not None else None)
    row["BrickLinkLatestSaleNewPrice"] = latest_new_price if latest_new_price is not None else (round(sold_new_avg, 2) if sold_new_avg is not None else None)
    row["BrickLinkLatestSaleUsedMonth"] = latest_used_month or (month_key if sold_used_avg is not None else None)
    row["BrickLinkLatestSaleUsedPrice"] = latest_used_price if latest_used_price is not None else (round(sold_used_avg, 2) if sold_used_avg is not None else None)

    # RRP and forecast helpers.
    rrp, rrp_currency = resolve_rrp_from_row(row)
    row["RRP"] = round(rrp, 2) if rrp is not None else None
    row["RRPCurrency"] = rrp_currency
    current_new_for_compare = first_non_none([stock_new_avg, sold_new_avg])
    if (
        rrp is not None
        and rrp > 0
        and current_new_for_compare is not None
        and collapse_ws(rrp_currency).upper() == collapse_ws(currency).upper()
    ):
        delta = current_new_for_compare - rrp
        row["CurrentNewVsRRPAmount"] = round(delta, 2)
        row["CurrentNewVsRRPPercent"] = round((delta / rrp) * 100.0, 2)
        row["CurrentRRPBaseline"] = round(rrp, 2)
    else:
        row["CurrentNewVsRRPAmount"] = None
        row["CurrentNewVsRRPPercent"] = None
        row["CurrentRRPBaseline"] = round(rrp, 2) if rrp is not None else None
    row["CurrentRRPBaselineCurrency"] = rrp_currency

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
    row["MarketFetchStatus"] = f"ok_{market_source}"
    row["MarketNoDataRetryAfterUTC"] = None

    # Preserve prior analytics when the current API response does not include that facet.
    for key, previous in previous_values.items():
        current = row.get(key)
        current_empty = (
            current is None
            or (isinstance(current, str) and collapse_ws(current) == "")
            or (isinstance(current, (list, dict)) and len(current) == 0)
        )
        previous_has_value = not (
            previous is None
            or (isinstance(previous, str) and collapse_ws(previous) == "")
            or (isinstance(previous, (list, dict)) and len(previous) == 0)
        )
        if current_empty and previous_has_value:
            row[key] = previous

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
    # Keep the large catalog payloads compact to stay under GitHub's 100 MB
    # object hard limit while preserving full data fidelity.
    compact_targets = {
        "Lego Star Wars Database.json",
        "Lego-Star-Wars-Minifigure-Database.json",
    }
    if path.name in compact_targets:
        updated = json.dumps(rows, ensure_ascii=False, separators=(",", ":")) + "\n"
    else:
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


def rotating_indices(total: int, start_index: int) -> List[int]:
    if total <= 0:
        return []
    start = start_index % total
    ordered: List[int] = []
    for idx in range(start, total):
        ordered.append(idx)
    for idx in range(0, start):
        ordered.append(idx)
    return ordered


def prioritize_rotating_indices(rotating: Sequence[int], priority: set[int]) -> List[int]:
    if not rotating:
        return []
    prioritized: List[int] = []
    non_prioritized: List[int] = []
    for idx in rotating:
        if idx in priority:
            prioritized.append(idx)
        else:
            non_prioritized.append(idx)
    return prioritized + non_prioritized


def build_priority_plan(
    rotating: Sequence[int],
    primary_priority: set[int],
    secondary_priority: set[int],
) -> Tuple[List[int], List[int]]:
    if not rotating:
        return ([], [])
    primary: List[int] = []
    secondary: List[int] = []
    non_priority: List[int] = []
    for idx in rotating:
        if idx in primary_priority:
            primary.append(idx)
        elif idx in secondary_priority:
            secondary.append(idx)
        else:
            non_priority.append(idx)
    return (primary + secondary + non_priority, non_priority)


def advance_rotation_cursor(
    *,
    total_rows: int,
    current_cursor: int,
    processed_indices: Sequence[int],
    non_priority_order: Sequence[int],
    rotating_order: Sequence[int],
) -> int:
    if total_rows <= 0:
        return 0
    if not processed_indices:
        return current_cursor % total_rows
    if not non_priority_order:
        if not rotating_order:
            return current_cursor % total_rows
        rotating_positions = {row_index: pos for pos, row_index in enumerate(rotating_order)}
        processed_positions = [
            rotating_positions[row_index]
            for row_index in processed_indices
            if row_index in rotating_positions
        ]
        if not processed_positions:
            return current_cursor % total_rows
        last_rotating_idx = rotating_order[max(processed_positions)]
        return (last_rotating_idx + 1) % total_rows

    non_priority_positions = {row_index: pos for pos, row_index in enumerate(non_priority_order)}
    processed_positions = [
        non_priority_positions[row_index]
        for row_index in processed_indices
        if row_index in non_priority_positions
    ]
    if not processed_positions:
        if not rotating_order:
            return current_cursor % total_rows
        rotating_positions = {row_index: pos for pos, row_index in enumerate(rotating_order)}
        processed_positions = [
            rotating_positions[row_index]
            for row_index in processed_indices
            if row_index in rotating_positions
        ]
        if not processed_positions:
            return current_cursor % total_rows
        last_rotating_idx = rotating_order[max(processed_positions)]
        return (last_rotating_idx + 1) % total_rows
    last_non_priority_idx = non_priority_order[max(processed_positions)]
    return (last_non_priority_idx + 1) % total_rows


def print_summary(label: str, stats: FileUpdateStats) -> None:
    print(
        (
            f"[{label}] total={stats.total_rows} "
            f"considered={stats.rows_considered} "
            f"succeeded={stats.rows_succeeded} "
            f"changed={stats.rows_changed} "
            f"fetch_failures={stats.fetch_failures} "
            f"no_price_data_skips={stats.no_price_data_skips} "
            f"cooldown_skips={stats.cooldown_skips} "
            f"parse_misses={stats.parse_misses} "
            f"cross_changed={stats.cross_rows_changed}"
        ),
        flush=True,
    )


def merge_update_stats(base: FileUpdateStats, add: FileUpdateStats) -> FileUpdateStats:
    base.rows_considered += add.rows_considered
    base.rows_succeeded += add.rows_succeeded
    base.rows_changed += add.rows_changed
    base.fetch_failures += add.fetch_failures
    base.no_price_data_skips += add.no_price_data_skips
    base.cooldown_skips += add.cooldown_skips
    base.parse_misses += add.parse_misses
    base.cross_rows_changed += add.cross_rows_changed
    base.processed_indices.extend(add.processed_indices)
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
    run_started_at: datetime,
    no_data_cooldown_hours: float,
    set_alias_cache: Optional[Dict[str, str]] = None,
    alias_lookup_budget: Optional[List[int]] = None,
) -> FileUpdateStats:
    stats = FileUpdateStats(total_rows=len(rows))
    if indexes is not None:
        iter_indices = [i for i in indexes if 0 <= i < len(rows)]
    else:
        iter_indices = list(range(len(rows)))

    for idx in iter_indices:
        if client.auth_failed and not cfg.allow_html_fallback:
            break
        if client.budget_exhausted:
            break
        if indexes is None and idx < start_index:
            continue
        if limit is not None and stats.rows_considered >= limit:
            break

        row = rows[idx]
        retry_at = parse_status_retry_at(row)
        if retry_at is not None and run_started_at < retry_at:
            stats.cooldown_skips += 1
            continue

        stats.rows_considered += 1
        stats.last_index_processed = idx
        stats.processed_indices.append(idx)

        canonical_set_code = ""
        if item_type == "SET":
            canonical_set_code = normalize_set_code(row.get("Number"), row.get("Variant")).lower()
            cached_alias = None
            if set_alias_cache is not None and canonical_set_code:
                cached_alias = set_alias_cache.get(canonical_set_code)
            set_candidates = build_set_item_candidates(
                row.get("Number"),
                row.get("Variant"),
                row.get("link"),
                row.get("BrickLinkPriceGuideURL"),
                alias_set_code=cached_alias,
            )
            item_candidates: List[Tuple[str, str]] = [("SET", value) for value in set_candidates]
        else:
            minifig_candidates = build_minifig_item_candidates(
                row.get("Number"),
                row.get("link"),
                row.get("BrickLinkPriceGuideURL"),
            )
            item_candidates = [("MINIFIG", value) for value in minifig_candidates]

        if not item_candidates:
            stats.parse_misses += 1
            continue

        item_no = item_candidates[0][1]
        before = json.dumps(row, sort_keys=True, ensure_ascii=False)
        ok = False
        used_item_no = item_no
        used_item_type = item_type
        for candidate_type, candidate_no in item_candidates:
            used_item_type = candidate_type
            used_item_no = candidate_no
            ok = apply_market_to_row(
                row,
                item_type=candidate_type,
                item_no=candidate_no,
                currency_code=cfg.currency_code,
                fallback_currency_codes=cfg.fallback_currency_codes,
                allow_html_fallback=cfg.allow_html_fallback,
                client=client,
                throttle=throttle,
                month_key=month_key,
            )
            if ok:
                break
            if (client.auth_failed and not cfg.allow_html_fallback) or client.budget_exhausted:
                break

        if (
            not ok
            and item_type == "SET"
            and set_alias_cache is not None
            and canonical_set_code
            and canonical_set_code not in set_alias_cache
            and client.last_error_kind in {"not_found", "no_data", "http_error"}
        ):
            remaining_alias_lookups = alias_lookup_budget[0] if isinstance(alias_lookup_budget, list) and alias_lookup_budget else 0
            if remaining_alias_lookups > 0:
                alias_lookup_budget[0] = max(0, remaining_alias_lookups - 1)
                discovered_alias = client.fetch_set_alias_from_brickset(row.get("link"), throttle)
                discovered_alias = canonicalize_set_item_no(discovered_alias)
                if discovered_alias and discovered_alias.lower() != used_item_no.lower():
                    set_alias_cache[canonical_set_code] = discovered_alias
                    ok = apply_market_to_row(
                        row,
                        item_type="SET",
                        item_no=discovered_alias,
                        currency_code=cfg.currency_code,
                        fallback_currency_codes=cfg.fallback_currency_codes,
                        allow_html_fallback=cfg.allow_html_fallback,
                        client=client,
                        throttle=throttle,
                        month_key=month_key,
                    )
                    if ok:
                        used_item_no = discovered_alias
                        used_item_type = "SET"

        if not ok:
            status_value = infer_fetch_status(client.last_error_kind, client.last_http_status)
            row["MarketFetchStatus"] = status_value
            row["MarketLastUpdatedUTC"] = run_started_at.strftime("%Y-%m-%dT%H:%M:%SZ")
            if status_value in {"not_found", "no_data", "http_400", "api_error"}:
                retry_after = run_started_at.timestamp() + max(1.0, no_data_cooldown_hours) * 3600.0
                row["MarketNoDataRetryAfterUTC"] = datetime.fromtimestamp(retry_after, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                row["MarketNoDataRetryAfterUTC"] = None
            after_failed = json.dumps(row, sort_keys=True, ensure_ascii=False)
            if before != after_failed:
                stats.rows_changed += 1

            if client.last_error_kind in {"not_found", "no_data"}:
                stats.no_price_data_skips += 1
                if cfg.verbose:
                    print(f"[{label}] no price data: {item_no}", flush=True)
            else:
                stats.fetch_failures += 1
                if cfg.verbose:
                    print(f"[{label}] failed: {item_no}", flush=True)
            if client.auth_failed and not cfg.allow_html_fallback:
                break
            if client.budget_exhausted:
                break
            continue

        stats.rows_succeeded += 1
        row["MarketNoDataRetryAfterUTC"] = None
        after = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if before != after:
            stats.rows_changed += 1

        if cfg.verbose:
            resolved = used_item_no if used_item_no != item_no else item_no
            print(
                f"[{label}] {stats.rows_considered}/{stats.total_rows}: {item_no} -> {resolved} ({used_item_type}) updated",
                flush=True,
            )

    return stats


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update LEGO market values via BrickLink API.")
    parser.add_argument("--sets-json", default="dist/Lego Star Wars Database.json", help="Path to sets JSON file.")
    parser.add_argument("--minifigs-json", default="dist/Lego-Star-Wars-Minifigure-Database.json", help="Path to minifig JSON file.")

    parser.add_argument("--bricklink-base-url", default=BRICKLINK_API_BASE_URL, help="BrickLink API base URL.")
    parser.add_argument("--currency-code", default=os.getenv("BRICKLINK_CURRENCY", "GBP"), help="Price currency code (default GBP).")
    parser.add_argument(
        "--fallback-currencies",
        default=os.getenv("BRICKLINK_FALLBACK_CURRENCIES", "USD,EUR"),
        help="Comma-separated fallback currencies to try when primary currency has no rows (default USD,EUR).",
    )

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
        "--set-aliases-json",
        default="dist/bricklink-set-aliases.json",
        help="Optional JSON map of canonical set numbers to BrickLink item_no aliases.",
    )
    parser.add_argument(
        "--priority-updated-limit",
        type=int,
        default=1200,
        help="Maximum recently changed set numbers to prioritize from catalog sync state.",
    )
    parser.add_argument(
        "--priority-themes",
        default=os.getenv("MARKET_PRIORITY_THEMES", "Star Wars,Marvel Super Heroes,Disney,NINJAGO"),
        help="Comma-separated set themes to prioritize before full rotation.",
    )
    parser.add_argument(
        "--priority-minifig-categories",
        default=os.getenv("MARKET_PRIORITY_MINIFIG_CATEGORIES", "Star Wars,Marvel Super Heroes,Disney,NINJAGO"),
        help="Comma-separated minifig categories/themes to prioritize before full rotation.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional per-file row limit for testing.")
    parser.add_argument(
        "--item-type",
        choices=["both", "set", "minifig"],
        default="both",
        help="Choose whether to update sets, minifigs, or both (default both).",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Optional per-file start index.")
    parser.add_argument("--skip-cross-enrichment", action="store_true", help="Skip exclusivity/appears-in enrichment.")
    parser.add_argument(
        "--no-data-cooldown-hours",
        type=float,
        default=float(os.getenv("MARKET_NO_DATA_COOLDOWN_HOURS", "72")),
        help="Hours to defer retrying rows that returned hard no-data/not-found failures.",
    )
    parser.add_argument(
        "--max-alias-lookups-per-run",
        type=int,
        default=int(os.getenv("MARKET_MAX_ALIAS_LOOKUPS_PER_RUN", "80")),
        help="Maximum Brickset page alias lookups per run for sets with no direct BrickLink item number.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--disable-html-fallback",
        action="store_true",
        help="Disable BrickLink catalogPG HTML fallback when API values are missing or auth fails.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    allow_html_fallback = not bool(args.disable_html_fallback)

    raw_required = {
        "BRICKLINK_CONSUMER_KEY": collapse_ws(args.consumer_key),
        "BRICKLINK_CONSUMER_SECRET": collapse_ws(args.consumer_secret),
        "BRICKLINK_TOKEN_VALUE": collapse_ws(args.token_value),
        "BRICKLINK_TOKEN_SECRET": collapse_ws(args.token_secret),
    }
    required = {key: sanitize_secret(value) for key, value in raw_required.items()}
    missing = [key for key, value in required.items() if not value]
    if missing:
        if allow_html_fallback:
            print(
                "[API] missing credentials; running in HTML fallback mode only: "
                + ", ".join(missing),
                flush=True,
            )
            for key in missing:
                required[key] = "MISSING"
        else:
            print("Missing BrickLink credentials: " + ", ".join(missing), file=sys.stderr)
            print(bricklink_auth_help_hint(), file=sys.stderr)
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
        fallback_currency_codes=tuple(
            code
            for code in (
                normalize_currency_code(part)
                for part in re.split(r"[,\s]+", str(args.fallback_currencies or ""))
            )
            if code
        ),
        allow_html_fallback=allow_html_fallback,
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
    if client.auth_failed and not cfg.allow_html_fallback:
        msg = client.auth_error_message or "BrickLink API authentication failed."
        msg = f"{msg} {bricklink_auth_help_hint()}"
        print(msg, file=sys.stderr)
        if cfg.verbose:
            diag = ", ".join(
                f"{k}:len={len(v)}{('*sanitized*' if raw_required.get(k) != v else '')}"
                for k, v in required.items()
            )
            print(f"[AuthDiag] {diag}", file=sys.stderr)
        return 1
    if client.auth_failed and cfg.allow_html_fallback and cfg.verbose:
        msg = client.auth_error_message or "BrickLink API authentication failed."
        print(f"[API] warning: {msg} Falling back to BrickLink catalogPG HTML parsing.", flush=True)

    sets_rows = load_json_array(sets_path)
    minifigs_rows = load_json_array(minifigs_path)
    market_state_path = Path(args.market_state_json)
    market_state = load_json_object(market_state_path)
    catalog_state = load_json_object(Path(args.catalog_sync_state_json))
    static_aliases = load_json_object(Path(args.set_aliases_json))
    raw_alias_cache = market_state.get("setAliasByNumber")
    set_alias_cache: Dict[str, str] = {}
    if isinstance(raw_alias_cache, dict):
        for k, v in raw_alias_cache.items():
            key = normalize_set_code(k, 1).lower() if collapse_ws(k) else ""
            val = canonicalize_set_item_no(v)
            if key and val:
                set_alias_cache[key] = val
    if isinstance(static_aliases, dict):
        for k, v in static_aliases.items():
            key = normalize_set_code(k, 1).lower() if collapse_ws(k) else ""
            val = canonicalize_set_item_no(v)
            if key and val:
                set_alias_cache[key] = val
    alias_lookup_budget = [max(0, args.max_alias_lookups_per_run)]

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

    priority_themes = {
        collapse_ws(value).casefold()
        for value in re.split(r"[,;]", str(args.priority_themes or ""))
        if collapse_ws(value)
    }
    priority_minifig_categories = {
        collapse_ws(value).casefold()
        for value in re.split(r"[,;]", str(args.priority_minifig_categories or ""))
        if collapse_ws(value)
    }

    do_sets = args.item_type in {"both", "set"}
    do_minifigs = args.item_type in {"both", "minifig"}

    set_theme_priority_indices: List[int] = []
    set_changed_priority_indices: List[int] = []
    if do_sets:
        for idx, row in enumerate(sets_rows):
            code = normalize_set_code(row.get("Number"), row.get("Variant")).lower()
            theme = collapse_ws(row.get("Theme")).casefold()
            if theme in priority_themes:
                set_theme_priority_indices.append(idx)
            elif code in changed_set_lookup:
                set_changed_priority_indices.append(idx)

    minifig_theme_priority_indices: List[int] = []
    if do_minifigs:
        for idx, row in enumerate(minifigs_rows):
            category = collapse_ws(row.get("Category") or row.get("Theme")).casefold()
            if category in priority_minifig_categories:
                minifig_theme_priority_indices.append(idx)

    stored_set_cursor = _parse_int(market_state.get("nextSetIndex"))
    stored_minifig_cursor = _parse_int(market_state.get("nextMinifigIndex"))
    set_cursor = stored_set_cursor if stored_set_cursor is not None else max(0, args.start_index)
    minifig_cursor = stored_minifig_cursor if stored_minifig_cursor is not None else max(0, args.start_index)

    set_theme_priority_lookup = set(set_theme_priority_indices)
    set_changed_priority_lookup = set(set_changed_priority_indices)
    minifig_theme_priority_lookup = set(minifig_theme_priority_indices)
    set_rotating_indices = rotating_indices(len(sets_rows), set_cursor) if do_sets else []
    minifig_rotating_indices = rotating_indices(len(minifigs_rows), minifig_cursor) if do_minifigs else []
    set_plan, set_non_priority_rotation = build_priority_plan(
        set_rotating_indices,
        set_theme_priority_lookup,
        set_changed_priority_lookup,
    )
    minifig_plan, minifig_non_priority_rotation = build_priority_plan(
        minifig_rotating_indices,
        minifig_theme_priority_lookup,
        set(),
    )

    if cfg.verbose:
        print(
            (
                f"[Plan] set_priority={len(set_theme_priority_indices)} set_rotating={len(set_rotating_indices)} "
                f"minifig_priority={len(minifig_theme_priority_indices)} minifig_rotating={len(minifig_rotating_indices)} "
                f"set_changed_priority={len(set_changed_priority_indices)}"
            ),
            flush=True,
        )

    if do_sets:
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
            run_started_at=now,
            no_data_cooldown_hours=max(1.0, args.no_data_cooldown_hours),
            set_alias_cache=set_alias_cache,
            alias_lookup_budget=alias_lookup_budget,
        )
    else:
        sets_stats = FileUpdateStats(total_rows=len(sets_rows))

    if do_minifigs:
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
            run_started_at=now,
            no_data_cooldown_hours=max(1.0, args.no_data_cooldown_hours),
            set_alias_cache=None,
            alias_lookup_budget=None,
        )
    else:
        minifigs_stats = FileUpdateStats(total_rows=len(minifigs_rows))

    next_set_cursor = set_cursor
    if do_sets and sets_rows:
        next_set_cursor = advance_rotation_cursor(
            total_rows=len(sets_rows),
            current_cursor=set_cursor,
            processed_indices=sets_stats.processed_indices,
            non_priority_order=set_non_priority_rotation,
            rotating_order=set_rotating_indices,
        )
    next_minifig_cursor = minifig_cursor
    if do_minifigs and minifigs_rows:
        next_minifig_cursor = advance_rotation_cursor(
            total_rows=len(minifigs_rows),
            current_cursor=minifig_cursor,
            processed_indices=minifigs_stats.processed_indices,
            non_priority_order=minifig_non_priority_rotation,
            rotating_order=minifig_rotating_indices,
        )

    set_rows_with_new = sum(1 for row in sets_rows if bool(collapse_ws(row.get("New"))))
    set_rows_with_used = sum(1 for row in sets_rows if bool(collapse_ws(row.get("Used"))))
    minifig_rows_with_new = sum(1 for row in minifigs_rows if bool(collapse_ws(row.get("New"))))
    minifig_rows_with_used = sum(1 for row in minifigs_rows if bool(collapse_ws(row.get("Used"))))

    if client.auth_failed and not cfg.allow_html_fallback:
        msg = client.auth_error_message or "BrickLink API authentication failed."
        msg = f"{msg} {bricklink_auth_help_hint()}"
        print(msg, file=sys.stderr)
        if cfg.verbose:
            diag = ", ".join(
                f"{k}:len={len(v)}{('*sanitized*' if raw_required.get(k) != v else '')}"
                for k, v in required.items()
            )
            print(f"[AuthDiag] {diag}", file=sys.stderr)
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
            "lastSetRowsSucceeded": sets_stats.rows_succeeded,
            "lastSetCooldownSkips": sets_stats.cooldown_skips,
            "lastMinifigRowsConsidered": minifigs_stats.rows_considered,
            "lastMinifigRowsSucceeded": minifigs_stats.rows_succeeded,
            "lastMinifigCooldownSkips": minifigs_stats.cooldown_skips,
            "lastSetPriorityCount": len(set_theme_priority_indices),
            "lastMinifigPriorityCount": len(minifig_theme_priority_indices),
            "lastChangedSetPriorityCount": len(changed_set_codes),
            "lastNoDataCooldownHours": max(1.0, args.no_data_cooldown_hours),
            "lastAliasLookupsRemaining": alias_lookup_budget[0],
            "setAliasByNumber": dict(sorted(set_alias_cache.items())),
            "setRowsWithNew": set_rows_with_new,
            "setRowsWithUsed": set_rows_with_used,
            "setRowsTotal": len(sets_rows),
            "minifigRowsWithNew": minifig_rows_with_new,
            "minifigRowsWithUsed": minifig_rows_with_used,
            "minifigRowsTotal": len(minifigs_rows),
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
