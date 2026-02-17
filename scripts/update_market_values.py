#!/usr/bin/env python3
"""Update LEGO market values and BrickLink analytics in JSON catalog files.

This script now does more than `New`/`Used`:
- Scrapes BrickLink Price Guide pages (catalogPG.asp) for each row.
- Updates current display values (`New`, `Used`) from 6-month sold averages.
- Stores detailed analytics (6-month sold stats, current listing stats, monthly sales series,
  latest sale snapshots, RRP comparison, and simple 2Y/5Y forecast fields).
- Adds cross-catalog exclusivity mappings between sets and minifigures.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import ssl
import sys
import time
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import Request, urlopen

PRICE_TOKEN_PATTERN = (
    r"(?:[~≈]?\s*(?:£|\$|€)\s*[0-9][0-9,]*(?:\.[0-9]{1,2})?"
    r"|(?:USD|GBP|EUR)\s*[0-9][0-9,]*(?:\.[0-9]{1,2})"
    r"|[0-9][0-9,]*(?:\.[0-9]{1,2}))"
)
PRICE_TOKEN_RE = re.compile(PRICE_TOKEN_PATTERN, re.IGNORECASE)
PRICE_VALUE_RE = re.compile(r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)")

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
MONTH_NAME_PATTERN = "|".join(MONTH_NAMES)
MONTH_INDEX_BY_NAME = {name.lower(): idx + 1 for idx, name in enumerate(MONTH_NAMES)}
MONTH_HEADER_RE = re.compile(rf"^(?P<month>{MONTH_NAME_PATTERN})\s+(?P<year>\d{{4}})$", re.IGNORECASE)

SOLD_BLOCK_RE = re.compile(
    rf"Times Sold:\s*(?P<times>[0-9,]+)\s*"
    rf"Total Qty:\s*(?P<qty>[0-9,]+)\s*"
    rf"Min Price:\s*(?P<min>.*?)\s*"
    rf"Avg Price:\s*(?P<avg>.*?)\s*"
    rf"Qty Avg Price:\s*(?P<qtyavg>.*?)\s*"
    rf"Max Price:\s*(?P<max>.*?)(?=\s*(?:Times Sold:|Total Lots:|Currently Available|(?:{MONTH_NAME_PATTERN})\s+[0-9]{{4}}|$))",
    re.IGNORECASE | re.DOTALL,
)

LOTS_BLOCK_RE = re.compile(
    rf"Total Lots:\s*(?P<lots>[0-9,]+)\s*"
    rf"Total Qty:\s*(?P<qty>[0-9,]+)\s*"
    rf"Min Price:\s*(?P<min>.*?)\s*"
    rf"Avg Price:\s*(?P<avg>.*?)\s*"
    rf"Qty Avg Price:\s*(?P<qtyavg>.*?)\s*"
    rf"Max Price:\s*(?P<max>.*?)(?=\s*(?:Total Lots:|Currently Available|(?:{MONTH_NAME_PATTERN})\s+[0-9]{{4}}|$))",
    re.IGNORECASE | re.DOTALL,
)

TRANSACTION_LINE_RE = re.compile(
    r"^(?P<qty>[0-9][0-9,]*)\s+"
    r"(?P<price>[~≈]?\s*(?:GBP|USD|EUR|[£$€])?\s*[0-9][0-9,]*(?:\.[0-9]{1,2})?)$",
    re.IGNORECASE,
)


@dataclass
class FetchConfig:
    timeout: float
    retries: int
    delay: float
    jitter: float
    verbose: bool
    insecure: bool
    link_fallback: bool


@dataclass
class FileUpdateStats:
    total_rows: int = 0
    rows_considered: int = 0
    rows_with_link: int = 0
    rows_with_price_guide: int = 0
    rows_changed: int = 0
    fetch_failures: int = 0
    parse_misses: int = 0
    cross_rows_changed: int = 0


@dataclass
class RuntimeThrottle:
    min_delay: float
    jitter: float
    max_delay: float = 8.0
    cooldown_factor: float = 0.96
    backoff_factor: float = 1.55
    current_delay: float = 0.0

    def __post_init__(self) -> None:
        self.min_delay = max(0.0, self.min_delay)
        self.jitter = max(0.0, self.jitter)
        self.current_delay = self.min_delay

    def sleep_between_requests(self) -> None:
        delay = max(0.0, self.current_delay)
        if self.jitter > 0:
            delay += random.uniform(0.0, self.jitter)
        if delay > 0:
            time.sleep(delay)

    def apply_success(self) -> None:
        self.current_delay = max(self.min_delay, self.current_delay * self.cooldown_factor)

    def apply_rate_limit(self, retry_after: Optional[float] = None) -> None:
        backoff = max(self.current_delay * self.backoff_factor, self.min_delay * 1.5)
        if retry_after is not None:
            backoff = max(backoff, retry_after)
        self.current_delay = min(self.max_delay, backoff)


def log(message: str, *, enabled: bool = True) -> None:
    if enabled:
        print(message, flush=True)


def collapse_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def decode_html_entities(value: str) -> str:
    text = unescape(value)
    replacements = {
        "\u00a0": " ",
        "&#163;": "£",
        "&#xA3;": "£",
        "&#36;": "$",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def plain_text_from_html(html: str) -> str:
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = decode_html_entities(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lines_from_html(html: str) -> List[str]:
    text = decode_html_entities(html)
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", "\n", text)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", "\n", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(tr|p|div|li|dd|dt|h[1-6]|table|tbody|thead|tfoot|ul|ol)>", "\n", text)
    text = re.sub(r"(?i)<(tr|p|div|li|dd|dt|h[1-6]|table|tbody|thead|tfoot|ul|ol)[^>]*>", "\n", text)
    text = re.sub(r"(?i)</td>", "\t", text)
    text = re.sub(r"(?i)<td[^>]*>", "", text)
    text = re.sub(r"<[^>]+>", " ", text)
    raw_lines = [collapse_ws(line) for line in text.splitlines()]
    return [line for line in raw_lines if line]


def parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = collapse_ws(str(value))
    if not text:
        return None
    match = re.search(r"-?[0-9][0-9,]*", text)
    if not match:
        return None
    try:
        return int(match.group(0).replace(",", ""))
    except ValueError:
        return None


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    text = collapse_ws(str(value))
    if not text:
        return None
    match = re.search(r"-?[0-9][0-9,]*(?:\.[0-9]+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def detect_currency_code(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    text = decode_html_entities(raw)
    upper = text.upper()
    if "GBP" in upper or "£" in text:
        return "GBP"
    if "USD" in upper or "$" in text:
        return "USD"
    if "EUR" in upper or "€" in text:
        return "EUR"
    return None


def parse_price_token(raw: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
    if not raw:
        return (None, None)
    text = collapse_ws(decode_html_entities(str(raw)))
    if not text:
        return (None, None)

    match = PRICE_VALUE_RE.search(text)
    if not match:
        return (None, detect_currency_code(text))

    try:
        amount = float(match.group(1).replace(",", ""))
    except ValueError:
        amount = None

    return (amount, detect_currency_code(text))


def currency_symbol_for(code: Optional[str]) -> str:
    if code == "GBP":
        return "£"
    if code == "USD":
        return "$"
    if code == "EUR":
        return "€"
    return ""


def format_amount(amount: float) -> str:
    return f"{amount:,.2f}".rstrip("0").rstrip(".")


def format_display_price(amount: Optional[float], currency: Optional[str], *, approximate: bool = True) -> Optional[str]:
    if amount is None:
        return None
    symbol = currency_symbol_for(currency)
    prefix = "~" if approximate else ""
    if symbol:
        return f"{prefix}{symbol}{format_amount(amount)}"
    return f"{prefix}{format_amount(amount)}"


def normalized_price_text(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    compact = re.sub(r"\s+", " ", value).strip()
    if not compact:
        return None

    approximation_prefix = ""
    if compact.startswith("~") or compact.startswith("≈"):
        approximation_prefix = compact[0]
        compact = compact[1:].strip()

    compact = re.sub(r"(?i)^USD\s*", "$", compact)
    compact = re.sub(r"(?i)^GBP\s*", "£", compact)
    compact = re.sub(r"(?i)^EUR\s*", "€", compact)
    compact = re.sub(r"([£$€])\s+([0-9])", r"\1\2", compact)

    if not compact:
        return None

    return f"{approximation_prefix}{compact}"


def extract_price_candidates(html_fragment: str) -> List[str]:
    decoded = decode_html_entities(html_fragment)
    visible_text = re.sub(r"<[^>]+>", " ", decoded)
    return [m.group(0) for m in PRICE_TOKEN_RE.finditer(visible_text)]


def parse_new_used_segments(html_fragment: str) -> Optional[Tuple[Optional[str], Optional[str]]]:
    lower_fragment = html_fragment.lower()
    new_match = re.search(r"new\s*:", lower_fragment)
    if not new_match:
        return None

    new_start = new_match.end()
    used_match = re.search(r"used\s*:", lower_fragment[new_start:])

    if used_match:
        used_abs_start = new_start + used_match.start()
        used_abs_end = new_start + used_match.end()
        new_segment = html_fragment[new_start:used_abs_start]
        used_segment = html_fragment[used_abs_end : used_abs_end + 1200]
    else:
        new_segment = html_fragment[new_start : new_start + 2400]
        used_segment = ""

    new_candidates = extract_price_candidates(new_segment)
    used_candidates = extract_price_candidates(used_segment)

    if not used_candidates and len(new_candidates) >= 2:
        used_candidates = [new_candidates[1]]

    new_price = normalized_price_text(new_candidates[0] if new_candidates else None)
    used_price = normalized_price_text(used_candidates[0] if used_candidates else None)

    if not new_price and not used_price:
        return None

    return (new_price, used_price)


def parse_current_value_market_prices(html: str) -> Optional[Tuple[Optional[str], Optional[str]]]:
    pattern = re.compile(
        r"(?is)<dt[^>]*>\s*current\s*value\s*</dt>\s*<dd[^>]*>(.*?)</dd>"
    )

    for match in pattern.finditer(html):
        dd_block = match.group(1)
        parsed = parse_new_used_segments(dd_block)
        if parsed:
            return parsed

        candidates = extract_price_candidates(dd_block)
        if candidates:
            new_price = normalized_price_text(candidates[0])
            used_price = normalized_price_text(candidates[1]) if len(candidates) > 1 else None
            if new_price or used_price:
                return (new_price, used_price)

    return None


def parse_structured_market_prices(html: str) -> Optional[Tuple[Optional[str], Optional[str]]]:
    pattern = re.compile(r"(?is)<dd[^>]*>\s*New:\s*(.*?)\s*Used:\s*(.*?)</dd>")

    best: Optional[Tuple[int, Optional[str], Optional[str]]] = None

    for match in pattern.finditer(html):
        block = match.group(0)
        if "#T=P" not in block and "bricklink" not in block.lower():
            continue

        new_segment = match.group(1)
        used_segment = match.group(2)

        new_candidates = extract_price_candidates(new_segment)
        used_candidates = extract_price_candidates(used_segment)
        inferred_used_from_second_new = False

        if not used_candidates and len(new_candidates) >= 2:
            used_candidates = [new_candidates[1]]
            inferred_used_from_second_new = True

        new_price = normalized_price_text(new_candidates[0] if new_candidates else None)
        used_price = normalized_price_text(used_candidates[0] if used_candidates else None)

        if not new_price and not used_price:
            continue

        context_start = max(0, match.start() - 240)
        context_end = min(len(html), match.end() + 480)
        context = html[context_start:context_end].lower()
        block_lower = block.lower()

        score = 0
        if "current value" in context:
            score += 120
        if "price guide" in context:
            score += 30
        if inferred_used_from_second_new:
            score += 35
        if new_price:
            score += 10
        if used_price:
            score += 10
        if "sold" in context:
            score -= 20
        if "6 month" in context or "6m" in context:
            score -= 15
        if "avg" in block_lower or "average" in block_lower:
            score -= 15

        candidate = (score, new_price, used_price)
        if best is None or score > best[0]:
            best = candidate

    if best is None:
        return None

    return (best[1], best[2])


def first_capture(text: str, pattern: str) -> Optional[str]:
    match = re.search(pattern, text)
    if not match:
        return None
    return match.group(1)


def extract_price(label: str, text: str) -> Optional[str]:
    escaped_label = re.escape(label)
    patterns = [
        rf"(?i)\b{escaped_label}\b\s*[:\-]?\s*({PRICE_TOKEN_PATTERN})",
        rf"(?i)\b{escaped_label}\b[^.\n\r]{{0,80}}?({PRICE_TOKEN_PATTERN})",
        rf"(?i)({PRICE_TOKEN_PATTERN})\s*\b{escaped_label}\b",
    ]

    for pattern in patterns:
        value = first_capture(text, pattern)
        normalized = normalized_price_text(value)
        if normalized:
            return normalized

    return None


def parse_market_prices(html: str) -> Tuple[Optional[str], Optional[str]]:
    current_value = parse_current_value_market_prices(html)
    if current_value:
        return current_value

    structured = parse_structured_market_prices(html)
    if structured:
        return structured

    text = plain_text_from_html(html)
    if not text:
        return (None, None)

    new_labels = ["New", "New Price", "New Value", "New Avg Price"]
    used_labels = ["Used", "Used Price", "Used Value", "Used Avg Price"]

    new_price: Optional[str] = None
    for label in new_labels:
        candidate = extract_price(label, text)
        if candidate:
            new_price = candidate
            break

    used_price: Optional[str] = None
    for label in used_labels:
        candidate = extract_price(label, text)
        if candidate:
            used_price = candidate
            break

    return (new_price, used_price)


def decode_body(data: bytes) -> str:
    for encoding in ("utf-8", "iso-8859-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def fetch_html(url: str, cfg: FetchConfig, throttle: RuntimeThrottle) -> Optional[str]:
    parsed_url = urlparse(url)
    if parsed_url.scheme.lower() not in {"http", "https"}:
        return None

    attempts = max(1, cfg.retries + 1)
    ssl_context = ssl._create_unverified_context() if cfg.insecure else None

    for attempt in range(1, attempts + 1):
        request = Request(url)
        request.add_header(
            "User-Agent",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1",
        )
        request.add_header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
        request.add_header("Accept-Language", "en-GB,en;q=0.9")
        request.add_header("Cache-Control", "no-cache")
        request.add_header("Referer", "https://bricklink.com/")

        try:
            with urlopen(request, timeout=cfg.timeout, context=ssl_context) as response:
                status = getattr(response, "status", None) or response.getcode()
                if status == 429:
                    retry_after_header = response.headers.get("Retry-After")
                    retry_after = parse_float(retry_after_header) if retry_after_header else None
                    throttle.apply_rate_limit(retry_after)
                    if attempt < attempts:
                        sleep_for = max(throttle.current_delay, retry_after or 0.0, 1.0)
                        log(
                            f"[Market] HTTP 429 for {url}; retrying in {sleep_for:.1f}s "
                            f"(adaptive_delay={throttle.current_delay:.2f}s)",
                            enabled=cfg.verbose,
                        )
                        time.sleep(sleep_for)
                        continue
                    log(f"[Market] HTTP 429 for {url}", enabled=cfg.verbose)
                    return None

                if status < 200 or status >= 300:
                    log(f"[Market] HTTP {status} for {url}", enabled=cfg.verbose)
                    return None

                throttle.apply_success()
                return decode_body(response.read())

        except HTTPError as exc:
            if exc.code == 429 and attempt < attempts:
                retry_after = parse_float(exc.headers.get("Retry-After")) if exc.headers else None
                throttle.apply_rate_limit(retry_after)
                sleep_for = max(throttle.current_delay, retry_after or 0.0, 1.0)
                log(
                    f"[Market] HTTP 429 for {url}; retrying in {sleep_for:.1f}s "
                    f"(adaptive_delay={throttle.current_delay:.2f}s)",
                    enabled=cfg.verbose,
                )
                time.sleep(sleep_for)
                continue

            log(f"[Market] HTTP {exc.code} for {url}", enabled=cfg.verbose)
            return None
        except (URLError, TimeoutError) as exc:
            log(f"[Market] Request failed for {url}: {exc}", enabled=cfg.verbose)
            return None

    return None


def maybe_sleep(throttle: RuntimeThrottle) -> None:
    throttle.sleep_between_requests()


def normalize_bricklink_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme:
        return ""
    scheme = "https"
    host = parsed.netloc or "www.bricklink.com"
    path = parsed.path
    query = parsed.query
    return f"{scheme}://{host}{path}" + (f"?{query}" if query else "")


def derive_price_guide_url_from_link(link: str) -> Optional[str]:
    raw = collapse_ws(link)
    if not raw:
        return None

    parsed = urlparse(raw)
    host = parsed.netloc.lower()
    path_lower = parsed.path.lower()
    query = parse_qs(parsed.query)

    if "bricklink.com" not in host:
        return None

    if path_lower.endswith("catalogpg.asp"):
        if "S" in query and query["S"]:
            code = query["S"][0].strip()
            if code:
                return f"https://www.bricklink.com/catalogPG.asp?S={quote(code)}&ColorID=0"
        if "M" in query and query["M"]:
            code = query["M"][0].strip().lower()
            if code:
                return f"https://www.bricklink.com/catalogPG.asp?M={quote(code)}&ColorID=0"
        return normalize_bricklink_url(raw)

    if path_lower.endswith("catalogitem.page"):
        if "S" in query and query["S"]:
            code = query["S"][0].strip()
            if code:
                return f"https://www.bricklink.com/catalogPG.asp?S={quote(code)}&ColorID=0"
        if "M" in query and query["M"]:
            code = query["M"][0].strip().lower()
            if code:
                return f"https://www.bricklink.com/catalogPG.asp?M={quote(code)}&ColorID=0"

    return None


def is_minifigure_row(row: Dict[str, Any]) -> bool:
    row_type = collapse_ws(str(row.get("type") or "")).lower()
    if row_type:
        return "minifig" in row_type

    number = collapse_ws(str(row.get("Number") or "")).lower()
    return number.startswith("sw")


def normalize_set_code(number: Any, variant: Any) -> str:
    base = collapse_ws(str(number or "")).upper()
    if not base:
        return ""
    if re.search(r"-[0-9]+$", base):
        return base
    variant_number = parse_int(variant) or 1
    return f"{base}-{variant_number}"


def derive_price_guide_url_from_row(row: Dict[str, Any]) -> Optional[str]:
    number = row.get("Number")
    if number is None:
        return None

    if is_minifigure_row(row):
        code = collapse_ws(str(number)).lower()
        if not code:
            return None
        return f"https://www.bricklink.com/catalogPG.asp?M={quote(code)}&ColorID=0"

    set_code = normalize_set_code(number, row.get("Variant"))
    if not set_code:
        return None
    return f"https://www.bricklink.com/catalogPG.asp?S={quote(set_code)}&ColorID=0"


def extract_price_guide_url_from_html(html: str) -> Optional[str]:
    match = re.search(r"(?is)href=['\"]([^'\"]*catalogPG\.asp[^'\"]*)['\"]", html)
    if not match:
        return None

    href = collapse_ws(match.group(1))
    if not href:
        return None

    if href.startswith("//"):
        href = f"https:{href}"
    elif href.startswith("/"):
        href = f"https://www.bricklink.com{href}"
    elif not href.lower().startswith(("http://", "https://")):
        href = f"https://www.bricklink.com/{href.lstrip('/')}"

    return derive_price_guide_url_from_link(href) or normalize_bricklink_url(href)


def extract_int_by_label(text: str, label: str) -> Optional[int]:
    pattern = re.compile(rf"{re.escape(label)}:\s*([0-9,]+)", re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return None
    return parse_int(match.group(1))


def extract_price_by_label(text: str, label: str) -> Tuple[Optional[float], Optional[str]]:
    pattern = re.compile(
        rf"{re.escape(label)}:\s*([~≈]?\s*(?:GBP|USD|EUR|[£$€])?\s*[0-9][0-9,]*(?:\.[0-9]{{1,2}})?)",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return (None, None)
    return parse_price_token(match.group(1))


def parse_sold_blocks(text: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for match in SOLD_BLOCK_RE.finditer(text):
        min_price, min_currency = parse_price_token(match.group("min"))
        avg_price, avg_currency = parse_price_token(match.group("avg"))
        qty_avg_price, qty_avg_currency = parse_price_token(match.group("qtyavg"))
        max_price, max_currency = parse_price_token(match.group("max"))

        blocks.append(
            {
                "times_sold": parse_int(match.group("times")),
                "total_qty": parse_int(match.group("qty")),
                "min_price": min_price,
                "avg_price": avg_price,
                "qty_avg_price": qty_avg_price,
                "max_price": max_price,
                "currency": min_currency or avg_currency or qty_avg_currency or max_currency,
            }
        )
    return blocks


def parse_lot_blocks(text: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for match in LOTS_BLOCK_RE.finditer(text):
        min_price, min_currency = parse_price_token(match.group("min"))
        avg_price, avg_currency = parse_price_token(match.group("avg"))
        qty_avg_price, qty_avg_currency = parse_price_token(match.group("qtyavg"))
        max_price, max_currency = parse_price_token(match.group("max"))

        blocks.append(
            {
                "total_lots": parse_int(match.group("lots")),
                "total_qty": parse_int(match.group("qty")),
                "min_price": min_price,
                "avg_price": avg_price,
                "qty_avg_price": qty_avg_price,
                "max_price": max_price,
                "currency": min_currency or avg_currency or qty_avg_currency or max_currency,
            }
        )
    return blocks


def parse_month_transactions(
    month_key: str,
    month_label: str,
    chunk_lines: List[str],
) -> List[Dict[str, Any]]:
    transactions: List[Dict[str, Any]] = []
    for raw_line in chunk_lines:
        lower = raw_line.lower()
        if lower.startswith("total lots:"):
            break
        if lower in {"qty", "each", "qty each"}:
            continue

        match = TRANSACTION_LINE_RE.match(raw_line)
        if not match:
            continue

        qty = parse_int(match.group("qty"))
        amount, currency = parse_price_token(match.group("price"))
        if qty is None or amount is None:
            continue

        transactions.append(
            {
                "month": month_key,
                "month_label": month_label,
                "qty": qty,
                "each_price": amount,
                "currency": currency,
            }
        )

    return transactions


def parse_monthly_blocks(lines: List[str]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    i = 0

    while i < len(lines):
        header_match = MONTH_HEADER_RE.match(lines[i])
        if not header_match:
            i += 1
            continue

        month_name = header_match.group("month").title()
        year = int(header_match.group("year"))
        month_index = MONTH_INDEX_BY_NAME.get(month_name.lower())
        if month_index is None:
            i += 1
            continue

        month_key = f"{year:04d}-{month_index:02d}"
        month_label = f"{month_name} {year}"

        i += 1
        chunk: List[str] = []
        while i < len(lines):
            if MONTH_HEADER_RE.match(lines[i]):
                break
            if lines[i].lower() == "currently available":
                break
            chunk.append(lines[i])
            i += 1

        block_text = " ".join(chunk)
        if not block_text:
            continue

        total_lots = extract_int_by_label(block_text, "Total Lots")
        total_qty = extract_int_by_label(block_text, "Total Qty")
        avg_price, avg_currency = extract_price_by_label(block_text, "Avg Price")
        transactions = parse_month_transactions(month_key, month_label, chunk)

        latest_sale_price: Optional[float] = None
        latest_sale_currency: Optional[str] = None
        if transactions:
            latest_sale_price = transactions[0].get("each_price")
            latest_sale_currency = transactions[0].get("currency")
        else:
            sale_rows_text = block_text.split("Total Lots:", 1)[0]
            sale_price_tokens = PRICE_TOKEN_RE.findall(sale_rows_text)
            if sale_price_tokens:
                latest_sale_price, latest_sale_currency = parse_price_token(sale_price_tokens[0])

        blocks.append(
            {
                "month": month_key,
                "month_label": month_label,
                "total_lots": total_lots,
                "total_qty": total_qty,
                "avg_price": avg_price,
                "currency": avg_currency or latest_sale_currency,
                "latest_sale_price": latest_sale_price,
                "transactions": transactions,
            }
        )

    return blocks


def split_monthly_blocks(
    monthly_blocks: List[Dict[str, Any]],
    sold_new_times: Optional[int],
    sold_used_times: Optional[int],
    new_anchor: Optional[float],
    used_anchor: Optional[float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not monthly_blocks:
        return ([], [])

    # Preferred strategy: BrickLink commonly lists New month blocks first, then Used.
    if sold_new_times and sold_used_times:
        new_blocks: List[Dict[str, Any]] = []
        used_blocks: List[Dict[str, Any]] = []
        consumed_new_lots = 0

        for block in monthly_blocks:
            lots = block.get("total_lots") or 0
            if consumed_new_lots < sold_new_times:
                new_blocks.append(block)
                consumed_new_lots += lots if lots > 0 else 1
            else:
                used_blocks.append(block)

        if new_blocks and used_blocks:
            return (new_blocks, used_blocks)

    # Fallback strategy: classify by proximity to New/Used anchors.
    new_blocks = []
    used_blocks = []

    for block in monthly_blocks:
        avg_price = block.get("avg_price")
        if avg_price is not None and new_anchor is not None and used_anchor is not None:
            if abs(avg_price - new_anchor) <= abs(avg_price - used_anchor):
                new_blocks.append(block)
            else:
                used_blocks.append(block)
            continue

        if new_anchor is not None and used_anchor is None:
            new_blocks.append(block)
            continue

        if used_anchor is not None and new_anchor is None:
            used_blocks.append(block)
            continue

        if len(new_blocks) <= len(used_blocks):
            new_blocks.append(block)
        else:
            used_blocks.append(block)

    return (new_blocks, used_blocks)


def monthly_series(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(blocks, key=lambda block: block.get("month", ""))
    out: List[Dict[str, Any]] = []
    for block in ordered:
        out.append(
            {
                "month": block.get("month"),
                "monthLabel": block.get("month_label"),
                "avgPrice": round(block["avg_price"], 2) if isinstance(block.get("avg_price"), (int, float)) else None,
                "totalLots": block.get("total_lots"),
                "totalQty": block.get("total_qty"),
            }
        )
    return out


def transaction_series(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered_blocks = sorted(blocks, key=lambda block: block.get("month", ""))
    out: List[Dict[str, Any]] = []

    for block in ordered_blocks:
        month = block.get("month")
        month_label = block.get("month_label")
        transactions = block.get("transactions") or []
        for sequence, transaction in enumerate(transactions, start=1):
            each_price = transaction.get("each_price")
            out.append(
                {
                    "month": month,
                    "monthLabel": month_label,
                    "sequence": sequence,
                    "qty": transaction.get("qty"),
                    "eachPrice": round(each_price, 2) if isinstance(each_price, (int, float)) else None,
                    "currency": transaction.get("currency"),
                }
            )

    return out


def latest_sale_from_blocks(blocks: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    if not blocks:
        return (None, None, None)

    latest_block = max(blocks, key=lambda block: block.get("month", ""))
    price = latest_block.get("latest_sale_price")
    if price is None:
        price = latest_block.get("avg_price")

    currency = latest_block.get("currency")
    return (latest_block.get("month"), price, currency)


def compute_forecast(current_avg: Optional[float], sold_avg: Optional[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if current_avg is None or current_avg <= 0:
        return (None, None, None)

    annual_growth: float
    if sold_avg is not None and sold_avg > 0:
        # Approximate annualized trend from 6-month bridge between sold and current averages.
        annual_growth = (current_avg / sold_avg) ** 2 - 1
        annual_growth = max(min(annual_growth, 1.25), -0.50)
    else:
        annual_growth = 0.0

    forecast_2y = round(current_avg * ((1 + annual_growth) ** 2), 2)
    forecast_5y = round(current_avg * ((1 + annual_growth) ** 5), 2)
    growth_pct = round(annual_growth * 100.0, 2)
    return (forecast_2y, forecast_5y, growth_pct)


def first_non_none(values: Sequence[Optional[Any]]) -> Optional[Any]:
    for value in values:
        if value is not None:
            return value
    return None


def parse_price_guide_analytics(
    html: str,
    row: Dict[str, Any],
    *,
    price_guide_url: str,
) -> Tuple[Dict[str, Any], Optional[str], Optional[str], bool]:
    text = plain_text_from_html(html)
    lines = lines_from_html(html)

    sold_blocks = parse_sold_blocks(text)
    lot_blocks = parse_lot_blocks(text)
    month_blocks = parse_monthly_blocks(lines)

    sold_new = sold_blocks[0] if len(sold_blocks) >= 1 else None
    sold_used = sold_blocks[1] if len(sold_blocks) >= 2 else None

    current_new = lot_blocks[0] if len(lot_blocks) >= 1 else None
    current_used = lot_blocks[1] if len(lot_blocks) >= 2 else None

    currency = first_non_none(
        [
            sold_new.get("currency") if sold_new else None,
            sold_used.get("currency") if sold_used else None,
            current_new.get("currency") if current_new else None,
            current_used.get("currency") if current_used else None,
        ]
    )

    analytics: Dict[str, Any] = {
        "BrickLinkPriceGuideURL": price_guide_url,
        "BrickLinkPriceGuideCurrency": currency,
        "BrickLink6MSoldNewTimesSold": sold_new.get("times_sold") if sold_new else None,
        "BrickLink6MSoldNewTotalQty": sold_new.get("total_qty") if sold_new else None,
        "BrickLink6MSoldNewMinPrice": round(sold_new["min_price"], 2) if sold_new and sold_new.get("min_price") is not None else None,
        "BrickLink6MSoldNewAvgPrice": round(sold_new["avg_price"], 2) if sold_new and sold_new.get("avg_price") is not None else None,
        "BrickLink6MSoldNewQtyAvgPrice": round(sold_new["qty_avg_price"], 2) if sold_new and sold_new.get("qty_avg_price") is not None else None,
        "BrickLink6MSoldNewMaxPrice": round(sold_new["max_price"], 2) if sold_new and sold_new.get("max_price") is not None else None,
        "BrickLink6MSoldUsedTimesSold": sold_used.get("times_sold") if sold_used else None,
        "BrickLink6MSoldUsedTotalQty": sold_used.get("total_qty") if sold_used else None,
        "BrickLink6MSoldUsedMinPrice": round(sold_used["min_price"], 2) if sold_used and sold_used.get("min_price") is not None else None,
        "BrickLink6MSoldUsedAvgPrice": round(sold_used["avg_price"], 2) if sold_used and sold_used.get("avg_price") is not None else None,
        "BrickLink6MSoldUsedQtyAvgPrice": round(sold_used["qty_avg_price"], 2) if sold_used and sold_used.get("qty_avg_price") is not None else None,
        "BrickLink6MSoldUsedMaxPrice": round(sold_used["max_price"], 2) if sold_used and sold_used.get("max_price") is not None else None,
        "BrickLinkCurrentNewTotalLots": current_new.get("total_lots") if current_new else None,
        "BrickLinkCurrentNewTotalQty": current_new.get("total_qty") if current_new else None,
        "BrickLinkCurrentNewMinPrice": round(current_new["min_price"], 2) if current_new and current_new.get("min_price") is not None else None,
        "BrickLinkCurrentNewAvgPrice": round(current_new["avg_price"], 2) if current_new and current_new.get("avg_price") is not None else None,
        "BrickLinkCurrentNewQtyAvgPrice": round(current_new["qty_avg_price"], 2) if current_new and current_new.get("qty_avg_price") is not None else None,
        "BrickLinkCurrentNewMaxPrice": round(current_new["max_price"], 2) if current_new and current_new.get("max_price") is not None else None,
        "BrickLinkCurrentUsedTotalLots": current_used.get("total_lots") if current_used else None,
        "BrickLinkCurrentUsedTotalQty": current_used.get("total_qty") if current_used else None,
        "BrickLinkCurrentUsedMinPrice": round(current_used["min_price"], 2) if current_used and current_used.get("min_price") is not None else None,
        "BrickLinkCurrentUsedAvgPrice": round(current_used["avg_price"], 2) if current_used and current_used.get("avg_price") is not None else None,
        "BrickLinkCurrentUsedQtyAvgPrice": round(current_used["qty_avg_price"], 2) if current_used and current_used.get("qty_avg_price") is not None else None,
        "BrickLinkCurrentUsedMaxPrice": round(current_used["max_price"], 2) if current_used and current_used.get("max_price") is not None else None,
    }

    sold_new_avg = sold_new.get("avg_price") if sold_new else None
    sold_used_avg = sold_used.get("avg_price") if sold_used else None
    current_new_avg = current_new.get("avg_price") if current_new else None
    current_used_avg = current_used.get("avg_price") if current_used else None

    new_anchor = first_non_none([sold_new_avg, current_new_avg])
    used_anchor = first_non_none([sold_used_avg, current_used_avg])
    sold_new_times = sold_new.get("times_sold") if sold_new else None
    sold_used_times = sold_used.get("times_sold") if sold_used else None

    monthly_new, monthly_used = split_monthly_blocks(
        month_blocks,
        sold_new_times=sold_new_times,
        sold_used_times=sold_used_times,
        new_anchor=new_anchor,
        used_anchor=used_anchor,
    )

    analytics["BrickLinkMonthlySalesNew"] = monthly_series(monthly_new)
    analytics["BrickLinkMonthlySalesUsed"] = monthly_series(monthly_used)
    analytics["BrickLinkTransactionsNew"] = transaction_series(monthly_new)
    analytics["BrickLinkTransactionsUsed"] = transaction_series(monthly_used)
    analytics["BrickLinkTransactionsNewCount"] = len(analytics["BrickLinkTransactionsNew"])
    analytics["BrickLinkTransactionsUsedCount"] = len(analytics["BrickLinkTransactionsUsed"])

    latest_new_month, latest_new_price, _latest_new_currency = latest_sale_from_blocks(monthly_new)
    latest_used_month, latest_used_price, _latest_used_currency = latest_sale_from_blocks(monthly_used)

    analytics["BrickLinkLatestSaleNewMonth"] = latest_new_month
    analytics["BrickLinkLatestSaleNewPrice"] = round(latest_new_price, 2) if latest_new_price is not None else None
    analytics["BrickLinkLatestSaleUsedMonth"] = latest_used_month
    analytics["BrickLinkLatestSaleUsedPrice"] = round(latest_used_price, 2) if latest_used_price is not None else None

    used_min_candidates = [
        current_used.get("min_price") if current_used else None,
        sold_used.get("min_price") if sold_used else None,
    ]
    used_max_candidates = [
        current_used.get("max_price") if current_used else None,
        sold_used.get("max_price") if sold_used else None,
    ]

    used_min = min((value for value in used_min_candidates if value is not None), default=None)
    used_max = max((value for value in used_max_candidates if value is not None), default=None)
    analytics["BrickLinkUsedPriceRangeMin"] = round(used_min, 2) if used_min is not None else None
    analytics["BrickLinkUsedPriceRangeMax"] = round(used_max, 2) if used_max is not None else None

    comparison_rrp = parse_float(row.get("UKRetailPrice"))
    current_new_for_compare = first_non_none([current_new_avg, sold_new_avg])
    if currency == "GBP" and comparison_rrp is not None and comparison_rrp > 0 and current_new_for_compare is not None:
        delta = current_new_for_compare - comparison_rrp
        pct = (delta / comparison_rrp) * 100.0
        analytics["CurrentNewVsRRPAmount"] = round(delta, 2)
        analytics["CurrentNewVsRRPPercent"] = round(pct, 2)
        analytics["CurrentRRPBaseline"] = round(comparison_rrp, 2)
    else:
        analytics["CurrentNewVsRRPAmount"] = None
        analytics["CurrentNewVsRRPPercent"] = None
        analytics["CurrentRRPBaseline"] = round(comparison_rrp, 2) if comparison_rrp is not None else None

    forecast2_new, forecast5_new, growth_new_pct = compute_forecast(current_new_avg, sold_new_avg)
    forecast2_used, forecast5_used, growth_used_pct = compute_forecast(current_used_avg, sold_used_avg)

    analytics["PriceForecast2YNew"] = forecast2_new
    analytics["PriceForecast5YNew"] = forecast5_new
    analytics["PriceForecast2YUsed"] = forecast2_used
    analytics["PriceForecast5YUsed"] = forecast5_used
    analytics["PriceTrendAnnualizedNewPercent"] = growth_new_pct
    analytics["PriceTrendAnnualizedUsedPercent"] = growth_used_pct

    analytics["PriceForecastMethod"] = "6m_sold_to_current_avg_annualized"

    new_display_amount = first_non_none([sold_new_avg, current_new_avg])
    used_display_amount = first_non_none([sold_used_avg, current_used_avg])

    new_display_currency = first_non_none([sold_new.get("currency") if sold_new else None, current_new.get("currency") if current_new else None, currency])
    used_display_currency = first_non_none([sold_used.get("currency") if sold_used else None, current_used.get("currency") if current_used else None, currency])

    new_display = format_display_price(new_display_amount, new_display_currency, approximate=True)
    used_display = format_display_price(used_display_amount, used_display_currency, approximate=True)

    parsed_any = any(
        value not in (None, [], "")
        for key, value in analytics.items()
        if key not in {"BrickLinkPriceGuideURL", "PriceForecastMethod"}
    ) or new_display is not None or used_display is not None

    return (analytics, new_display, used_display, parsed_any)


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
        exclusive = sorted(
            [
                code
                for code in unique_minifigs
                if len(minifig_to_sets.get(code, set())) == 1
            ]
        )

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
        number = collapse_ws(str(row.get("Number") or "")).lower()
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


def update_rows(
    rows: List[Dict[str, Any]],
    cfg: FetchConfig,
    throttle: RuntimeThrottle,
    *,
    limit: Optional[int],
    start_index: int,
    label: str,
) -> FileUpdateStats:
    stats = FileUpdateStats(total_rows=len(rows))

    html_cache: Dict[str, Optional[str]] = {}
    fallback_cache: Dict[str, Optional[str]] = {}

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue

        if idx < start_index:
            continue
        if limit is not None and stats.rows_considered >= limit:
            break

        stats.rows_considered += 1

        link = collapse_ws(str(row.get("link") or ""))
        if link:
            stats.rows_with_link += 1

        pg_url = derive_price_guide_url_from_link(link)
        if not pg_url:
            pg_url = derive_price_guide_url_from_row(row)

        if not pg_url and link:
            # As a last resort, fetch linked page and look for a catalogPG anchor.
            if link in fallback_cache:
                primary_html = fallback_cache[link]
            else:
                log(f"[{label}] Fetching linked page for PG discovery: {link}", enabled=cfg.verbose)
                primary_html = fetch_html(link, cfg, throttle)
                fallback_cache[link] = primary_html
                maybe_sleep(throttle)

            if primary_html:
                pg_url = extract_price_guide_url_from_html(primary_html)

        if not pg_url:
            continue

        stats.rows_with_price_guide += 1

        if pg_url in html_cache:
            price_guide_html = html_cache[pg_url]
        else:
            log(f"[{label}] Fetching price guide row {idx + 1}/{len(rows)}: {pg_url}", enabled=cfg.verbose)
            price_guide_html = fetch_html(pg_url, cfg, throttle)
            html_cache[pg_url] = price_guide_html
            maybe_sleep(throttle)

        if not price_guide_html:
            stats.fetch_failures += 1
            continue

        analytics, new_price, used_price, parsed_any = parse_price_guide_analytics(
            price_guide_html,
            row,
            price_guide_url=pg_url,
        )

        row_changed = False
        for key, value in analytics.items():
            if row.get(key) != value:
                row[key] = value
                row_changed = True

        if new_price is not None and row.get("New") != new_price:
            row["New"] = new_price
            row_changed = True

        if used_price is not None and row.get("Used") != used_price:
            row["Used"] = used_price
            row_changed = True

        if is_minifigure_row(row):
            theme_value = collapse_ws(str(row.get("Theme") or row.get("Category") or "Star Wars"))
            if row.get("Theme") != theme_value:
                row["Theme"] = theme_value
                row_changed = True

        # Optional fallback to existing linked page parsing (off by default).
        if cfg.link_fallback and (new_price is None or used_price is None) and link:
            if link in fallback_cache:
                linked_html = fallback_cache[link]
            else:
                log(f"[{label}] Fallback fetch for display values: {link}", enabled=cfg.verbose)
                linked_html = fetch_html(link, cfg, throttle)
                fallback_cache[link] = linked_html
                maybe_sleep(throttle)

            if linked_html:
                fallback_new, fallback_used = parse_market_prices(linked_html)
                if new_price is None and fallback_new and row.get("New") != fallback_new:
                    row["New"] = fallback_new
                    row_changed = True
                if used_price is None and fallback_used and row.get("Used") != fallback_used:
                    row["Used"] = fallback_used
                    row_changed = True

        if not parsed_any:
            stats.parse_misses += 1

        if row_changed:
            stats.rows_changed += 1

    return stats


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level array in {path}")
    out: List[Dict[str, Any]] = []
    for row in data:
        if isinstance(row, dict):
            out.append(row)
    return out


def maybe_write_json(path: Path, rows: List[Dict[str, Any]]) -> bool:
    original_text = path.read_text(encoding="utf-8")
    updated_text = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    if updated_text == original_text:
        return False
    path.write_text(updated_text, encoding="utf-8")
    return True


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update LEGO market values and BrickLink analytics in JSON files.")
    parser.add_argument(
        "--sets-json",
        default="dist/Lego Star Wars Database.json",
        help="Path to sets JSON file.",
    )
    parser.add_argument(
        "--minifigs-json",
        default="dist/Lego-Star-Wars-Minifigure-Database.json",
        help="Path to minifigures JSON file.",
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=1, help="Retries after first attempt.")
    parser.add_argument("--delay", type=float, default=0.65, help="Minimum delay between requests in seconds.")
    parser.add_argument("--jitter", type=float, default=0.15, help="Max random jitter added to adaptive delay.")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-file row limit for testing.")
    parser.add_argument("--start-index", type=int, default=0, help="Optional per-file start index.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL certificate verification (for local troubleshooting only).",
    )
    parser.add_argument(
        "--link-fallback",
        action="store_true",
        help="If New/Used cannot be parsed from price guide, fall back to parsing row.link page.",
    )
    parser.add_argument(
        "--skip-cross-enrichment",
        action="store_true",
        help="Skip exclusivity and appears-in cross-catalog enrichment fields.",
    )
    return parser.parse_args(argv)


def print_summary(label: str, stats: FileUpdateStats) -> None:
    print(
        (
            f"[{label}] total={stats.total_rows} "
            f"considered={stats.rows_considered} "
            f"with_link={stats.rows_with_link} "
            f"with_pg={stats.rows_with_price_guide} "
            f"changed={stats.rows_changed} "
            f"fetch_failures={stats.fetch_failures} "
            f"parse_misses={stats.parse_misses} "
            f"cross_changed={stats.cross_rows_changed}"
        ),
        flush=True,
    )


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    cfg = FetchConfig(
        timeout=max(1.0, args.timeout),
        retries=max(0, args.retries),
        delay=max(0.0, args.delay),
        jitter=max(0.0, args.jitter),
        verbose=args.verbose,
        insecure=args.insecure,
        link_fallback=bool(args.link_fallback),
    )
    throttle = RuntimeThrottle(min_delay=cfg.delay, jitter=cfg.jitter)

    sets_path = Path(args.sets_json)
    minifigs_path = Path(args.minifigs_json)

    if not sets_path.exists():
        print(f"Missing sets JSON: {sets_path}", file=sys.stderr)
        return 1

    if not minifigs_path.exists():
        print(f"Missing minifig JSON: {minifigs_path}", file=sys.stderr)
        return 1

    set_rows = load_json_array(sets_path)
    minifig_rows = load_json_array(minifigs_path)

    sets_stats = update_rows(
        set_rows,
        cfg,
        throttle,
        limit=args.limit,
        start_index=max(0, args.start_index),
        label="Sets",
    )
    minifigs_stats = update_rows(
        minifig_rows,
        cfg,
        throttle,
        limit=args.limit,
        start_index=max(0, args.start_index),
        label="Minifigs",
    )

    if not args.skip_cross_enrichment:
        set_cross, minifig_cross = apply_cross_catalog_enrichment(set_rows, minifig_rows)
        sets_stats.cross_rows_changed = set_cross
        minifigs_stats.cross_rows_changed = minifig_cross

    sets_written = maybe_write_json(sets_path, set_rows)
    minifigs_written = maybe_write_json(minifigs_path, minifig_rows)

    if args.verbose:
        print(f"[Write] sets_written={sets_written} minifigs_written={minifigs_written}", flush=True)

    print_summary("Sets", sets_stats)
    print_summary("Minifigs", minifigs_stats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
