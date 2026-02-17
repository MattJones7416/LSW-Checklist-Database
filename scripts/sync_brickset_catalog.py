#!/usr/bin/env python3
"""Synchronize set/minifigure catalogs from Brickset.

What this script does:
- Pulls sets from Brickset API v3 (`getSets`) across all themes.
- Optionally crawls Brickset minifigure listing pages.
- Merges new/updated records into existing JSON files.
- Refreshes `New` / `Used` current value fields from listing data.
- Expands set `MinifigNumbers` multiplicities from BrickLink inventories.

This is intended for unattended runs in GitHub Actions.
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
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse

import requests


SETS_BASE_URL = "https://brickset.com/sets/theme-Star-Wars"
MINIFIGS_BASE_URL = "https://brickset.com/minifigs/category-Star-Wars"
BRICKLINK_INVENTORY_URL_TEMPLATE = "https://www.bricklink.com/catalogItemInv.asp?S={set_code}&viewItemType=M"
BRICKSET_API_BASE_URL = "https://brickset.com/api/v3.asmx"
SCRIPT_VERSION = "2026-02-17.7"
SOFT_BLOCK_MARKERS = (
    "cf-chl-",
    "/cdn-cgi/challenge-platform/",
    "attention required!",
    "checking your browser before accessing",
    "verify you are human",
    "captcha",
    "please complete the security check",
)

PRICE_TOKEN_PATTERN = (
    r"(?:[~≈]?\s*(?:£|\$|€)\s*[0-9][0-9,]*(?:\.[0-9]{1,2})?"
    r"|(?:USD|GBP|EUR)\s*[0-9][0-9,]*(?:\.[0-9]{1,2})"
    r"|[0-9][0-9,]*(?:\.[0-9]{1,2}))"
)
PRICE_TOKEN_RE = re.compile(PRICE_TOKEN_PATTERN, re.IGNORECASE)

ARTICLE_WITH_CLASS_RE = re.compile(
    r"(?is)<article\b[^>]*class\s*=\s*(['\"])(.*?)\1[^>]*>.*?</article>"
)
ARTICLE_GENERIC_RE = re.compile(r"(?is)<article\b[^>]*>.*?</article>")
DT_DD_RE = re.compile(r"<dt>(.*?)</dt>\s*<dd(?:[^>]*)>(.*?)</dd>", re.IGNORECASE | re.DOTALL)
SET_CODE_RE = re.compile(r"/sets/([A-Za-z0-9]+-[0-9]+)(?:/|['\"?#])", re.IGNORECASE)
MINIFIG_CODE_RE = re.compile(r"/minifigs/([a-z0-9]+)(?:/|['\"?#])", re.IGNORECASE)
PAGE_NUMBER_RE_TEMPLATE = r"{path}/page-(\d+)"
BRICKLINK_ROW_RE = re.compile(
    r'<TR[^>]*class="IV_([A-Za-z0-9]+)\s+IV_ITEM"[^>]*>.*?'
    r'<TD[^>]*ALIGN="RIGHT"[^>]*>\s*&nbsp;\s*([0-9]+)\s*&nbsp;\s*</TD>',
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class FetchConfig:
    timeout_seconds: float
    retries: int
    page_delay_seconds: float
    page_jitter_seconds: float
    bricklink_delay_seconds: float
    bricklink_jitter_seconds: float
    verbose: bool


@dataclass
class CrawlStats:
    pages_fetched: int = 0
    articles_parsed: int = 0
    records_parsed: int = 0
    failed_pages: int = 0


@dataclass
class MultiplicityStats:
    considered: int = 0
    fetched: int = 0
    updated: int = 0
    failures: int = 0


def log(msg: str, *, enabled: bool) -> None:
    if enabled:
        print(msg, flush=True)


def collapse_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def decode_entities(value: str) -> str:
    text = unescape(value)
    text = text.replace("\u00a0", " ")
    return text


def strip_tags(value: str) -> str:
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", value)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = decode_entities(text)
    return collapse_ws(text)


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
    return int(match.group(0).replace(",", ""))


def normalize_price_text(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None

    text = collapse_ws(raw)
    if not text:
        return None

    prefix = ""
    if text.startswith("~") or text.startswith("≈"):
        prefix = "~"
        text = collapse_ws(text[1:])

    text = re.sub(r"(?i)^USD\s*", "$", text)
    text = re.sub(r"(?i)^GBP\s*", "£", text)
    text = re.sub(r"(?i)^EUR\s*", "€", text)
    text = re.sub(r"([£$€])\s+([0-9])", r"\1\2", text)

    if not text:
        return None

    return f"{prefix}{text}" if prefix else text


def extract_price_from_html(html_fragment: Optional[str]) -> Optional[str]:
    if not html_fragment:
        return None
    text = decode_entities(html_fragment)
    text = re.sub(r"<[^>]+>", " ", text)
    text = collapse_ws(text)
    match = PRICE_TOKEN_RE.search(text)
    if not match:
        return None
    return normalize_price_text(match.group(0))


def maybe_sleep(base_delay: float, jitter: float) -> None:
    delay = max(0.0, base_delay)
    if jitter > 0:
        delay += random.uniform(0.0, jitter)
    if delay > 0:
        time.sleep(delay)


def looks_like_soft_block(html: str) -> bool:
    lower = html.lower()
    if any(marker in lower for marker in SOFT_BLOCK_MARKERS):
        return True
    # Cloudflare challenge pages commonly include this combined wording.
    return "attention required" in lower and "cloudflare" in lower


def extract_html_title(html: str) -> str:
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    if not match:
        return ""
    return collapse_ws(strip_tags(match.group(1)))


def parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = collapse_ws(str(value)).lower()
    if not text:
        return None
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0"}:
        return False
    return None


def api_pick(row: Dict[str, Any], *keys: str) -> Any:
    lower_map = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        candidate = lower_map.get(key.lower())
        if candidate is not None:
            return candidate
    return None


def parse_api_payload_text(text: str) -> Dict[str, Any]:
    data: Any = json.loads(text)
    if isinstance(data, dict) and "d" in data:
        wrapped = data["d"]
        if isinstance(wrapped, str):
            wrapped = json.loads(wrapped)
        data = wrapped
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        raise ValueError("Brickset API response was not an object")
    return data


def format_catalog_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = collapse_ws(str(value))
    if not text:
        return None
    # Already in app-compatible format.
    if re.match(r"^[0-9]{2}/[0-9]{2}/[0-9]{4}\s+[0-9]{2}:[0-9]{2}:[0-9]{2}$", text):
        return text
    # Common API shapes.
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%d %b %Y", "%d %B %Y"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.strftime("%d/%m/%Y %H:%M:%S")
        except ValueError:
            continue
    return None


def extract_article_blocks(html: str, *, mode: str) -> List[str]:
    blocks: List[str] = []

    def is_relevant(block: str) -> bool:
        if mode == "sets":
            return SET_CODE_RE.search(block) is not None
        return MINIFIG_CODE_RE.search(block) is not None or "/minifigs/" in block.lower()

    for match in ARTICLE_WITH_CLASS_RE.finditer(html):
        class_names = collapse_ws(match.group(2)).lower()
        if "set" not in class_names:
            continue
        block = match.group(0)
        if is_relevant(block):
            blocks.append(block)

    if blocks:
        return blocks

    for match in ARTICLE_GENERIC_RE.finditer(html):
        block = match.group(0)
        if is_relevant(block):
            blocks.append(block)

    return blocks


def to_absolute_url(path_or_url: Optional[str]) -> str:
    if not path_or_url:
        return ""
    raw = collapse_ws(path_or_url)
    if raw.startswith("//"):
        return f"https:{raw}"
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    return urljoin("https://brickset.com", raw)


def strip_query(url: str) -> str:
    return url.split("?", 1)[0]


def normalize_set_image_url(path_or_url: Optional[str]) -> str:
    absolute = to_absolute_url(path_or_url)
    if not absolute:
        return ""
    normalized = strip_query(absolute)
    normalized = normalized.replace("/sets/small/", "/sets/images/")
    normalized = normalized.replace("/sets/large/", "/sets/images/")
    return normalized


def unique_preserving_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def parse_minifig_numbers(raw: Any) -> List[str]:
    if raw is None:
        return []
    return [
        token.strip().lower()
        for token in re.split(r"[,;\n\r]+", str(raw))
        if token.strip()
    ]


def minifig_numbers_string(values: Sequence[str]) -> Optional[str]:
    if not values:
        return None
    return ",".join(values) + ","


def extract_dt_map(article_html: str) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for raw_dt, raw_dd in DT_DD_RE.findall(article_html):
        key = strip_tags(raw_dt).lower()
        if not key:
            continue
        mapping.setdefault(key, []).append(raw_dd)
    return mapping


def first_from_map(mapping: Dict[str, List[str]], key: str) -> Optional[str]:
    values = mapping.get(key.lower())
    if not values:
        return None
    return values[0]


def nth_from_map(mapping: Dict[str, List[str]], key: str, index: int) -> Optional[str]:
    values = mapping.get(key.lower())
    if not values:
        return None
    if index < 0 or index >= len(values):
        return None
    return values[index]


def parse_launch_exit_dates(launch_exit_text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not launch_exit_text:
        return (None, None)

    text = collapse_ws(launch_exit_text)
    match = re.search(
        r"([0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})\s*-\s*([0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})",
        text,
    )
    if not match:
        return (None, None)

    def convert_date(raw: str) -> Optional[str]:
        for fmt in ("%d %b %Y", "%d %B %Y"):
            try:
                parsed = datetime.strptime(raw, fmt)
                return parsed.strftime("%d/%m/%Y 00:00:00")
            except ValueError:
                continue
        return None

    launch = convert_date(match.group(1))
    exit_ = convert_date(match.group(2))
    return (launch, exit_)


def set_key(number: Any, variant: Any) -> Tuple[str, int]:
    number_key = collapse_ws(str(number)).upper()
    variant_value = parse_int(variant) or 1
    return (number_key, variant_value)


def minifig_key(number: Any) -> str:
    return collapse_ws(str(number)).lower()


def parse_set_article(article_html: str) -> Optional[Dict[str, Any]]:
    code_match = SET_CODE_RE.search(article_html)
    if not code_match:
        return None

    set_code = code_match.group(1)
    if "-" not in set_code:
        return None

    raw_number, raw_variant = set_code.rsplit("-", 1)
    variant = parse_int(raw_variant) or 1
    number: Any = int(raw_number) if raw_number.isdigit() else raw_number.upper()

    dt_map = extract_dt_map(article_html)

    meta_link_match = re.search(
        r"<div\b[^>]*class=['\"][^'\"]*\bmeta\b[^'\"]*['\"][^>]*>\s*<h1>\s*<a\b[^>]*href=['\"]([^'\"]+)['\"]",
        article_html,
        re.IGNORECASE | re.DOTALL,
    )
    if not meta_link_match:
        meta_link_match = re.search(
            r"<h1>\s*<a\b[^>]*href=['\"]([^'\"]+)['\"]",
            article_html,
            re.IGNORECASE | re.DOTALL,
        )
    if not meta_link_match:
        return None
    link = to_absolute_url(meta_link_match.group(1))

    name_match = re.search(
        r"<div\b[^>]*class=['\"][^'\"]*\bmeta\b[^'\"]*['\"][^>]*>\s*<h1>\s*<a\b[^>]*>(.*?)</a>\s*</h1>",
        article_html,
        re.IGNORECASE | re.DOTALL,
    )
    if not name_match:
        name_match = re.search(
            r"<h1>\s*<a\b[^>]*>(.*?)</a>\s*</h1>",
            article_html,
            re.IGNORECASE | re.DOTALL,
        )
    set_name_raw = strip_tags(name_match.group(1)) if name_match else ""
    set_name = collapse_ws(set_name_raw)
    set_prefixes = [set_code, raw_number, f"{raw_number}-{variant}"]
    for prefix in set_prefixes:
        if not prefix:
            continue
        if set_name.lower().startswith(prefix.lower()):
            trimmed = set_name[len(prefix):].lstrip(" :-")
            if trimmed:
                set_name = collapse_ws(trimmed)
            break
    if not set_name:
        return None

    year_match = re.search(r"<a\b[^>]*class=['\"][^'\"]*\byear\b[^'\"]*['\"][^>]*>([^<]+)</a>", article_html, re.IGNORECASE)
    year_from = parse_int(year_match.group(1)) if year_match else None
    if year_from == 0:
        year_from = None

    subtheme_match = re.search(
        r"<a\b[^>]*class=['\"][^'\"]*\bsubtheme\b[^'\"]*['\"][^>]*>(.*?)</a>",
        article_html,
        re.IGNORECASE | re.DOTALL,
    )
    subtheme = strip_tags(subtheme_match.group(1)) if subtheme_match else ""

    pieces = parse_int(strip_tags(first_from_map(dt_map, "pieces") or ""))
    minifigs_count = parse_int(strip_tags(first_from_map(dt_map, "minifigs") or ""))

    minifigs_tags_html = nth_from_map(dt_map, "minifigs", 1)
    minifig_numbers: List[str] = []
    if minifigs_tags_html:
        for code in MINIFIG_CODE_RE.findall(minifigs_tags_html):
            normalized = code.lower()
            if normalized.startswith("in-"):
                continue
            minifig_numbers.append(normalized)
    minifig_numbers = unique_preserving_order(minifig_numbers)

    packaging = strip_tags(first_from_map(dt_map, "packaging") or "")
    packaging = packaging or None

    additional_images = parse_int(strip_tags(first_from_map(dt_map, "additional images") or ""))

    new_value = extract_price_from_html(first_from_map(dt_map, "value new"))
    used_value = extract_price_from_html(first_from_map(dt_map, "value used"))

    launch_exit_text = strip_tags(first_from_map(dt_map, "launch/exit") or "")
    launch_date, exit_date = parse_launch_exit_dates(launch_exit_text)

    image_match = re.search(r"<img\s+[^>]*src=\"([^\"]+)\"", article_html)
    product_image = normalize_set_image_url(image_match.group(1) if image_match else None)

    community_text = strip_tags(article_html)
    own_count: Optional[int] = None
    want_count: Optional[int] = None
    community_match = re.search(r"([0-9,]+)\s+own this set,\s*([0-9,]+)\s+want it", community_text, re.IGNORECASE)
    if community_match:
        own_count = parse_int(community_match.group(1))
        want_count = parse_int(community_match.group(2))

    parsed: Dict[str, Any] = {
        "Number": number,
        "Variant": variant,
        "SetName": set_name,
        "YearFrom": year_from,
        "Subtheme": subtheme or None,
        "Theme": "Star Wars",
        "Pieces": pieces,
        "Minifigs": minifigs_count,
        "MinifigNumbers": minifig_numbers_string(minifig_numbers),
        "PackagingType": packaging,
        "AdditionalImageCount": additional_images,
        "OwnCount": own_count,
        "WantCount": want_count,
        "LaunchDate": launch_date,
        "ExitDate": exit_date,
        "New": new_value,
        "Used": used_value,
        "link": link,
        "productImage": product_image,
        "ImageFilename": f"{raw_number}-{variant}",
        "type": "Set",
    }
    return parsed


def parse_minifig_article(article_html: str) -> Optional[Dict[str, Any]]:
    meta_link_match = re.search(
        r"<div\b[^>]*class=['\"][^'\"]*\bmeta\b[^'\"]*['\"][^>]*>\s*<h1>\s*<a\b[^>]*href=['\"]([^'\"]+)['\"]",
        article_html,
        re.IGNORECASE | re.DOTALL,
    )
    if not meta_link_match:
        meta_link_match = re.search(
            r"<h1>\s*<a\b[^>]*href=['\"]([^'\"]+)['\"]",
            article_html,
            re.IGNORECASE | re.DOTALL,
        )
    if not meta_link_match:
        return None

    link = to_absolute_url(meta_link_match.group(1))
    code_match = re.search(r"/minifigs/([a-z0-9]+)", link, re.IGNORECASE)
    if not code_match:
        return None
    number = code_match.group(1).lower()

    name_match = re.search(
        r"<div\b[^>]*class=['\"][^'\"]*\bmeta\b[^'\"]*['\"][^>]*>\s*<h1>\s*<a\b[^>]*>(.*?)</a>\s*</h1>",
        article_html,
        re.IGNORECASE | re.DOTALL,
    )
    if not name_match:
        name_match = re.search(
            r"<h1>\s*<a\b[^>]*>(.*?)</a>\s*</h1>",
            article_html,
            re.IGNORECASE | re.DOTALL,
        )
    minifig_name_raw = strip_tags(name_match.group(1)) if name_match else ""
    minifig_name = collapse_ws(minifig_name_raw)
    if minifig_name.lower().startswith(number):
        trimmed = minifig_name[len(number):].lstrip(" :-")
        if trimmed:
            minifig_name = collapse_ws(trimmed)
    if not minifig_name:
        return None

    character_match = re.search(
        r"<a\b[^>]*class=['\"][^'\"]*\bname\b[^'\"]*['\"][^>]*>(.*?)</a>",
        article_html,
        re.IGNORECASE | re.DOTALL,
    )
    character_name = strip_tags(character_match.group(1)) if character_match else ""

    year_match = re.search(r"<a\b[^>]*class=['\"][^'\"]*\byear\b[^'\"]*['\"][^>]*>([^<]+)</a>", article_html, re.IGNORECASE)
    year = parse_int(year_match.group(1)) if year_match else None
    if year == 0:
        year = None

    dt_map = extract_dt_map(article_html)

    appears_text = strip_tags(first_from_map(dt_map, "appears in") or "")
    in_sets = parse_int(appears_text)
    if in_sets is None:
        in_sets = 0

    new_value = extract_price_from_html(first_from_map(dt_map, "value new"))
    used_value = extract_price_from_html(first_from_map(dt_map, "value used"))

    image_match = re.search(r"<img\s+[^>]*src=\"([^\"]+)\"", article_html)
    product_image = strip_query(to_absolute_url(image_match.group(1) if image_match else ""))

    parsed: Dict[str, Any] = {
        "Number": number,
        "Minifig name": minifig_name,
        "Character name": character_name,
        "Category": "Star Wars",
        "Year": str(year) if year is not None else "",
        "In sets": str(in_sets),
        "New": new_value,
        "Used": used_value,
        "link": link,
        "instructionsLink": "",
        "productImage": product_image,
        "type": "Minifigure",
    }
    return parsed


def fetch_html(session: requests.Session, url: str, cfg: FetchConfig, *, source: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Cache-Control": "no-cache",
        "Referer": "https://brickset.com/",
    }

    attempts = max(1, cfg.retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, headers=headers, timeout=cfg.timeout_seconds)
        except requests.RequestException as exc:
            if attempt == attempts:
                raise RuntimeError(f"{source}: request failed for {url}: {exc}") from exc
            sleep_for = min(8.0, 1.2 * attempt)
            time.sleep(sleep_for)
            continue

        if response.status_code == 429:
            if attempt == attempts:
                raise RuntimeError(f"{source}: HTTP 429 for {url}")
            retry_after = response.headers.get("Retry-After")
            wait = parse_int(retry_after) if retry_after else None
            if wait is None:
                wait_seconds = min(180.0, max(10.0, attempt * 20.0))
            else:
                wait_seconds = float(max(1, wait))
            log(f"[{source}] HTTP 429 for {url}; waiting {wait_seconds:.0f}s", enabled=cfg.verbose)
            time.sleep(wait_seconds)
            continue

        if response.status_code >= 400:
            raise RuntimeError(f"{source}: HTTP {response.status_code} for {url}")

        response.encoding = response.encoding or "utf-8"
        return response.text

    raise RuntimeError(f"{source}: unreachable failure for {url}")


def fetch_brickset_api(
    session: requests.Session,
    cfg: FetchConfig,
    *,
    api_base_url: str,
    api_key: str,
    method: str,
    user_hash: str,
    method_params: Dict[str, Any],
) -> Dict[str, Any]:
    base = api_base_url.rstrip("/")
    url = f"{base}/{method}"
    params_json = json.dumps(method_params, separators=(",", ":"))
    query = {
        "apiKey": api_key,
        # Brickset v3 methods are defined with userHash in the signature.
        # Sending an empty userHash is valid for unauthenticated catalog queries.
        "userHash": user_hash,
        "params": params_json,
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-GB,en;q=0.9",
        "Cache-Control": "no-cache",
        "Referer": "https://brickset.com/",
    }

    attempts = max(1, cfg.retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, params=query, headers=headers, timeout=cfg.timeout_seconds)
        except requests.RequestException as exc:
            if attempt == attempts:
                raise RuntimeError(f"api:{method}: request failed: {exc}") from exc
            time.sleep(min(10.0, 1.5 * attempt))
            continue

        if response.status_code >= 500:
            try:
                post_response = session.post(url, data=query, headers=headers, timeout=cfg.timeout_seconds)
                if post_response.status_code < 500:
                    response = post_response
                else:
                    log(
                        (
                            f"[API] GET {method} returned HTTP {response.status_code}; "
                            f"POST fallback returned HTTP {post_response.status_code}"
                        ),
                        enabled=cfg.verbose,
                    )
                    response = post_response
            except requests.RequestException as exc:
                if attempt == attempts:
                    raise RuntimeError(f"api:{method}: request failed: {exc}") from exc
                wait_seconds = min(10.0, 1.5 * attempt)
                time.sleep(wait_seconds)
                continue

        if response.status_code == 429:
            retry_after = parse_int(response.headers.get("Retry-After", ""))
            wait_seconds = float(retry_after if retry_after is not None else max(15, attempt * 20))
            if attempt == attempts:
                raise RuntimeError(f"api:{method}: HTTP 429")
            log(f"[API] HTTP 429 for {method}; waiting {wait_seconds:.0f}s", enabled=cfg.verbose)
            time.sleep(wait_seconds)
            continue

        if response.status_code >= 500:
            if attempt == attempts:
                raise RuntimeError(f"api:{method}: HTTP {response.status_code}")
            wait_seconds = float(min(240.0, max(15.0, attempt * 25.0)))
            log(
                f"[API] HTTP {response.status_code} for {method}; waiting {wait_seconds:.0f}s before retry",
                enabled=cfg.verbose,
            )
            time.sleep(wait_seconds)
            continue

        if response.status_code >= 400:
            raise RuntimeError(f"api:{method}: HTTP {response.status_code}")

        try:
            payload = parse_api_payload_text(response.text)
        except Exception as exc:
            if attempt == attempts:
                raise RuntimeError(f"api:{method}: invalid JSON payload: {exc}") from exc
            time.sleep(min(8.0, 1.5 * attempt))
            continue

        status = collapse_ws(str(payload.get("status") or "")).lower()
        if status and status not in {"success", "ok"}:
            message = collapse_ws(str(payload.get("message") or "unknown API error"))
            raise RuntimeError(f"api:{method}: {message}")

        return payload

    raise RuntimeError(f"api:{method}: unreachable failure")


def parse_set_from_api(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    number_value = api_pick(raw, "number")
    if number_value is None:
        return None
    number_text = collapse_ws(str(number_value))
    if not number_text:
        return None

    variant = parse_int(api_pick(raw, "numberVariant", "variant")) or 1
    number_out: Any = int(number_text) if number_text.isdigit() else number_text.upper()

    set_name = collapse_ws(str(api_pick(raw, "name", "setName") or ""))
    if not set_name:
        return None

    theme = collapse_ws(str(api_pick(raw, "theme") or "")) or "Unknown"
    set_link = to_absolute_url(str(api_pick(raw, "bricksetURL", "setURL", "url") or ""))
    image_url = to_absolute_url(
        str(
            api_pick(
                raw,
                "imageURL",
                "largeThumbnailURL",
                "thumbnailURL",
                "image",
            )
            or ""
        )
    )

    parsed: Dict[str, Any] = {
        "Number": number_out,
        "Variant": variant,
        "SetName": set_name,
        "YearFrom": parse_int(api_pick(raw, "year", "yearFrom")),
        "Subtheme": api_pick(raw, "subtheme"),
        "Theme": theme,
        "ThemeGroup": api_pick(raw, "themeGroup"),
        "Category": api_pick(raw, "category"),
        "Released": parse_bool(api_pick(raw, "released")),
        "Pieces": parse_int(api_pick(raw, "pieces")),
        "Minifigs": parse_int(api_pick(raw, "minifigs")),
        "SetID": parse_int(api_pick(raw, "setID", "setId")),
        "OwnCount": parse_int(api_pick(raw, "ownedBy", "ownCount")),
        "WantCount": parse_int(api_pick(raw, "wantedBy", "wantCount")),
        "Rating": api_pick(raw, "rating"),
        "Availability": api_pick(raw, "availability"),
        "PackagingType": api_pick(raw, "packagingType"),
        "AdditionalImageCount": parse_int(api_pick(raw, "additionalImageCount")),
        "USRetailPrice": api_pick(raw, "USRetailPrice"),
        "UKRetailPrice": api_pick(raw, "UKRetailPrice"),
        "CARetailPrice": api_pick(raw, "CARetailPrice"),
        "DERetailPrice": api_pick(raw, "DERetailPrice"),
        "USDateAdded": format_catalog_date(api_pick(raw, "USDateAdded")),
        "USDateRemoved": format_catalog_date(api_pick(raw, "USDateRemoved")),
        "LaunchDate": format_catalog_date(api_pick(raw, "LaunchDate")),
        "ExitDate": format_catalog_date(api_pick(raw, "ExitDate")),
        "EAN": api_pick(raw, "EAN"),
        "UPC": api_pick(raw, "UPC"),
        "USItemNumber": api_pick(raw, "USItemNumber"),
        "EUItemNumber": api_pick(raw, "EUItemNumber"),
        "Weight": api_pick(raw, "weight"),
        "Height": api_pick(raw, "height"),
        "Width": api_pick(raw, "width"),
        "Depth": api_pick(raw, "depth"),
        "InstructionsCount": parse_int(api_pick(raw, "instructionsCount")),
        "AgeMin": parse_int(api_pick(raw, "ageMin")),
        "AgeMax": parse_int(api_pick(raw, "ageMax")),
        "ModelDimension1": api_pick(raw, "modelDimension1"),
        "ModelDimension2": api_pick(raw, "modelDimension2"),
        "ModelDimension3": api_pick(raw, "modelDimension3"),
        "Designers": api_pick(raw, "designers"),
        "link": set_link,
        "instructionsLink": "",
        "productImage": image_url,
        "ImageFilename": f"{number_text}-{variant}",
        "type": "Set",
    }
    return parsed


def crawl_sets_via_api(
    session: requests.Session,
    cfg: FetchConfig,
    *,
    api_base_url: str,
    api_key: str,
    page_size: int,
    max_pages: Optional[int],
    theme_filter: Optional[str],
) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], CrawlStats]:
    stats = CrawlStats()
    records: Dict[Tuple[str, int], Dict[str, Any]] = {}
    effective_page_size = max(50, min(500, page_size))

    page = 1
    matches: Optional[int] = None
    while True:
        page_attempt = 0
        payload: Optional[Dict[str, Any]] = None
        while True:
            page_attempt += 1
            request_params: Dict[str, Any] = {
                "pageSize": effective_page_size,
                "pageNumber": page,
            }
            if theme_filter:
                request_params["theme"] = theme_filter

            try:
                payload = fetch_brickset_api(
                    session,
                    cfg,
                    api_base_url=api_base_url,
                    api_key=api_key,
                    method="getSets",
                    user_hash="",
                    method_params=request_params,
                )
                stats.pages_fetched += 1
                break
            except RuntimeError as exc:
                msg = str(exc)
                retryable_5xx = any(code in msg for code in ("HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504"))
                if retryable_5xx and effective_page_size > 50:
                    next_page_size = max(50, effective_page_size // 2)
                    if next_page_size < effective_page_size:
                        log(
                            (
                                f"[API Sets] page {page}: {msg}; "
                                f"reducing pageSize {effective_page_size}->{next_page_size} and retrying"
                            ),
                            enabled=True,
                        )
                        effective_page_size = next_page_size
                        continue
                if retryable_5xx and page_attempt < 4:
                    wait_seconds = float(min(240.0, max(20.0, page_attempt * 30.0)))
                    log(
                        f"[API Sets] page {page}: {msg}; retrying in {wait_seconds:.0f}s",
                        enabled=True,
                    )
                    time.sleep(wait_seconds)
                    continue
                raise

        if payload is None:
            raise RuntimeError(f"api:getSets: page {page} failed to return payload")

        if matches is None:
            matches = parse_int(payload.get("matches"))

        api_sets = payload.get("sets")
        if not isinstance(api_sets, list):
            raise RuntimeError("api:getSets: response missing sets array")

        if not api_sets:
            break

        stats.articles_parsed += len(api_sets)
        page_added = 0
        for raw in api_sets:
            if not isinstance(raw, dict):
                continue
            parsed = parse_set_from_api(raw)
            if not parsed:
                continue
            key = set_key(parsed["Number"], parsed["Variant"])
            if key not in records:
                page_added += 1
            records[key] = parsed
            stats.records_parsed += 1

        log(
            (
                f"[API Sets] page {page}: parsed={len(api_sets)} unique_added={page_added} "
                f"total_unique={len(records)} pageSize={effective_page_size}"
            ),
            enabled=cfg.verbose,
        )

        if max_pages is not None and page >= max_pages:
            break
        if matches is not None and len(records) >= matches:
            break

        page += 1
        maybe_sleep(cfg.page_delay_seconds, cfg.page_jitter_seconds)

    if not records:
        raise RuntimeError("api:getSets: no set records parsed")
    return records, stats


def discover_total_pages(html: str, base_url: str) -> int:
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    escaped = re.escape(path)
    page_re = re.compile(PAGE_NUMBER_RE_TEMPLATE.format(path=escaped), re.IGNORECASE)
    matches = [int(m) for m in page_re.findall(html)]
    return max(matches) if matches else 1


def crawl_sets(session: requests.Session, cfg: FetchConfig, base_url: str, max_pages: Optional[int]) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], CrawlStats]:
    stats = CrawlStats()
    first_page_html = fetch_html(session, base_url, cfg, source="sets")
    stats.pages_fetched += 1

    total_pages = discover_total_pages(first_page_html, base_url)
    if max_pages is not None:
        total_pages = min(total_pages, max(1, max_pages))

    records: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for page in range(1, total_pages + 1):
        article_blocks: List[str] = []
        attempts = 5
        saw_soft_block = False

        for page_attempt in range(1, attempts + 1):
            if page == 1 and page_attempt == 1:
                html = first_page_html
            else:
                page_url = f"{base_url}/page-{page}"
                html = fetch_html(session, page_url, cfg, source="sets")
                stats.pages_fetched += 1
                maybe_sleep(cfg.page_delay_seconds, cfg.page_jitter_seconds)

            article_blocks = extract_article_blocks(html, mode="sets")
            if article_blocks:
                break

            if looks_like_soft_block(html):
                saw_soft_block = True
                cfg.page_delay_seconds = min(4.0, max(cfg.page_delay_seconds, 2.0))
                cfg.page_jitter_seconds = max(cfg.page_jitter_seconds, 0.5)
                wait_seconds = min(300.0, max(45.0, page_attempt * 45.0))
                title = extract_html_title(html)
                if title:
                    log(
                        f"[Sets] page {page}/{total_pages}: soft block detected ({title}); waiting {wait_seconds:.0f}s",
                        enabled=True,
                    )
                else:
                    log(
                        f"[Sets] page {page}/{total_pages}: soft block detected; waiting {wait_seconds:.0f}s",
                        enabled=True,
                    )
                try:
                    session.cookies.clear()
                except Exception:
                    pass
                time.sleep(wait_seconds)
                continue

            if page_attempt < attempts:
                wait_seconds = 30.0 * page_attempt if not saw_soft_block else min(240.0, 60.0 * page_attempt)
                log(
                    f"[Sets] page {page}/{total_pages}: no articles, retrying in {wait_seconds:.0f}s",
                    enabled=cfg.verbose,
                )
                time.sleep(wait_seconds)

        if not article_blocks:
            stats.failed_pages += 1
            cfg.page_delay_seconds = min(4.0, max(cfg.page_delay_seconds, 2.5))
            cfg.page_jitter_seconds = max(cfg.page_jitter_seconds, 0.6)
            try:
                first_page_html = fetch_html(session, base_url, cfg, source="sets")
                stats.pages_fetched += 1
                maybe_sleep(cfg.page_delay_seconds, cfg.page_jitter_seconds)
            except Exception as exc:
                log(f"[Sets] page {page}/{total_pages}: base refresh failed: {exc}", enabled=True)
            log(
                f"[Sets] page {page}/{total_pages}: skipping after retries (no articles).",
                enabled=True,
            )
            continue

        stats.articles_parsed += len(article_blocks)

        for article_html in article_blocks:
            parsed = parse_set_article(article_html)
            if not parsed:
                continue
            key = set_key(parsed["Number"], parsed["Variant"])
            records[key] = parsed
            stats.records_parsed += 1

        log(f"[Sets] page {page}/{total_pages}: parsed {len(article_blocks)} articles", enabled=cfg.verbose)

    if not records:
        raise RuntimeError("sets: no set records parsed from Brickset")

    return records, stats


def crawl_minifigs(session: requests.Session, cfg: FetchConfig, base_url: str, max_pages: Optional[int]) -> Tuple[Dict[str, Dict[str, Any]], CrawlStats]:
    stats = CrawlStats()
    first_page_html = fetch_html(session, base_url, cfg, source="minifigs")
    stats.pages_fetched += 1

    total_pages = discover_total_pages(first_page_html, base_url)
    if max_pages is not None:
        total_pages = min(total_pages, max(1, max_pages))

    records: Dict[str, Dict[str, Any]] = {}

    for page in range(1, total_pages + 1):
        article_blocks: List[str] = []
        attempts = 5
        saw_soft_block = False

        for page_attempt in range(1, attempts + 1):
            if page == 1 and page_attempt == 1:
                html = first_page_html
            else:
                page_url = f"{base_url}/page-{page}"
                html = fetch_html(session, page_url, cfg, source="minifigs")
                stats.pages_fetched += 1
                maybe_sleep(cfg.page_delay_seconds, cfg.page_jitter_seconds)

            article_blocks = extract_article_blocks(html, mode="minifigs")
            if article_blocks:
                break

            if looks_like_soft_block(html):
                saw_soft_block = True
                cfg.page_delay_seconds = min(4.0, max(cfg.page_delay_seconds, 2.0))
                cfg.page_jitter_seconds = max(cfg.page_jitter_seconds, 0.5)
                wait_seconds = min(300.0, max(45.0, page_attempt * 45.0))
                title = extract_html_title(html)
                if title:
                    log(
                        f"[Minifigs] page {page}/{total_pages}: soft block detected ({title}); waiting {wait_seconds:.0f}s",
                        enabled=True,
                    )
                else:
                    log(
                        f"[Minifigs] page {page}/{total_pages}: soft block detected; waiting {wait_seconds:.0f}s",
                        enabled=True,
                    )
                try:
                    session.cookies.clear()
                except Exception:
                    pass
                time.sleep(wait_seconds)
                continue

            if page_attempt < attempts:
                wait_seconds = 30.0 * page_attempt if not saw_soft_block else min(240.0, 60.0 * page_attempt)
                log(
                    f"[Minifigs] page {page}/{total_pages}: no articles, retrying in {wait_seconds:.0f}s",
                    enabled=cfg.verbose,
                )
                time.sleep(wait_seconds)

        if not article_blocks:
            stats.failed_pages += 1
            cfg.page_delay_seconds = min(4.0, max(cfg.page_delay_seconds, 2.5))
            cfg.page_jitter_seconds = max(cfg.page_jitter_seconds, 0.6)
            try:
                first_page_html = fetch_html(session, base_url, cfg, source="minifigs")
                stats.pages_fetched += 1
                maybe_sleep(cfg.page_delay_seconds, cfg.page_jitter_seconds)
            except Exception as exc:
                log(f"[Minifigs] page {page}/{total_pages}: base refresh failed: {exc}", enabled=True)
            log(
                f"[Minifigs] page {page}/{total_pages}: skipping after retries (no articles).",
                enabled=True,
            )
            continue

        stats.articles_parsed += len(article_blocks)

        for article_html in article_blocks:
            parsed = parse_minifig_article(article_html)
            if not parsed:
                continue
            key = minifig_key(parsed["Number"])
            records[key] = parsed
            stats.records_parsed += 1

        log(f"[Minifigs] page {page}/{total_pages}: parsed {len(article_blocks)} articles", enabled=cfg.verbose)

    if not records:
        raise RuntimeError("minifigs: no minifigure records parsed from Brickset")

    return records, stats


def ordered_columns(rows: List[Dict[str, Any]]) -> List[str]:
    columns: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)
    return columns


def merge_set_rows(existing_rows: List[Dict[str, Any]], scraped_by_key: Dict[Tuple[str, int], Dict[str, Any]]) -> List[Dict[str, Any]]:
    columns = ordered_columns(existing_rows)
    for required in ("New", "Used", "link", "instructionsLink", "productImage", "type"):
        if required not in columns:
            columns.append(required)

    template: Dict[str, Any] = {key: None for key in columns}
    template.update({
        "Category": "Normal",
        "Theme": "Unknown",
        "ThemeGroup": "Unknown",
        "Image": "X",
        "instructionsLink": "",
        "link": "",
        "productImage": "",
        "type": "Set",
    })

    existing_by_key: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in existing_rows:
        key = set_key(row.get("Number"), row.get("Variant"))
        existing_by_key.setdefault(key, []).append(row)

    next_set_id = 1
    for row in existing_rows:
        current = parse_int(row.get("SetID"))
        if current is not None:
            next_set_id = max(next_set_id, current + 1)

    seen: set[Tuple[str, int]] = set()
    merged: List[Dict[str, Any]] = []

    for key in sorted(scraped_by_key.keys()):
        scraped = scraped_by_key[key]
        existing_list = existing_by_key.get(key) or []
        existing = existing_list.pop(0) if existing_list else None

        row = dict(template)
        if existing:
            row.update(existing)

        for field, value in scraped.items():
            if value is None:
                continue
            if isinstance(value, str) and value == "" and field not in {"link", "instructionsLink", "productImage"}:
                continue
            row[field] = value

        if parse_int(row.get("SetID")) is None:
            row["SetID"] = next_set_id
            next_set_id += 1

        row["instructionsLink"] = collapse_ws(str(row.get("instructionsLink") or ""))
        row["link"] = collapse_ws(str(row.get("link") or ""))
        row["productImage"] = collapse_ws(str(row.get("productImage") or ""))
        row["type"] = "Set"
        row["Theme"] = row.get("Theme") or "Unknown"
        row["ThemeGroup"] = row.get("ThemeGroup") or "Unknown"
        row["Category"] = row.get("Category") or "Normal"

        seen.add(key)
        merged.append(row)

    for existing_list in existing_by_key.values():
        for existing in existing_list:
            row = dict(template)
            row.update(existing)
            row["instructionsLink"] = collapse_ws(str(row.get("instructionsLink") or ""))
            row["link"] = collapse_ws(str(row.get("link") or ""))
            row["productImage"] = collapse_ws(str(row.get("productImage") or ""))
            row["type"] = "Set"
            merged.append(row)

    def sort_key(row: Dict[str, Any]) -> Tuple[int, int, str, int]:
        year = parse_int(row.get("YearFrom"))
        if year is None:
            year = 9999

        number_raw = collapse_ws(str(row.get("Number") or "")).upper()
        number_is_numeric = 1 if number_raw.isdigit() else 2
        number_sort = str(int(number_raw)) if number_raw.isdigit() else number_raw
        variant = parse_int(row.get("Variant")) or 1
        return (year, number_is_numeric, number_sort, variant)

    merged.sort(key=sort_key)
    return merged


def merge_minifig_rows(existing_rows: List[Dict[str, Any]], scraped_by_key: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    columns = ordered_columns(existing_rows)
    for required in ("Number", "Minifig name", "Character name", "Category", "Year", "In sets", "New", "Used", "link", "instructionsLink", "productImage", "type"):
        if required not in columns:
            columns.append(required)

    template: Dict[str, Any] = {key: None for key in columns}
    template.update({
        "Category": "Star Wars",
        "Year": "",
        "In sets": "0",
        "link": "",
        "instructionsLink": "",
        "productImage": "",
        "type": "Minifigure",
    })

    existing_by_key: Dict[str, List[Dict[str, Any]]] = {}
    for row in existing_rows:
        key = minifig_key(row.get("Number"))
        existing_by_key.setdefault(key, []).append(row)

    seen: set[str] = set()
    merged: List[Dict[str, Any]] = []

    for key in sorted(scraped_by_key.keys()):
        scraped = scraped_by_key[key]
        existing_list = existing_by_key.get(key) or []
        existing = existing_list.pop(0) if existing_list else None

        row = dict(template)
        if existing:
            row.update(existing)

        for field, value in scraped.items():
            if value is None:
                continue
            if isinstance(value, str) and value == "" and field not in {"Character name", "Year", "In sets", "link", "instructionsLink", "productImage"}:
                continue
            row[field] = value

        row["Number"] = collapse_ws(str(row.get("Number") or "")).lower()
        row["Category"] = row.get("Category") or "Star Wars"
        row["Year"] = collapse_ws(str(row.get("Year") or ""))
        row["In sets"] = collapse_ws(str(row.get("In sets") or "0"))
        row["link"] = collapse_ws(str(row.get("link") or ""))
        row["instructionsLink"] = collapse_ws(str(row.get("instructionsLink") or ""))
        row["productImage"] = collapse_ws(str(row.get("productImage") or ""))
        row["type"] = "Minifigure"

        seen.add(key)
        merged.append(row)

    for existing_list in existing_by_key.values():
        for existing in existing_list:
            row = dict(template)
            row.update(existing)
            row["Number"] = collapse_ws(str(row.get("Number") or "")).lower()
            row["Category"] = row.get("Category") or "Star Wars"
            row["Year"] = collapse_ws(str(row.get("Year") or ""))
            row["In sets"] = collapse_ws(str(row.get("In sets") or "0"))
            row["link"] = collapse_ws(str(row.get("link") or ""))
            row["instructionsLink"] = collapse_ws(str(row.get("instructionsLink") or ""))
            row["productImage"] = collapse_ws(str(row.get("productImage") or ""))
            row["type"] = "Minifigure"
            merged.append(row)

    merged.sort(key=lambda row: minifig_key(row.get("Number")))
    return merged


def fetch_bricklink_multiplicity(
    session: requests.Session,
    number: Any,
    variant: Any,
    cfg: FetchConfig,
) -> Tuple[Optional[Dict[str, int]], Optional[str]]:
    set_number = collapse_ws(str(number)).upper()
    set_variant = parse_int(variant) or 1
    set_code = f"{set_number}-{set_variant}"
    url = BRICKLINK_INVENTORY_URL_TEMPLATE.format(set_code=set_code)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.bricklink.com/",
    }

    attempts = max(1, cfg.retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, headers=headers, timeout=cfg.timeout_seconds)
        except requests.RequestException as exc:
            if attempt == attempts:
                return (None, f"request error: {exc}")
            time.sleep(min(8.0, 1.4 * attempt))
            continue

        if response.status_code == 429:
            if attempt == attempts:
                return (None, "rate-limited (HTTP 429)")
            retry_after = parse_int(response.headers.get("Retry-After", ""))
            time.sleep(float(retry_after if retry_after is not None else max(2, attempt * 2)))
            continue

        if response.status_code >= 400:
            return (None, f"http {response.status_code}")

        html = response.text
        matches = BRICKLINK_ROW_RE.findall(html)
        if not matches:
            return ({}, "no minifigure rows parsed")

        counts: Dict[str, int] = {}
        for code, quantity in matches:
            key = code.lower()
            counts[key] = counts.get(key, 0) + int(quantity)
        return (counts, None)

    return (None, "unreachable")


def apply_set_minifigure_multiplicity(
    session: requests.Session,
    set_rows: List[Dict[str, Any]],
    cfg: FetchConfig,
    max_fetches: Optional[int] = None,
    *,
    include_rows_without_minifig_numbers: bool = False,
) -> MultiplicityStats:
    stats = MultiplicityStats()

    for row in set_rows:
        base_list = parse_minifig_numbers(row.get("MinifigNumbers"))
        unique_list = unique_preserving_order(base_list)
        minifigs_total = parse_int(row.get("Minifigs")) or 0

        if minifigs_total <= 0:
            continue

        if not unique_list and minifigs_total <= 0:
            continue

        should_fetch = minifigs_total > len(unique_list) and bool(unique_list)
        if include_rows_without_minifig_numbers and (not unique_list and minifigs_total > 0):
            should_fetch = True
        if not should_fetch:
            continue

        if max_fetches is not None and stats.fetched >= max_fetches:
            break

        stats.considered += 1

        counts, warning = fetch_bricklink_multiplicity(session, row.get("Number"), row.get("Variant"), cfg)
        stats.fetched += 1
        maybe_sleep(cfg.bricklink_delay_seconds, cfg.bricklink_jitter_seconds)

        if counts is None:
            stats.failures += 1
            log(
                f"[Multiplicity] warn: {row.get('Number')}-{row.get('Variant')}: {warning}",
                enabled=cfg.verbose,
            )
            continue

        codes_in_order: List[str]
        if unique_list:
            codes_in_order = list(unique_list)
        else:
            codes_in_order = sorted(counts.keys())

        seen = {code.lower() for code in codes_in_order}
        for extra in sorted(counts.keys()):
            if extra not in seen:
                codes_in_order.append(extra)

        expanded: List[str] = []
        for code in codes_in_order:
            qty = max(1, counts.get(code.lower(), 1))
            expanded.extend([code.lower()] * qty)

        if not expanded:
            continue

        updated_value = minifig_numbers_string(expanded)
        if row.get("MinifigNumbers") != updated_value:
            row["MinifigNumbers"] = updated_value
            stats.updated += 1

        if parse_int(row.get("Minifigs")) in (None, 0):
            row["Minifigs"] = len(expanded)

        if warning:
            log(
                f"[Multiplicity] note: {row.get('Number')}-{row.get('Variant')}: {warning}",
                enabled=cfg.verbose,
            )

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


def write_json_array(path: Path, rows: List[Dict[str, Any]]) -> None:
    text = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def build_theme_index(set_rows: List[Dict[str, Any]], minifig_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    set_counts: Dict[str, int] = {}
    for row in set_rows:
        theme = collapse_ws(str(row.get("Theme") or "Unknown"))
        if not theme:
            theme = "Unknown"
        set_counts[theme] = set_counts.get(theme, 0) + 1

    minifig_counts: Dict[str, int] = {}
    for row in minifig_rows:
        raw_theme = row.get("Theme") or row.get("Category") or "Unknown"
        theme = collapse_ws(str(raw_theme))
        if not theme:
            theme = "Unknown"
        minifig_counts[theme] = minifig_counts.get(theme, 0) + 1

    all_themes = sorted(set(set_counts.keys()) | set(minifig_counts.keys()), key=lambda t: t.lower())
    return [
        {
            "Theme": theme,
            "SetCount": set_counts.get(theme, 0),
            "MinifigCount": minifig_counts.get(theme, 0),
        }
        for theme in all_themes
    ]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync Brickset catalogs to JSON (API-first for sets).")
    parser.add_argument("--sets-json", default="dist/Lego Star Wars Database.json", help="Sets JSON output path.")
    parser.add_argument(
        "--minifigs-json",
        default="dist/Lego-Star-Wars-Minifigure-Database.json",
        help="Minifigures JSON output path.",
    )
    parser.add_argument(
        "--themes-json",
        default="dist/Themes.json",
        help="Theme index JSON output path.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Brickset API key. Falls back to BRICKSET_API_KEY env var.",
    )
    parser.add_argument(
        "--api-base-url",
        default=BRICKSET_API_BASE_URL,
        help="Brickset API base URL.",
    )
    parser.add_argument(
        "--theme",
        default=None,
        help="Optional theme filter for API getSets (e.g. 'Star Wars'). Default: all themes.",
    )
    parser.add_argument(
        "--api-page-size",
        type=int,
        default=500,
        help="Page size for API getSets calls (max 500).",
    )
    parser.add_argument(
        "--sets-url",
        default=SETS_BASE_URL,
        help="Fallback Brickset URL for web-crawled sets (legacy/debug).",
    )
    parser.add_argument(
        "--minifigs-url",
        default=MINIFIGS_BASE_URL,
        help="Brickset URL for web-crawled minifigures.",
    )
    parser.add_argument("--timeout", type=float, default=25.0, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=4, help="Retry count for network requests.")
    parser.add_argument("--page-delay", type=float, default=0.75, help="Delay between Brickset page requests.")
    parser.add_argument("--page-jitter", type=float, default=0.20, help="Random jitter added to page delay.")
    parser.add_argument("--bricklink-delay", type=float, default=1.50, help="Delay between BrickLink multiplicity requests.")
    parser.add_argument("--bricklink-jitter", type=float, default=0.35, help="Random jitter added to BrickLink delay.")
    parser.add_argument("--max-set-pages", type=int, default=None, help="Optional cap for set pages (debug).")
    parser.add_argument("--max-minifig-pages", type=int, default=None, help="Optional cap for minifig pages (debug).")
    parser.add_argument(
        "--crawl-minifigs",
        action="store_true",
        help="Crawl minifig pages from web. Default keeps existing minifig JSON unchanged.",
    )
    parser.add_argument("--skip-multiplicity", action="store_true", help="Skip BrickLink multiplicity enrichment.")
    parser.add_argument(
        "--multiplicity-include-empty",
        action="store_true",
        help="Also fetch BrickLink multiplicity for rows without MinifigNumbers (very expensive).",
    )
    parser.add_argument(
        "--max-multiplicity-fetches",
        type=int,
        default=None,
        help="Optional cap for BrickLink multiplicity fetches (debug).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write files.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    print(f"[Sync] script_version={SCRIPT_VERSION}", flush=True)

    sets_path = Path(args.sets_json)
    minifigs_path = Path(args.minifigs_json)
    themes_path = Path(args.themes_json)

    if not sets_path.exists():
        print(f"Missing sets JSON: {sets_path}", file=sys.stderr)
        return 1
    if not minifigs_path.exists():
        print(f"Missing minifigs JSON: {minifigs_path}", file=sys.stderr)
        return 1

    cfg = FetchConfig(
        timeout_seconds=max(5.0, args.timeout),
        retries=max(0, args.retries),
        page_delay_seconds=max(0.0, args.page_delay),
        page_jitter_seconds=max(0.0, args.page_jitter),
        bricklink_delay_seconds=max(0.0, args.bricklink_delay),
        bricklink_jitter_seconds=max(0.0, args.bricklink_jitter),
        verbose=args.verbose,
    )

    api_key = collapse_ws(str(args.api_key or os.getenv("BRICKSET_API_KEY") or ""))
    if not api_key:
        print("Missing Brickset API key. Provide --api-key or set BRICKSET_API_KEY.", file=sys.stderr)
        return 1

    existing_sets = load_json_array(sets_path)
    existing_minifigs = load_json_array(minifigs_path)

    log(
        f"[Start] existing sets={len(existing_sets)} minifigs={len(existing_minifigs)}",
        enabled=cfg.verbose,
    )

    session = requests.Session()
    try:
        scraped_sets_by_key, set_stats = crawl_sets_via_api(
            session,
            cfg,
            api_base_url=args.api_base_url,
            api_key=api_key,
            page_size=max(50, min(500, args.api_page_size)),
            max_pages=args.max_set_pages,
            theme_filter=collapse_ws(str(args.theme or "")) or None,
        )
    except Exception as exc:
        print(
            (
                f"[Sets] API sync failed ({exc}). "
                "Proceeding with existing set catalog for this run."
            ),
            flush=True,
        )
        scraped_sets_by_key = {}
        set_stats = CrawlStats(failed_pages=1)

    if args.crawl_minifigs:
        scraped_minifigs_by_key, minifig_stats = crawl_minifigs(session, cfg, args.minifigs_url, args.max_minifig_pages)
    else:
        scraped_minifigs_by_key = {}
        minifig_stats = CrawlStats()
        log("[Minifigs] crawl skipped; preserving existing minifigure JSON.", enabled=cfg.verbose)

    merged_sets = merge_set_rows(existing_sets, scraped_sets_by_key)
    merged_minifigs = merge_minifig_rows(existing_minifigs, scraped_minifigs_by_key)
    themes_index = build_theme_index(merged_sets, merged_minifigs)

    multiplicity_stats = MultiplicityStats()
    if not args.skip_multiplicity:
        multiplicity_stats = apply_set_minifigure_multiplicity(
            session,
            merged_sets,
            cfg,
            max_fetches=max(0, args.max_multiplicity_fetches) if args.max_multiplicity_fetches is not None else None,
            include_rows_without_minifig_numbers=bool(args.multiplicity_include_empty),
        )

    if args.dry_run:
        print("[Dry run] No files written.")
    else:
        write_json_array(sets_path, merged_sets)
        write_json_array(minifigs_path, merged_minifigs)
        write_json_array(themes_path, themes_index)

    print(
        (
            f"[Sets] pages={set_stats.pages_fetched} articles={set_stats.articles_parsed} "
            f"parsed={set_stats.records_parsed} failed_pages={set_stats.failed_pages} merged={len(merged_sets)}"
        ),
        flush=True,
    )
    print(
        (
            f"[Minifigs] pages={minifig_stats.pages_fetched} articles={minifig_stats.articles_parsed} "
            f"parsed={minifig_stats.records_parsed} failed_pages={minifig_stats.failed_pages} merged={len(merged_minifigs)}"
        ),
        flush=True,
    )
    print(
        (
            f"[Multiplicity] considered={multiplicity_stats.considered} fetched={multiplicity_stats.fetched} "
            f"updated={multiplicity_stats.updated} failures={multiplicity_stats.failures}"
        ),
        flush=True,
    )
    print(f"[Themes] total={len(themes_index)}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
