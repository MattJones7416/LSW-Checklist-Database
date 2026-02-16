#!/usr/bin/env python3
"""Update market prices in LEGO Star Wars JSON files by scraping each item's link.

This script updates/creates the `New` and `Used` fields using values found on each
linked page (typically Brickset).
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
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

PRICE_TOKEN_PATTERN = (
    r"(?:[~≈]?\s*(?:£|\$|€)\s*[0-9][0-9,]*(?:\.[0-9]{1,2})?"
    r"|(?:USD|GBP|EUR)\s*[0-9][0-9,]*(?:\.[0-9]{1,2})"
    r"|[0-9][0-9,]*(?:\.[0-9]{1,2}))"
)
PRICE_TOKEN_RE = re.compile(PRICE_TOKEN_PATTERN, re.IGNORECASE)


@dataclass
class FetchConfig:
    timeout: float
    retries: int
    delay: float
    jitter: float
    verbose: bool
    insecure: bool


@dataclass
class FileUpdateStats:
    total_rows: int = 0
    rows_considered: int = 0
    rows_with_link: int = 0
    rows_changed: int = 0
    fetch_failures: int = 0
    parse_misses: int = 0


def log(message: str, *, enabled: bool = True) -> None:
    if enabled:
        print(message, flush=True)


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
    # Common Brickset block:
    # <dd>New: <a>~£661</a><a>~£336</a><br>Used: </dd>
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


def fetch_market_prices(link: str, cfg: FetchConfig) -> Optional[Tuple[Optional[str], Optional[str]]]:
    parsed_url = urlparse(link)
    if parsed_url.scheme.lower() not in {"http", "https"}:
        return None

    attempts = max(1, cfg.retries + 1)
    ssl_context = ssl._create_unverified_context() if cfg.insecure else None

    for attempt in range(1, attempts + 1):
        request = Request(link)
        request.add_header(
            "User-Agent",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1",
        )
        request.add_header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
        request.add_header("Accept-Language", "en-GB,en;q=0.9")
        request.add_header("Cache-Control", "no-cache")
        request.add_header("Referer", "https://brickset.com/")

        try:
            with urlopen(request, timeout=cfg.timeout, context=ssl_context) as response:
                status = getattr(response, "status", None) or response.getcode()
                if status == 429:
                    if attempt < attempts:
                        sleep_for = max(cfg.delay, 2.0)
                        log(f"[Market] HTTP 429 for {link}; retrying in {sleep_for:.1f}s", enabled=cfg.verbose)
                        time.sleep(sleep_for)
                        continue
                    log(f"[Market] HTTP 429 for {link}", enabled=cfg.verbose)
                    return None

                if status < 200 or status >= 300:
                    log(f"[Market] HTTP {status} for {link}", enabled=cfg.verbose)
                    return None

                html = decode_body(response.read())
                return parse_market_prices(html)

        except HTTPError as exc:
            if exc.code == 429 and attempt < attempts:
                sleep_for = max(cfg.delay, 2.0)
                log(f"[Market] HTTP 429 for {link}; retrying in {sleep_for:.1f}s", enabled=cfg.verbose)
                time.sleep(sleep_for)
                continue

            log(f"[Market] HTTP {exc.code} for {link}", enabled=cfg.verbose)
            return None
        except (URLError, TimeoutError) as exc:
            log(f"[Market] Request failed for {link}: {exc}", enabled=cfg.verbose)
            return None

    return None


def maybe_sleep(cfg: FetchConfig) -> None:
    delay = max(0.0, cfg.delay)
    if cfg.jitter > 0:
        delay += random.uniform(0.0, cfg.jitter)
    if delay > 0:
        time.sleep(delay)


def update_json_file(path: Path, cfg: FetchConfig, *, limit: Optional[int], start_index: int) -> FileUpdateStats:
    original_text = path.read_text(encoding="utf-8")
    data = json.loads(original_text)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")

    stats = FileUpdateStats(total_rows=len(data))

    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            continue

        if idx < start_index:
            continue
        if limit is not None and stats.rows_considered >= limit:
            break

        stats.rows_considered += 1

        link = str(row.get("link", "")).strip()
        if not link:
            continue

        stats.rows_with_link += 1
        log(f"[Market] Fetching row {idx + 1}/{len(data)} from {link}", enabled=cfg.verbose)

        parsed = fetch_market_prices(link, cfg)
        maybe_sleep(cfg)

        if parsed is None:
            stats.fetch_failures += 1
            continue

        new_price, used_price = parsed

        if not new_price and not used_price:
            stats.parse_misses += 1
            continue

        row_changed = False

        if new_price and row.get("New") != new_price:
            row["New"] = new_price
            row_changed = True

        if used_price and row.get("Used") != used_price:
            row["Used"] = used_price
            row_changed = True

        if row_changed:
            stats.rows_changed += 1

    if stats.rows_changed > 0:
        updated_text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        if updated_text != original_text:
            path.write_text(updated_text, encoding="utf-8")

    return stats


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update LEGO market values in JSON files.")
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
    parser.add_argument("--delay", type=float, default=2.5, help="Delay between requests in seconds.")
    parser.add_argument("--jitter", type=float, default=0.35, help="Max random jitter added to delay.")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-file row limit for testing.")
    parser.add_argument("--start-index", type=int, default=0, help="Optional per-file start index.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL certificate verification (for local troubleshooting only).",
    )
    return parser.parse_args(argv)


def print_summary(label: str, stats: FileUpdateStats) -> None:
    print(
        (
            f"[{label}] total={stats.total_rows} "
            f"considered={stats.rows_considered} "
            f"with_link={stats.rows_with_link} "
            f"changed={stats.rows_changed} "
            f"fetch_failures={stats.fetch_failures} "
            f"parse_misses={stats.parse_misses}"
        ),
        flush=True,
    )


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    cfg = FetchConfig(
        timeout=max(1.0, args.timeout),
        retries=max(0, args.retries),
        delay=max(0.0, args.delay),
        jitter=max(0.0, args.jitter),
        verbose=args.verbose,
        insecure=args.insecure,
    )

    sets_path = Path(args.sets_json)
    minifigs_path = Path(args.minifigs_json)

    if not sets_path.exists():
        print(f"Missing sets JSON: {sets_path}", file=sys.stderr)
        return 1

    if not minifigs_path.exists():
        print(f"Missing minifig JSON: {minifigs_path}", file=sys.stderr)
        return 1

    sets_stats = update_json_file(sets_path, cfg, limit=args.limit, start_index=max(0, args.start_index))
    minifigs_stats = update_json_file(minifigs_path, cfg, limit=args.limit, start_index=max(0, args.start_index))

    print_summary("Sets", sets_stats)
    print_summary("Minifigs", minifigs_stats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
