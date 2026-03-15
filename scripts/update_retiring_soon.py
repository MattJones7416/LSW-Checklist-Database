#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import ssl
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BRICKECONOMY_URL = "https://www.brickeconomy.com/sets/retiring-soon"
BRICKSET_URL_TEMPLATE = "https://brickset.com/sets/retiringin-{year:04d}-{month:02d}"
OUTPUT_PATH = Path("dist/retiring-soon.json")
NUMBER_PATTERNS = [
    re.compile(r'/sets?/([0-9][A-Z0-9.]*-[0-9]+)', re.IGNORECASE),
    re.compile(r'>([0-9][A-Z0-9.]*-[0-9]+)<', re.IGNORECASE),
]
BLOCK_MARKERS = (
    'just a moment',
    'enable javascript and cookies to continue',
)


def fetch_html_with_urllib(url: str) -> str:
    request = Request(
        url,
        headers={
            'User-Agent': 'Mozilla/5.0 (compatible; LSWChecklistBot/1.0; +https://github.com/MattJones7416/LSW-Checklist-Database)',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'en-GB,en;q=0.9',
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            return response.read().decode('utf-8', errors='ignore')
    except ssl.SSLCertVerificationError:
        insecure = ssl._create_unverified_context()
        with urlopen(request, timeout=30, context=insecure) as response:
            return response.read().decode('utf-8', errors='ignore')
    except URLError as error:
        if isinstance(error.reason, ssl.SSLCertVerificationError):
            insecure = ssl._create_unverified_context()
            with urlopen(request, timeout=30, context=insecure) as response:
                return response.read().decode('utf-8', errors='ignore')
        raise


def fetch_html_with_curl(url: str) -> str:
    result = subprocess.run(
        ['curl', '-Lk', '--compressed', url],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def fetch_html(url: str) -> str:
    errors: list[str] = []
    for fetcher in (fetch_html_with_urllib, fetch_html_with_curl):
        try:
            html = fetcher(url)
            if html and not any(marker in html.lower() for marker in BLOCK_MARKERS):
                return html
            errors.append(f'{fetcher.__name__}: blocked or empty response')
        except (URLError, HTTPError, subprocess.CalledProcessError) as error:
            errors.append(f'{fetcher.__name__}: {error}')
    raise RuntimeError('; '.join(errors))


def extract_set_numbers(html: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for pattern in NUMBER_PATTERNS:
        for match in pattern.findall(html):
            value = match.strip().upper()
            if value and value not in seen:
                seen.add(value)
                found.append(value)
    return found


def brickset_month_urls(month_count: int = 6) -> list[str]:
    cursor = datetime.now(timezone.utc)
    year = cursor.year
    month = cursor.month
    urls: list[str] = []
    for offset in range(month_count):
        absolute_month = month - 1 + offset
        target_year = year + absolute_month // 12
        target_month = absolute_month % 12 + 1
        urls.append(BRICKSET_URL_TEMPLATE.format(year=target_year, month=target_month))
    return urls


def collect_brickset_fallback_numbers() -> list[str]:
    seen: set[str] = set()
    found: list[str] = []
    for url in brickset_month_urls():
        html = fetch_html(url)
        for value in extract_set_numbers(html):
            if value not in seen:
                seen.add(value)
                found.append(value)
    return found


def main() -> int:
    source = 'BrickEconomy'
    source_urls = [BRICKECONOMY_URL]
    try:
        html = fetch_html(BRICKECONOMY_URL)
        set_numbers = extract_set_numbers(html)
        if not set_numbers:
            raise RuntimeError('No set numbers parsed from BrickEconomy page')
    except Exception as error:
        source = 'Brickset'
        source_urls = brickset_month_urls()
        set_numbers = collect_brickset_fallback_numbers()
        if not set_numbers:
            raise SystemExit(f'Unable to build retiring-soon feed: {error}')

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'source': source,
        'sourceURL': source_urls[0],
        'sourceURLs': source_urls,
        'fetchedAt': datetime.now(timezone.utc).isoformat(),
        'setNumbers': set_numbers,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f'[RetiringSoon] wrote {len(set_numbers)} set numbers to {OUTPUT_PATH} from {source}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
