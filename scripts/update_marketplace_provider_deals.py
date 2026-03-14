#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import quote_plus, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

USER_AGENT = os.getenv(
    "MARKETPLACE_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
)
OUTPUT_FILENAME_BY_REGION = {
    "UK": "uk.json",
    "US": "us.json",
    "EU": "eu.json",
    "ALL": "all.json",
}

BLOCK_PATTERNS = (
    "access denied",
    "enable javascript and cookies to continue",
    "just a moment",
    "service unavailable error",
    "blocked access",
    "request could not be satisfied",
)
RETAIL_AVAILABILITY_HINTS = {
    "retail",
    "retail - limited",
    "lego exclusive",
    "promotional",
    "legoland exclusive",
    "insiders reward",
}
PROVIDER_SOURCES = {
    "amazon": "Amazon",
    "bricklink": "BrickLink",
    "brickowl": "BrickOwl",
    "johnlewis": "John Lewis",
    "lego": "LEGO",
    "very": "Very",
    "vinted": "Vinted",
}
PROVIDER_SEARCH_URLS = {
    "amazon": "https://www.amazon.co.uk/s?k={query}",
    "brickowl": "https://www.brickowl.com/search/catalog?query={query}",
    "johnlewis": "https://www.johnlewis.com/search?search-term={query}",
    "lego": "https://www.lego.com/en-gb/search?q={query}",
    "very": "https://www.very.co.uk/search?searchTerm={query}",
    "vinted": "https://www.vinted.co.uk/catalog?search_text={query}",
}


@dataclass(frozen=True)
class Candidate:
    number: str
    name: str
    item_type: str
    category: str
    search_term: str
    priority_score: int = 0


class MarketplaceProviderError(RuntimeError):
    pass


class MarketplaceAccessDenied(MarketplaceProviderError):
    pass


class MarketplaceNoData(MarketplaceProviderError):
    pass


def collapse_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_price_text(value: Any) -> str:
    raw = collapse_ws(unescape(str(value or "")))
    return raw.replace("\xa0", " ").strip()


def parse_price_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = clean_price_text(value)
    if not raw:
        return None
    raw = raw.replace(",", "")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", raw)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def normalize_currency_code(value: Any, default: str = "GBP") -> str:
    raw = collapse_ws(value).upper()
    if raw in {"£", "GBP"}:
        return "GBP"
    if raw in {"$", "USD"}:
        return "USD"
    if raw in {"€", "EUR"}:
        return "EUR"
    if len(raw) == 3:
        return raw
    return default


def normalize_region(value: str) -> str:
    raw = collapse_ws(value).upper()
    if raw in {"GB", "UK"}:
        return "UK"
    if raw in {"US", "USA"}:
        return "US"
    if raw in {"EU", "DE", "FR", "ES", "IT"}:
        return "EU"
    return raw or "UK"


def normalize_set_number(number: Any, variant: Any) -> str:
    raw = collapse_ws(number)
    if not raw:
        return ""
    if re.search(r"-[0-9]+$", raw):
        return raw
    try:
        variant_no = int(float(collapse_ws(variant) or "1"))
    except ValueError:
        variant_no = 1
    return f"{raw}-{max(1, variant_no)}"


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [row for row in data if isinstance(row, dict)]


def write_json(path: Path, data: Any, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        if pretty
        else json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    path.write_text(payload, encoding="utf-8")


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def parse_priority_list(value: str) -> List[str]:
    return [collapse_ws(part) for part in re.split(r"[,\n]", value or "") if collapse_ws(part)]


def rotating_slice(items: Sequence[Candidate], start_index: int, limit: int) -> Tuple[List[Candidate], int]:
    if not items or limit <= 0:
        return [], 0
    start = start_index % len(items)
    output: List[Candidate] = []
    for offset in range(min(limit, len(items))):
        output.append(items[(start + offset) % len(items)])
    return output, (start + len(output)) % len(items)


def current_year_utc() -> int:
    return datetime.now(timezone.utc).year


def parse_optional_int(value: Any) -> Optional[int]:
    raw = collapse_ws(value)
    if not raw:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def is_currentish_retail_set(row: Dict[str, Any]) -> bool:
    availability = collapse_ws(row.get("Availability")).lower()
    year_value = parse_optional_int(row.get("YearFrom"))
    released = collapse_ws(row.get("Released")).lower()

    if availability in {value.lower() for value in RETAIL_AVAILABILITY_HINTS}:
        return True
    if year_value is not None and year_value >= current_year_utc() - 3 and released != "false":
        return True
    return False


def provider_search_term(provider: str, *, item_type: str, number: str, name: str) -> str:
    trimmed_name = " ".join(name.split()[:6])
    if item_type == "set" and provider not in {"bricklink", "brickowl"}:
        number_token = number.split("-", 1)[0]
    else:
        number_token = number
    return collapse_ws(f"LEGO {number_token} {trimmed_name}")


def theme_priority_bonus(value: str, priority_values: Sequence[str]) -> int:
    normalized = value.casefold()
    for index, priority in enumerate(priority_values):
        if normalized == priority.casefold():
            return max(0, 240 - (index * 18))
    return 0


def score_candidate(row: Dict[str, Any], *, item_type: str, priority_values: Sequence[str]) -> int:
    category = ""
    if item_type == "set":
        category = collapse_ws(row.get("Theme"))
    elif item_type == "minifig":
        category = collapse_ws(row.get("Category") or row.get("Theme"))
    else:
        category = collapse_ws(row.get("category"))

    score = theme_priority_bonus(category, priority_values)

    if item_type == "set":
        year_value = parse_optional_int(row.get("YearFrom"))
        availability = collapse_ws(row.get("Availability")).lower()
        released = collapse_ws(row.get("Released")).lower()
        pieces = parse_optional_int(row.get("Pieces")) or 0
        if availability in {value.lower() for value in RETAIL_AVAILABILITY_HINTS}:
            score += 220
        if year_value is not None:
            score += max(0, 120 - max(0, current_year_utc() - year_value) * 20)
        if released != "false":
            score += 40
        if pieces > 0:
            score += min(24, pieces // 250)
    elif item_type == "minifig":
        year_value = parse_optional_int(row.get("Year released") or row.get("Year"))
        if year_value is not None:
            score += max(0, 80 - max(0, current_year_utc() - year_value) * 14)
    else:
        category_lower = category.casefold()
        if "minifig" in category_lower:
            score += 40
        elif "tile" in category_lower or "plate" in category_lower or "brick" in category_lower:
            score += 28

    return score


def make_candidate(row: Dict[str, Any], *, provider: str, item_type: str, priority_values: Sequence[str]) -> Optional[Candidate]:
    if item_type == "set":
        number = normalize_set_number(row.get("Number"), row.get("Variant"))
        name = collapse_ws(row.get("SetName"))
        category = collapse_ws(row.get("Theme")) or "Unknown"
    elif item_type == "minifig":
        number = collapse_ws(row.get("Number"))
        name = collapse_ws(row.get("Minifig name") or row.get("Character name"))
        category = collapse_ws(row.get("Category") or row.get("Theme")) or "Unknown"
    else:
        number = collapse_ws(row.get("part_num"))
        name = collapse_ws(row.get("name"))
        category = collapse_ws(row.get("category")) or "General"

    if not number or not name:
        return None

    return Candidate(
        number=number,
        name=name,
        item_type=item_type,
        category=category,
        search_term=provider_search_term(provider, item_type=item_type, number=number, name=name),
        priority_score=score_candidate(row, item_type=item_type, priority_values=priority_values),
    )


def prioritized_candidates(
    rows: Sequence[Dict[str, Any]],
    *,
    provider: str,
    item_type: str,
    priority_values: Sequence[str],
    currentish_only: bool = False,
) -> List[Candidate]:
    candidates: List[Candidate] = []

    for row in rows:
        if currentish_only and item_type == "set" and not is_currentish_retail_set(row):
            continue
        candidate = make_candidate(row, provider=provider, item_type=item_type, priority_values=priority_values)
        if candidate is None:
            continue
        candidates.append(candidate)

    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate.priority_score,
            candidate.category.casefold(),
            candidate.number.casefold(),
        ),
    )


def filter_candidates_for_request(
    candidates: Sequence[Candidate],
    *,
    requested_number: str,
    requested_item_type: str,
) -> List[Candidate]:
    normalized_number = requested_number.casefold()
    normalized_type = requested_item_type.casefold()
    return [
        candidate
        for candidate in candidates
        if candidate.item_type.casefold() == normalized_type and candidate.number.casefold() == normalized_number
    ]


def build_candidate_plan(args: argparse.Namespace) -> Dict[str, List[Candidate]]:
    sets = load_json_array(Path(args.sets_json))
    minifigs = load_json_array(Path(args.minifigs_json))
    parts_path = Path(args.parts_json)
    parts = load_json_array(parts_path) if parts_path.exists() else []

    provider = args.provider
    priority_themes = parse_priority_list(args.priority_themes)
    priority_minifig_categories = parse_priority_list(args.priority_minifig_categories)
    priority_part_categories = parse_priority_list(args.priority_part_categories)

    if provider in {"lego", "johnlewis", "very"}:
        plan = {
            "set": prioritized_candidates(sets, provider=provider, item_type="set", priority_values=priority_themes, currentish_only=True),
            "minifig": [],
            "part": [],
        }
    elif provider == "vinted":
        plan = {
            "set": prioritized_candidates(sets, provider=provider, item_type="set", priority_values=priority_themes),
            "minifig": prioritized_candidates(minifigs, provider=provider, item_type="minifig", priority_values=priority_minifig_categories),
            "part": [],
        }
    else:
        plan = {
            "set": prioritized_candidates(sets, provider=provider, item_type="set", priority_values=priority_themes),
            "minifig": prioritized_candidates(minifigs, provider=provider, item_type="minifig", priority_values=priority_minifig_categories),
            "part": prioritized_candidates(parts, provider=provider, item_type="part", priority_values=priority_part_categories),
        }

    if args.only_number:
        requested_number = collapse_ws(args.only_number)
        requested_item_type = collapse_ws(args.only_item_type or "set")
        filtered_plan = {
            item_type: filter_candidates_for_request(
                candidates,
                requested_number=requested_number,
                requested_item_type=requested_item_type,
            )
            for item_type, candidates in plan.items()
        }
        if not any(filtered_plan.values()):
            raise MarketplaceNoData(f"No catalog candidate found for {requested_item_type} {requested_number}")
        return filtered_plan

    return plan


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch marketplace provider deals into provider-scoped artifacts.")
    parser.add_argument("--provider", required=True, choices=sorted(PROVIDER_SOURCES))
    parser.add_argument("--sets-json", default="dist/Lego Star Wars Database.json")
    parser.add_argument("--minifigs-json", default="dist/Lego-Star-Wars-Minifigure-Database.json")
    parser.add_argument("--parts-json", default="dist/parts/parts-catalog.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--regions", default="UK")
    parser.add_argument("--sets-per-region", type=int, default=80)
    parser.add_argument("--minifigs-per-region", type=int, default=40)
    parser.add_argument("--parts-per-region", type=int, default=30)
    parser.add_argument("--max-results-per-item", type=int, default=5)
    parser.add_argument("--max-product-pages-per-item", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--request-delay", type=float, default=0.15)
    parser.add_argument("--priority-themes", default="Star Wars,Icons,Technic,Speed Champions,Marvel Super Heroes,Harry Potter,Disney,City,Botanicals,NINJAGO")
    parser.add_argument("--priority-minifig-categories", default="Star Wars,Marvel Super Heroes,Harry Potter,Disney,Collectable Minifigures,NINJAGO")
    parser.add_argument("--priority-part-categories", default="Bricks,Plates,Tiles,Minifigure")
    parser.add_argument("--only-number", default="")
    parser.add_argument("--only-item-type", default="")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-GB,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    return session


def request_text(session: requests.Session, url: str, *, timeout: float, retries: int) -> str:
    attempts = max(1, retries + 1)
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, timeout=timeout)
            status = response.status_code
            text = response.text or ""
            lowered = text.casefold()
            if any(token in lowered for token in BLOCK_PATTERNS):
                raise MarketplaceAccessDenied(f"Access blocked for {url}")
            if status in {403, 429} or status >= 500:
                raise MarketplaceProviderError(f"HTTP {status} for {url}")
            if status >= 400:
                raise MarketplaceProviderError(f"HTTP {status} for {url}")
            return text
        except (requests.RequestException, MarketplaceProviderError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(min(15.0, attempt * 1.5))
    if isinstance(last_error, MarketplaceProviderError):
        raise last_error
    raise MarketplaceProviderError(f"Request failed for {url}: {last_error}")


def strip_fragment(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(fragment="").geturl()


def normalize_absolute_url(base_url: str, value: str) -> str:
    return strip_fragment(urljoin(base_url, value.strip()))


def flatten_json_ld(value: Any) -> Iterator[Dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from flatten_json_ld(nested)
    elif isinstance(value, list):
        for item in value:
            yield from flatten_json_ld(item)


def parse_json_ld_nodes(html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    output: List[Dict[str, Any]] = []
    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        raw = script.string or script.get_text(" ", strip=True)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        output.extend(list(flatten_json_ld(parsed)))
    return output


def title_matches_candidate(title: str, candidate: Candidate) -> bool:
    lowered = collapse_ws(title).casefold()
    number = candidate.number.casefold()
    compact_number = number.split("-", 1)[0]
    if number in lowered or compact_number in lowered:
        return True
    if candidate.item_type == "part" and compact_number in lowered:
        return True
    return False


def build_deal(
    *,
    provider: str,
    candidate: Candidate,
    region: str,
    url: str,
    title: str,
    price_value: Optional[float],
    price_text: Optional[str],
    currency_code: str,
    subtitle: Optional[str] = None,
    listed_at: Optional[str] = None,
    deal_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not url or not title or price_value is None:
        return None
    source = PROVIDER_SOURCES[provider]
    normalized_title = collapse_ws(title)
    if not title_matches_candidate(normalized_title, candidate) and candidate.number.split("-", 1)[0] not in url.casefold():
        return None
    return {
        "id": deal_id or f"{candidate.number.lower()}|{provider}|{url.lower()}",
        "number": candidate.number,
        "source": source,
        "title": normalized_title,
        "subtitle": collapse_ws(subtitle) or None,
        "priceValue": round(price_value, 2),
        "priceText": clean_price_text(price_text) or f"{currency_code} {price_value:.2f}",
        "currencyCode": normalize_currency_code(currency_code),
        "url": url,
        "regionCode": region,
        "listedAt": listed_at,
    }


def dedupe_and_sort(deals: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for deal in deals:
        deal_id = collapse_ws(deal.get("id")).lower()
        if not deal_id:
            continue
        merged[deal_id] = deal
    return sorted(
        merged.values(),
        key=lambda row: (
            collapse_ws(row.get("number")).casefold(),
            float(row.get("priceValue") or 0.0),
            collapse_ws(row.get("source")).casefold(),
            collapse_ws(row.get("title")).casefold(),
        ),
    )


def trim_deals_per_number(deals: Iterable[Dict[str, Any]], *, max_results_per_item: int) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for deal in dedupe_and_sort(deals):
        condition = ""
        subtitle = collapse_ws(deal.get("subtitle"))
        if subtitle:
            condition = subtitle.split("•", 1)[0].strip()
        key = (collapse_ws(deal.get("number")).lower(), condition.casefold())
        grouped.setdefault(key, []).append(deal)

    output: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: float(row.get("priceValue") or 0.0))
        output.extend(rows[:max(1, max_results_per_item)])
    return output


def parse_condition_from_text(value: str, *, default: str = "") -> str:
    lowered = collapse_ws(value).lower()
    if "used" in lowered:
        return "Used"
    if "sealed" in lowered or "new" in lowered:
        return "New"
    return default


def extract_johnlewis_urls(html: str, *, limit: int, base_url: str) -> List[str]:
    urls: List[str] = []
    seen: set[str] = set()
    for node in parse_json_ld_nodes(html):
        if collapse_ws(node.get("@type")) != "ItemList":
            continue
        items = node.get("itemListElement") if isinstance(node.get("itemListElement"), list) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            raw_url = collapse_ws(item.get("url") or (item.get("item") or {}).get("@id"))
            if not raw_url:
                continue
            absolute = normalize_absolute_url(base_url, raw_url)
            if absolute in seen:
                continue
            seen.add(absolute)
            urls.append(absolute)
            if len(urls) >= limit:
                return urls
    soup = BeautifulSoup(html, "html.parser")
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if not re.search(r"/p[0-9]+$", href):
            continue
        absolute = normalize_absolute_url(base_url, href)
        if absolute in seen:
            continue
        seen.add(absolute)
        urls.append(absolute)
        if len(urls) >= limit:
            break
    return urls


def parse_johnlewis_product_page(html: str, *, provider: str, candidate: Candidate, region: str, url: str) -> List[Dict[str, Any]]:
    for node in parse_json_ld_nodes(html):
        if collapse_ws(node.get("@type")) != "Product":
            continue
        offers = node.get("offers") if isinstance(node.get("offers"), dict) else {}
        title = collapse_ws(node.get("name"))
        price_value = parse_price_value(offers.get("price"))
        price_text = None
        if price_value is not None:
            price_text = f"GBP {price_value:.2f}"
        availability = collapse_ws(offers.get("availability") or "").rsplit("/", 1)[-1]
        subtitle = "John Lewis"
        if availability:
            subtitle = f"{subtitle} • {availability}"
        deal = build_deal(
            provider=provider,
            candidate=candidate,
            region=region,
            url=url,
            title=title,
            price_value=price_value,
            price_text=price_text,
            currency_code=normalize_currency_code(offers.get("priceCurrency"), "GBP"),
            subtitle=subtitle,
            deal_id=f"{candidate.number.lower()}|{provider}|{url.lower()}",
        )
        return [deal] if deal else []
    return []


def extract_vinted_deals(html: str, *, provider: str, candidate: Candidate, region: str, base_url: str, limit: int) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    output: List[Dict[str, Any]] = []
    for card in soup.select("div[data-testid='grid-item']"):
        link = card.select_one("a[href*='/items/']")
        if link is None:
            continue
        url = normalize_absolute_url(base_url, link.get("href", ""))
        title = collapse_ws(link.get("title") or "")
        if not title:
            title = collapse_ws(" ".join(card.stripped_strings))
        price_node = card.select_one("[data-testid$='--price-text']")
        price_text = clean_price_text(price_node.get_text(" ", strip=True) if price_node else "")
        total_node = card.select_one("[data-testid='total-combined-price']")
        total_price_text = clean_price_text(total_node.get_text(" ", strip=True) if total_node else "")
        price_value = parse_price_value(price_text) or parse_price_value(total_price_text)
        if price_value is None:
            continue
        desc_title = card.select_one("[data-testid$='--description-title']")
        desc_subtitle = card.select_one("[data-testid$='--description-subtitle']")
        subtitle_parts = [collapse_ws(desc_title.get_text(" ", strip=True) if desc_title else ""), collapse_ws(desc_subtitle.get_text(" ", strip=True) if desc_subtitle else "")]
        subtitle_parts = [part for part in subtitle_parts if part]
        deal = build_deal(
            provider=provider,
            candidate=candidate,
            region=region,
            url=url,
            title=title,
            price_value=price_value,
            price_text=price_text or total_price_text,
            currency_code="GBP",
            subtitle=" • ".join(subtitle_parts) or "Vinted",
        )
        if deal is not None:
            output.append(deal)
        if len(output) >= limit:
            break
    return output


def extract_amazon_deals(html: str, *, provider: str, candidate: Candidate, region: str, base_url: str, limit: int) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    output: List[Dict[str, Any]] = []
    for result in soup.select("div[data-component-type='s-search-result']"):
        title_node = result.select_one("h2 a span")
        link_node = result.select_one("h2 a[href]")
        price_node = result.select_one("span.a-price span.a-offscreen")
        if not title_node or not link_node or not price_node:
            continue
        url = normalize_absolute_url(base_url, link_node.get("href", ""))
        title = collapse_ws(title_node.get_text(" ", strip=True))
        price_text = clean_price_text(price_node.get_text(" ", strip=True))
        price_value = parse_price_value(price_text)
        subtitle_parts: List[str] = []
        condition = parse_condition_from_text(result.get_text(" ", strip=True), default="New")
        if condition:
            subtitle_parts.append(condition)
        subtitle_parts.append("Amazon")
        deal = build_deal(
            provider=provider,
            candidate=candidate,
            region=region,
            url=url,
            title=title,
            price_value=price_value,
            price_text=price_text,
            currency_code="GBP",
            subtitle=" • ".join(subtitle_parts),
        )
        if deal is not None:
            output.append(deal)
        if len(output) >= limit:
            break
    return output


def extract_product_urls_from_brickowl_search(html: str, *, limit: int, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls: List[str] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if not href.startswith("/catalog/"):
            continue
        absolute = normalize_absolute_url(base_url, href)
        if absolute in seen:
            continue
        seen.add(absolute)
        urls.append(absolute)
        if len(urls) >= limit:
            break
    return urls


def parse_brickowl_product_page(html: str, *, provider: str, candidate: Candidate, region: str, url: str, limit: int) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    output: List[Dict[str, Any]] = []
    buy_rows = soup.select("#buy table tr") or soup.select("tr")
    for row in buy_rows:
        columns = row.find_all("td")
        if len(columns) < 7:
            continue
        listing_link = columns[2].find("a", href=True) or columns[0].find("a", href=True)
        if listing_link is None:
            continue
        listing_url = normalize_absolute_url(url, listing_link.get("href", ""))
        title = collapse_ws(listing_link.get_text(" ", strip=True))
        price_text = clean_price_text(columns[4].get_text(" ", strip=True))
        price_value = parse_price_value(price_text)
        if price_value is None:
            continue
        condition_text = collapse_ws(columns[1].get_text(" ", strip=True))
        note = collapse_ws(columns[2].get_text(" ", strip=True).replace(title, "", 1))
        store_text = collapse_ws(columns[6].get_text(" ", strip=True))
        subtitle_parts = [condition_text, store_text]
        if note:
            subtitle_parts.append(note)
        deal = build_deal(
            provider=provider,
            candidate=candidate,
            region=region,
            url=listing_url,
            title=title,
            price_value=price_value,
            price_text=price_text,
            currency_code="GBP",
            subtitle=" • ".join(part for part in subtitle_parts if part),
        )
        if deal is not None:
            output.append(deal)
        if len(output) >= limit:
            break
    return output


def extract_generic_product_urls(html: str, *, base_url: str, limit: int, href_pattern: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls: List[str] = []
    seen: set[str] = set()
    regex = re.compile(href_pattern)
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if not regex.search(href):
            continue
        absolute = normalize_absolute_url(base_url, href)
        if absolute in seen:
            continue
        seen.add(absolute)
        urls.append(absolute)
        if len(urls) >= limit:
            break
    return urls


def parse_generic_product_offer_page(
    html: str,
    *,
    provider: str,
    candidate: Candidate,
    region: str,
    url: str,
    default_currency: str = "GBP",
) -> List[Dict[str, Any]]:
    for node in parse_json_ld_nodes(html):
        if collapse_ws(node.get("@type")) != "Product":
            continue
        offers = node.get("offers")
        if isinstance(offers, list):
            offers = next((entry for entry in offers if isinstance(entry, dict)), {})
        if not isinstance(offers, dict):
            offers = {}
        title = collapse_ws(node.get("name")) or collapse_ws(node.get("headline"))
        price_value = parse_price_value(offers.get("price") or offers.get("lowPrice") or node.get("price"))
        currency_code = normalize_currency_code(offers.get("priceCurrency") or node.get("priceCurrency"), default_currency)
        availability = collapse_ws(offers.get("availability") or node.get("availability") or "")
        availability = availability.rsplit("/", 1)[-1] if availability else ""
        seller = None
        if isinstance(offers.get("seller"), dict):
            seller = collapse_ws(offers["seller"].get("name"))
        subtitle_parts = [seller or PROVIDER_SOURCES[provider]]
        if availability:
            subtitle_parts.append(availability)
        price_text = f"{currency_code} {price_value:.2f}" if price_value is not None else None
        deal = build_deal(
            provider=provider,
            candidate=candidate,
            region=region,
            url=url,
            title=title,
            price_value=price_value,
            price_text=price_text,
            currency_code=currency_code,
            subtitle=" • ".join(part for part in subtitle_parts if part),
        )
        return [deal] if deal else []
    return []


def fetch_search_deals(
    session: requests.Session,
    *,
    provider: str,
    candidate: Candidate,
    region: str,
    timeout: float,
    retries: int,
    max_results_per_item: int,
    max_product_pages_per_item: int,
) -> List[Dict[str, Any]]:
    search_url = PROVIDER_SEARCH_URLS[provider].format(query=quote_plus(candidate.search_term))
    search_html = request_text(session, search_url, timeout=timeout, retries=retries)
    base_url = search_url

    if provider == "amazon":
        return extract_amazon_deals(search_html, provider=provider, candidate=candidate, region=region, base_url=base_url, limit=max_results_per_item)
    if provider == "vinted":
        return extract_vinted_deals(search_html, provider=provider, candidate=candidate, region=region, base_url=base_url, limit=max_results_per_item)
    if provider == "johnlewis":
        urls = extract_johnlewis_urls(search_html, limit=max_product_pages_per_item, base_url=base_url)
        output: List[Dict[str, Any]] = []
        for product_url in urls:
            product_html = request_text(session, product_url, timeout=timeout, retries=retries)
            output.extend(parse_johnlewis_product_page(product_html, provider=provider, candidate=candidate, region=region, url=product_url))
            if len(output) >= max_results_per_item:
                break
        return output[:max_results_per_item]
    if provider == "brickowl":
        urls = extract_product_urls_from_brickowl_search(search_html, limit=max_product_pages_per_item, base_url=base_url)
        output: List[Dict[str, Any]] = []
        for product_url in urls:
            product_html = request_text(session, product_url, timeout=timeout, retries=retries)
            output.extend(parse_brickowl_product_page(product_html, provider=provider, candidate=candidate, region=region, url=product_url, limit=max_results_per_item))
            if len(output) >= max_results_per_item:
                break
        return output[:max_results_per_item]
    if provider == "lego":
        urls = extract_generic_product_urls(search_html, base_url=base_url, limit=max_product_pages_per_item, href_pattern=rf"/product/[^\"']*{re.escape(candidate.number.split('-',1)[0])}")
        output: List[Dict[str, Any]] = []
        for product_url in urls:
            product_html = request_text(session, product_url, timeout=timeout, retries=retries)
            output.extend(parse_generic_product_offer_page(product_html, provider=provider, candidate=candidate, region=region, url=product_url))
            if len(output) >= max_results_per_item:
                break
        return output[:max_results_per_item]
    if provider == "very":
        urls = extract_generic_product_urls(search_html, base_url=base_url, limit=max_product_pages_per_item, href_pattern=rf"{re.escape(candidate.number.split('-',1)[0])}")
        output: List[Dict[str, Any]] = []
        for product_url in urls:
            product_html = request_text(session, product_url, timeout=timeout, retries=retries)
            output.extend(parse_generic_product_offer_page(product_html, provider=provider, candidate=candidate, region=region, url=product_url))
            if len(output) >= max_results_per_item:
                break
        return output[:max_results_per_item]
    raise MarketplaceProviderError(f"Unsupported provider search implementation: {provider}")


def build_bricklink_deals_from_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    item_type: str,
    region: str,
    max_results_per_item: int,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in rows:
        candidate = make_candidate(row, item_type=item_type)
        if candidate is None:
            continue
        condition_lists = [
            ("BrickLinkCurrentListingsNew", "New"),
            ("BrickLinkCurrentListingsUsed", "Used"),
        ]
        for key, condition in condition_lists:
            listings = row.get(key)
            if not isinstance(listings, list) or not listings:
                continue
            parsed: List[Dict[str, Any]] = []
            for entry in listings:
                if not isinstance(entry, dict):
                    continue
                url = collapse_ws(entry.get("listingURL"))
                price_value = parse_price_value(entry.get("eachPrice"))
                currency_code = normalize_currency_code(entry.get("currency"), "USD")
                if not url or price_value is None:
                    continue
                store_id = collapse_ws(entry.get("storeId"))
                region_text = collapse_ws(entry.get("region"))
                qty_text = collapse_ws(entry.get("qty"))
                subtitle_parts = [condition]
                if store_id:
                    subtitle_parts.append(f"Store #{store_id}")
                if region_text:
                    subtitle_parts.append(region_text)
                if qty_text:
                    subtitle_parts.append(f"Qty {qty_text}")
                deal = build_deal(
                    provider="bricklink",
                    candidate=candidate,
                    region=region,
                    url=url,
                    title=f"{candidate.name}",
                    price_value=price_value,
                    price_text=f"{currency_code} {price_value:.2f}",
                    currency_code=currency_code,
                    subtitle=" • ".join(subtitle_parts),
                    deal_id=f"{candidate.number.lower()}|bricklink|{url.lower()}",
                )
                if deal is not None:
                    parsed.append(deal)
            parsed = sorted(parsed, key=lambda row: float(row.get("priceValue") or 0.0))[:max_results_per_item]
            output.extend(parsed)
    return output


def build_bricklink_provider_output(args: argparse.Namespace, regions: List[str]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    sets = load_json_array(Path(args.sets_json))
    minifigs = load_json_array(Path(args.minifigs_json))
    parts_path = Path(args.parts_json)
    parts = load_json_array(parts_path) if parts_path.exists() else []
    if args.only_number:
        requested_number = collapse_ws(args.only_number).casefold()
        requested_item_type = collapse_ws(args.only_item_type or "set").casefold()
        if requested_item_type == "set":
            sets = [row for row in sets if normalize_set_number(row.get("Number"), row.get("Variant")).casefold() == requested_number]
            minifigs = []
            parts = []
        elif requested_item_type == "minifig":
            minifigs = [row for row in minifigs if collapse_ws(row.get("Number")).casefold() == requested_number]
            sets = []
            parts = []
        elif requested_item_type == "part":
            parts = [row for row in parts if collapse_ws(row.get("part_num")).casefold() == requested_number]
            sets = []
            minifigs = []
    region = regions[0] if regions else "UK"
    deals = []
    deals.extend(build_bricklink_deals_from_rows(sets, item_type="set", region=region, max_results_per_item=args.max_results_per_item))
    deals.extend(build_bricklink_deals_from_rows(minifigs, item_type="minifig", region=region, max_results_per_item=args.max_results_per_item))
    deals.extend(build_bricklink_deals_from_rows(parts, item_type="part", region=region, max_results_per_item=args.max_results_per_item))
    output_by_region = {region: dedupe_and_sort(deals)}
    stats = {
        "provider": args.provider,
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "regions": {region: {"deals": len(output_by_region[region])}},
    }
    return output_by_region, stats


def build_search_provider_output(args: argparse.Namespace, regions: List[str]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    candidate_plan = build_candidate_plan(args)
    state_path = Path(args.state_path)
    state = load_state(state_path)
    region_state = state.get("regions") if isinstance(state.get("regions"), dict) else {}
    session = make_session()

    output_by_region: Dict[str, List[Dict[str, Any]]] = {}
    next_region_state: Dict[str, Dict[str, Any]] = {}
    total_search_errors = 0
    total_access_denied = 0

    for region in regions:
        current_state = region_state.get(region) if isinstance(region_state.get(region), dict) else {}
        set_start = int(current_state.get("nextSetIndex", 0) or 0)
        minifig_start = int(current_state.get("nextMinifigIndex", 0) or 0)
        part_start = int(current_state.get("nextPartIndex", 0) or 0)

        if args.only_number:
            selected_sets = list(candidate_plan["set"])
            selected_minifigs = list(candidate_plan["minifig"])
            selected_parts = list(candidate_plan["part"])
            next_set_index = set_start
            next_minifig_index = minifig_start
            next_part_index = part_start
        else:
            selected_sets, next_set_index = rotating_slice(candidate_plan["set"], set_start, args.sets_per_region)
            selected_minifigs, next_minifig_index = rotating_slice(candidate_plan["minifig"], minifig_start, args.minifigs_per_region)
            selected_parts, next_part_index = rotating_slice(candidate_plan["part"], part_start, args.parts_per_region)

        region_deals: List[Dict[str, Any]] = []
        processed = 0
        for candidate in selected_sets + selected_minifigs + selected_parts:
            try:
                found = fetch_search_deals(
                    session,
                    provider=args.provider,
                    candidate=candidate,
                    region=region,
                    timeout=args.timeout,
                    retries=args.retries,
                    max_results_per_item=args.max_results_per_item,
                    max_product_pages_per_item=args.max_product_pages_per_item,
                )
                region_deals.extend(found)
                processed += 1
                if args.verbose:
                    print(f"[{args.provider}:{region}] {candidate.item_type} {candidate.number} -> {len(found)} deals", flush=True)
            except MarketplaceAccessDenied as exc:
                total_access_denied += 1
                total_search_errors += 1
                print(f"[{args.provider}:{region}] blocked {candidate.number}: {exc}", flush=True)
            except MarketplaceProviderError as exc:
                total_search_errors += 1
                print(f"[{args.provider}:{region}] error {candidate.number}: {exc}", flush=True)
            time.sleep(max(0.0, args.request_delay))

        output_by_region[region] = trim_deals_per_number(region_deals, max_results_per_item=args.max_results_per_item)
        next_region_state[region] = {
            "nextSetIndex": next_set_index,
            "nextMinifigIndex": next_minifig_index,
            "nextPartIndex": next_part_index,
            "processed": processed,
            "deals": len(output_by_region[region]),
            "updatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    attempted_items = sum(
        min(len(candidate_plan["set"]), args.sets_per_region)
        + min(len(candidate_plan["minifig"]), args.minifigs_per_region)
        + min(len(candidate_plan["part"]), args.parts_per_region)
        for _region in regions
    )
    if attempted_items > 0 and total_access_denied >= max(3, attempted_items // 2):
        raise MarketplaceAccessDenied(
            f"{args.provider} appears to be blocking automated access: access_denied={total_access_denied}, attempted_items={attempted_items}"
        )
    if attempted_items > 0 and total_search_errors == attempted_items and not any(output_by_region.values()):
        raise MarketplaceNoData(f"{args.provider} produced no successful listing fetches")

    stats = {
        "provider": args.provider,
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "regions": next_region_state,
        "searchErrors": total_search_errors,
        "accessDenied": total_access_denied,
    }
    write_json(state_path, stats, pretty=True)
    return output_by_region, stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    regions = [normalize_region(value) for value in re.split(r"[,\s]+", args.regions) if collapse_ws(value)]
    regions = [region for region in regions if region in OUTPUT_FILENAME_BY_REGION]
    if not regions:
        raise MarketplaceProviderError("No supported regions configured")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.provider == "bricklink":
        output_by_region, stats = build_bricklink_provider_output(args, regions)
    else:
        output_by_region, stats = build_search_provider_output(args, regions)

    for region, deals in output_by_region.items():
        filename = OUTPUT_FILENAME_BY_REGION[region]
        write_json(output_dir / filename, deals)
        print(f"[{args.provider}:{region}] deals={len(deals)}", flush=True)

    write_json(output_dir / "metadata.json", stats, pretty=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MarketplaceProviderError as exc:
        print(f"[marketplace] {exc}", file=sys.stderr)
        raise SystemExit(1)
