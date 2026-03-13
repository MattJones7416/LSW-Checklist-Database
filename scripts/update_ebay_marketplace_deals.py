#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

OAUTH_URL = "https://api.ebay.com/identity/v1/oauth2/token"
BROWSE_SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
APP_SCOPE = "https://api.ebay.com/oauth/api_scope"

MARKETPLACE_BY_REGION = {
    "UK": "EBAY_GB",
    "US": "EBAY_US",
    "EU": "EBAY_DE",
}
OUTPUT_FILENAME_BY_REGION = {
    "UK": "uk.json",
    "US": "us.json",
    "EU": "eu.json",
}


@dataclass(frozen=True)
class Candidate:
    number: str
    name: str
    item_type: str
    category: str
    search_term: str


class EbayApiError(RuntimeError):
    pass


def clean_credential(value: Any) -> str:
    raw = str(value or "").replace("\r", "").replace("\n", "").strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        raw = raw[1:-1].strip()
    return raw


def collapse_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [row for row in data if isinstance(row, dict)]


def write_json(path: Path, data: Any, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    else:
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
    path.write_text(payload, encoding="utf-8")


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


def rotating_slice(items: Sequence[Candidate], start_index: int, limit: int) -> Tuple[List[Candidate], int]:
    if not items or limit <= 0:
        return [], 0
    start = start_index % len(items)
    output: List[Candidate] = []
    for offset in range(min(limit, len(items))):
        output.append(items[(start + offset) % len(items)])
    return output, (start + len(output)) % len(items)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch eBay marketplace deals into dist/deals JSON artifacts.")
    parser.add_argument("--sets-json", default="dist/Lego Star Wars Database.json")
    parser.add_argument("--minifigs-json", default="dist/Lego-Star-Wars-Minifigure-Database.json")
    parser.add_argument("--parts-json", default="dist/parts/parts-catalog.json")
    parser.add_argument("--output-dir", default="dist/deals")
    parser.add_argument("--fallback-output", default="dist/marketplace-deals.json")
    parser.add_argument("--state-path", default="dist/ebay-market-state.json")
    parser.add_argument("--client-id", default=os.getenv("EBAY_CLIENT_ID", ""))
    parser.add_argument("--client-secret", default=os.getenv("EBAY_CLIENT_SECRET", ""))
    parser.add_argument("--regions", default=os.getenv("EBAY_MARKET_REGIONS", "UK,US"))
    parser.add_argument("--sets-per-region", type=int, default=int(os.getenv("EBAY_SETS_PER_REGION", "50")))
    parser.add_argument("--minifigs-per-region", type=int, default=int(os.getenv("EBAY_MINIFIGS_PER_REGION", "35")))
    parser.add_argument("--parts-per-region", type=int, default=int(os.getenv("EBAY_PARTS_PER_REGION", "25")))
    parser.add_argument("--max-results-per-item", type=int, default=int(os.getenv("EBAY_MAX_RESULTS_PER_ITEM", "5")))
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--priority-themes", default=os.getenv("EBAY_PRIORITY_THEMES", "Star Wars,Marvel Super Heroes,Disney,NINJAGO"))
    parser.add_argument("--priority-minifig-categories", default=os.getenv("EBAY_PRIORITY_MINIFIG_CATEGORIES", "Star Wars,Marvel Super Heroes,Disney,NINJAGO"))
    parser.add_argument("--priority-part-categories", default=os.getenv("EBAY_PRIORITY_PART_CATEGORIES", "Bricks,Plates,Tiles,Minifigure"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def parse_priority_list(value: str) -> List[str]:
    return [collapse_ws(part) for part in re.split(r"[,\n]", value or "") if collapse_ws(part)]


def prioritized_candidates(
    rows: Sequence[Dict[str, Any]],
    *,
    item_type: str,
    priority_values: Sequence[str],
) -> List[Candidate]:
    priority_set = {value.casefold() for value in priority_values if value}
    prioritized: List[Candidate] = []
    remainder: List[Candidate] = []

    for row in rows:
        candidate = make_candidate(row, item_type=item_type)
        if candidate is None:
            continue
        bucket = prioritized if candidate.category.casefold() in priority_set else remainder
        bucket.append(candidate)
    return prioritized + remainder


def make_candidate(row: Dict[str, Any], *, item_type: str) -> Optional[Candidate]:
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

    compact_number = number.split("-", 1)[0] if item_type == "set" else number
    trimmed_name = " ".join(name.split()[:6])
    search_term = collapse_ws(f"LEGO {compact_number} {trimmed_name}")
    return Candidate(number=number, name=name, item_type=item_type, category=category, search_term=search_term)


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def oauth_token(session: requests.Session, client_id: str, client_secret: str, timeout: float) -> str:
    client_id = clean_credential(client_id)
    client_secret = clean_credential(client_secret)
    if not client_id or not client_secret:
        raise EbayApiError("Missing EBAY_CLIENT_ID or EBAY_CLIENT_SECRET")
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
    response = session.post(
        OAUTH_URL,
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "client_credentials",
            "scope": APP_SCOPE,
        },
        timeout=timeout,
    )
    if response.status_code >= 400:
        if response.status_code == 401:
            raise EbayApiError(
                "OAuth failed: HTTP 401 invalid_client. eBay rejected the client credentials. "
                "Use the Production App ID as EBAY_CLIENT_ID and the matching Production Cert ID "
                "as EBAY_CLIENT_SECRET, and make sure no quotes or trailing newlines were pasted "
                "into the GitHub secret values."
            )
        raise EbayApiError(f"OAuth failed: HTTP {response.status_code} {response.text[:200]}")
    data = response.json()
    token = collapse_ws(data.get("access_token"))
    if not token:
        raise EbayApiError("OAuth succeeded but access_token missing")
    return token


def request_search(
    session: requests.Session,
    token: str,
    marketplace_id: str,
    query: str,
    *,
    limit: int,
    timeout: float,
    retries: int,
) -> List[Dict[str, Any]]:
    params = {
        "q": query,
        "limit": str(max(1, min(limit, 50))),
        "filter": "buyingOptions:{FIXED_PRICE}",
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "X-EBAY-C-MARKETPLACE-ID": marketplace_id,
    }

    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(BROWSE_SEARCH_URL, headers=headers, params=params, timeout=timeout)
        except requests.RequestException as exc:
            if attempt == attempts:
                raise EbayApiError(f"Search failed for '{query}': {exc}") from exc
            time.sleep(min(8.0, attempt * 1.5))
            continue

        if response.status_code == 429 or response.status_code >= 500:
            if attempt == attempts:
                raise EbayApiError(f"Search failed for '{query}': HTTP {response.status_code}")
            time.sleep(min(15.0, attempt * 2.0))
            continue
        if response.status_code >= 400:
            raise EbayApiError(f"Search failed for '{query}': HTTP {response.status_code} {response.text[:200]}")

        payload = response.json()
        items = payload.get("itemSummaries")
        return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []

    return []


def title_matches_candidate(title: str, candidate: Candidate) -> bool:
    lowered = title.casefold()
    number = candidate.number.casefold()
    compact_number = number.split("-", 1)[0]
    if number in lowered or compact_number in lowered:
        return True
    if candidate.item_type == "part" and compact_number in lowered:
        return True
    return False


def parse_deal(item: Dict[str, Any], candidate: Candidate, region: str) -> Optional[Dict[str, Any]]:
    title = collapse_ws(item.get("title"))
    url = collapse_ws(item.get("itemWebUrl"))
    if not title or not url:
        return None
    if not title_matches_candidate(title, candidate):
        return None

    price = item.get("price") if isinstance(item.get("price"), dict) else {}
    currency_code = collapse_ws(price.get("currency")) or None
    raw_value = collapse_ws(price.get("value"))
    price_value = None
    if raw_value:
        try:
            price_value = float(raw_value)
        except ValueError:
            price_value = None
    if price_value is None:
        return None

    subtitle_parts = [
        collapse_ws(item.get("condition")),
        collapse_ws(item.get("itemLocationCountry")),
    ]
    subtitle = " • ".join(part for part in subtitle_parts if part) or None

    listed_at = collapse_ws(item.get("itemOriginDate") or item.get("creationDate")) or None
    return {
        "id": collapse_ws(item.get("itemId")) or f"{candidate.number.lower()}|{region.lower()}|{url.lower()}",
        "number": candidate.number,
        "source": "eBay",
        "title": title,
        "subtitle": subtitle,
        "priceValue": round(price_value, 2),
        "priceText": f"{currency_code or ''} {price_value:.2f}".strip(),
        "currencyCode": currency_code,
        "url": url,
        "regionCode": region,
        "listedAt": listed_at,
    }


def merge_deals(existing: Iterable[Dict[str, Any]], incoming: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for deal in list(existing) + list(incoming):
        deal_id = collapse_ws(deal.get("id")).lower()
        if not deal_id:
            continue
        merged[deal_id] = deal
    return sorted(
        merged.values(),
        key=lambda row: (
            collapse_ws(row.get("number")).casefold(),
            float(row.get("priceValue") or 0.0),
            collapse_ws(row.get("title")).casefold(),
        ),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    sets = load_json_array(Path(args.sets_json))
    minifigs = load_json_array(Path(args.minifigs_json))
    parts_path = Path(args.parts_json)
    parts = load_json_array(parts_path) if parts_path.exists() else []
    output_dir = Path(args.output_dir)
    fallback_output = Path(args.fallback_output)
    state_path = Path(args.state_path)

    regions = [normalize_region(value) for value in re.split(r"[,\s]+", args.regions) if collapse_ws(value)]
    regions = [region for region in regions if region in MARKETPLACE_BY_REGION]
    if not regions:
        raise EbayApiError("No supported eBay regions configured")

    set_candidates = prioritized_candidates(sets, item_type="set", priority_values=parse_priority_list(args.priority_themes))
    minifig_candidates = prioritized_candidates(minifigs, item_type="minifig", priority_values=parse_priority_list(args.priority_minifig_categories))
    part_candidates = prioritized_candidates(parts, item_type="part", priority_values=parse_priority_list(args.priority_part_categories))

    state = load_state(state_path)
    region_state = state.get("regions") if isinstance(state.get("regions"), dict) else {}

    session = requests.Session()
    token = oauth_token(session, args.client_id, args.client_secret, args.timeout)

    output_by_region: Dict[str, List[Dict[str, Any]]] = {}
    next_region_state: Dict[str, Dict[str, Any]] = {}

    for region in regions:
        marketplace_id = MARKETPLACE_BY_REGION[region]
        current_state = region_state.get(region) if isinstance(region_state.get(region), dict) else {}
        set_start = int(current_state.get("nextSetIndex", 0) or 0)
        minifig_start = int(current_state.get("nextMinifigIndex", 0) or 0)
        part_start = int(current_state.get("nextPartIndex", 0) or 0)

        selected_sets, next_set_index = rotating_slice(set_candidates, set_start, args.sets_per_region)
        selected_minifigs, next_minifig_index = rotating_slice(minifig_candidates, minifig_start, args.minifigs_per_region)
        selected_parts, next_part_index = rotating_slice(part_candidates, part_start, args.parts_per_region)

        region_deals: List[Dict[str, Any]] = []
        for candidate in selected_sets + selected_minifigs + selected_parts:
            items = request_search(
                session,
                token,
                marketplace_id,
                candidate.search_term,
                limit=args.max_results_per_item,
                timeout=args.timeout,
                retries=args.retries,
            )
            parsed = [deal for deal in (parse_deal(item, candidate, region) for item in items) if deal is not None]
            region_deals = merge_deals(region_deals, parsed)
            if args.verbose:
                print(f"[eBay:{region}] {candidate.item_type} {candidate.number} -> {len(parsed)} deals", flush=True)
            time.sleep(0.15)

        output_by_region[region] = region_deals
        next_region_state[region] = {
            "nextSetIndex": next_set_index,
            "nextMinifigIndex": next_minifig_index,
            "nextPartIndex": next_part_index,
            "updatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    for region, deals in output_by_region.items():
        filename = OUTPUT_FILENAME_BY_REGION[region]
        write_json(output_dir / filename, deals)

    all_deals = []
    for deals in output_by_region.values():
        all_deals.extend(deals)
    write_json(output_dir / "all.json", merge_deals([], all_deals))
    write_json(
        fallback_output,
        {
            "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "regions": output_by_region,
        },
        pretty=True,
    )
    write_json(
        state_path,
        {
            "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "regions": next_region_state,
        },
        pretty=True,
    )

    for region, deals in output_by_region.items():
        print(f"[eBay:{region}] deals={len(deals)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
