#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


POPULAR_THEMES: List[str] = [
    "Star Wars",
    "Ultimate Collector Series",
    "Harry Potter",
    "Technic",
    "Botanicals",
    "City",
    "Friends",
    "Ninjago",
    "Disney",
    "Marvel",
    "Icons",
    "Creator 3-in-1",
    "Speed Champions",
    "Jurassic World",
    "Minecraft",
    "Architecture",
    "Animal Crossing",
    "Super Mario",
    "Dreamzzz",
    "Batman",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select release-priority BrickLink bootstrap targets.")
    parser.add_argument("--sets-json", default="dist/Lego Star Wars Database.json")
    parser.add_argument("--deals-json", default="dist/deals/uk.json")
    parser.add_argument("--output-path", default="dist/bootstrap-release-targets.txt")
    parser.add_argument("--metadata-path", default="dist/bootstrap-release-targets.json")
    parser.add_argument("--max-total", type=int, default=1500)
    parser.add_argument("--per-theme-cap", type=int, default=40)
    parser.add_argument("--popular-theme-limit", type=int, default=90)
    parser.add_argument("--active-fallback-limit", type=int, default=500)
    parser.add_argument("--new-sets-limit", type=int, default=120)
    parser.add_argument("--coming-soon-limit", type=int, default=80)
    parser.add_argument("--just-announced-limit", type=int, default=80)
    parser.add_argument("--best-deals-limit", type=int, default=120)
    parser.add_argument("--top-performers-limit", type=int, default=120)
    parser.add_argument("--highest-value-limit", type=int, default=120)
    parser.add_argument("--under-rrp-limit", type=int, default=120)
    parser.add_argument("--retiring-soon-limit", type=int, default=120)
    return parser.parse_args(argv)


def collapse_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def parse_int(value: Any) -> Optional[int]:
    text = collapse_ws(value)
    if not text:
        return None
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def parse_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = collapse_ws(value)
    if not text:
        return None
    match = re.search(r"-?[0-9][0-9,]*(?:\.[0-9]+)?", text.replace("~", ""))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def parse_date(value: Any) -> Optional[datetime]:
    text = collapse_ws(value)
    if not text:
        return None
    for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def parse_list(data: Any) -> List[Dict[str, Any]]:
    return data if isinstance(data, list) else []


def normalize_set_code(number: Any, variant: Any) -> str:
    number_text = collapse_ws(number)
    if not number_text:
        return ""
    variant_value = parse_int(variant) or 1
    return f"{number_text}-{variant_value}"


def is_buildable(row: Dict[str, Any]) -> bool:
    pieces = parse_int(row.get("Pieces")) or 0
    return pieces > 0


def theme_name(row: Dict[str, Any]) -> str:
    return collapse_ws(row.get("Theme"))


def availability_text(row: Dict[str, Any]) -> str:
    return collapse_ws(row.get("Availability")).lower()


def retail_amount(row: Dict[str, Any]) -> Optional[float]:
    for key in ("UKRetailPrice", "CurrentRRPBaseline", "RRP", "RetailPrice", "USRetailPrice", "CARetailPrice", "DERetailPrice"):
        value = parse_float(row.get(key))
        if value is not None and value > 0:
            return value
    return None


def current_new_amount(row: Dict[str, Any]) -> Optional[float]:
    for key in ("BrickLinkCurrentNewAvgPrice", "BrickLinkLatestSaleNewPrice", "BrickLinkSoldPriceNew", "New"):
        value = parse_float(row.get(key))
        if value is not None and value > 0:
            return value
    return None


def current_used_amount(row: Dict[str, Any]) -> Optional[float]:
    for key in ("BrickLinkCurrentUsedAvgPrice", "BrickLinkLatestSaleUsedPrice", "BrickLinkSoldPriceUsed", "Used"):
        value = parse_float(row.get(key))
        if value is not None and value > 0:
            return value
    return None


def current_value_amount(row: Dict[str, Any]) -> Optional[float]:
    for getter in (current_new_amount, current_used_amount):
        value = getter(row)
        if value is not None and value > 0:
            return value
    return retail_amount(row)


def is_coming_soon(row: Dict[str, Any], *, now: datetime) -> bool:
    launch_date = parse_date(row.get("LaunchDate"))
    if launch_date and launch_date > now:
        return True
    availability = availability_text(row)
    return any(token in availability for token in ("pre-order", "preorder", "coming soon", "coming", "not yet released"))


def is_just_announced(row: Dict[str, Any]) -> bool:
    text = " ".join(
        collapse_ws(row.get(key)).lower()
        for key in ("Availability", "Category", "Subtheme", "Collection")
        if collapse_ws(row.get(key))
    )
    return any(token in text for token in ("just announced", "announced", "reveal", "revealed", "rumor", "rumour"))


def exit_date(row: Dict[str, Any]) -> Optional[datetime]:
    return parse_date(row.get("ExitDate"))


def is_retired(row: Dict[str, Any], *, now: datetime) -> bool:
    end = exit_date(row)
    if end and end < now:
        return True
    availability = availability_text(row)
    return any(token in availability for token in ("retired", "discontinued", "not sold"))


def is_currently_available(row: Dict[str, Any], *, now: datetime) -> bool:
    if is_coming_soon(row, now=now):
        return False
    if is_retired(row, now=now):
        return False
    availability = availability_text(row)
    if any(token in availability for token in ("retail", "exclusive", "insiders")):
        return True
    end = exit_date(row)
    if end and end >= now:
        return True
    return False


def retiring_soon(row: Dict[str, Any], *, now: datetime) -> bool:
    end = exit_date(row)
    if not end:
        return False
    if end < now:
        return False
    if is_coming_soon(row, now=now):
        return False
    return end <= (now + timedelta(days=365))


def launch_sort_key(row: Dict[str, Any], *, now: datetime) -> Tuple[int, int, str]:
    launch = parse_date(row.get("LaunchDate"))
    if launch:
        return (launch.year, int(launch.timestamp()), normalize_set_code(row.get("Number"), row.get("Variant")))
    year = parse_int(row.get("Year released")) or parse_int(row.get("YearReleased")) or 0
    return (year, 0, normalize_set_code(row.get("Number"), row.get("Variant")))


def recent_release(row: Dict[str, Any], *, now: datetime) -> bool:
    launch = parse_date(row.get("LaunchDate"))
    if launch:
        return launch >= (now - timedelta(days=550))
    year = parse_int(row.get("Year released")) or parse_int(row.get("YearReleased"))
    if year is None:
        return False
    return year >= (now.year - 1)


def top_performer_tuple(row: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    retail = retail_amount(row)
    current = current_value_amount(row)
    if retail is None or current is None or retail <= 0:
        return None
    delta = current - retail
    if delta <= 0:
        return None
    percent = (delta / retail) * 100.0
    return (delta, percent)


def under_rrp_percent(row: Dict[str, Any], *, now: datetime) -> Optional[float]:
    if not is_currently_available(row, now=now):
        return None
    retail = retail_amount(row)
    current = current_new_amount(row)
    if retail is None or current is None or retail <= 0 or current <= 0:
        return None
    discount = ((retail - current) / retail) * 100.0
    return discount if discount > 0 else None


def load_deals(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    best: Dict[str, float] = {}
    if not isinstance(data, list):
        return best
    for entry in data:
        if not isinstance(entry, dict):
            continue
        number = collapse_ws(entry.get("number"))
        price = parse_float(entry.get("priceValue"))
        if not number or price is None or price <= 0:
            continue
        current = best.get(number)
        if current is None or price < current:
            best[number] = price
    return best


def best_deal_discount(row: Dict[str, Any], *, now: datetime, deals_by_number: Dict[str, float]) -> Optional[float]:
    if not is_currently_available(row, now=now):
        return None
    retail = retail_amount(row)
    if retail is None or retail <= 0:
        return None
    number = normalize_set_code(row.get("Number"), row.get("Variant"))
    deal_price = deals_by_number.get(number)
    if deal_price is None or deal_price <= 0:
        return None
    discount = ((retail - deal_price) / retail) * 100.0
    return discount if discount > 0 else None


def add_list(
    ordered: List[str],
    reasons: Dict[str, List[str]],
    items: Iterable[Dict[str, Any]],
    reason: str,
    *,
    max_total: int,
    per_theme_cap: int,
    theme_counts: Dict[str, int],
) -> None:
    seen = set(ordered)
    for row in items:
        key = normalize_set_code(row.get("Number"), row.get("Variant"))
        if not key:
            continue
        reasons[key].append(reason)
        if key in seen:
            continue
        theme = theme_name(row) or "Other"
        if per_theme_cap > 0 and theme_counts.get(theme, 0) >= per_theme_cap:
            continue
        if len(ordered) >= max_total:
            return
        ordered.append(key)
        seen.add(key)
        theme_counts[theme] = theme_counts.get(theme, 0) + 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    now = datetime.now(timezone.utc)
    rows = parse_list(json.loads(Path(args.sets_json).read_text(encoding="utf-8")))
    deals_by_number = load_deals(Path(args.deals_json))

    buildable_rows = [row for row in rows if is_buildable(row)]
    popular_theme_rank = {theme.lower(): idx for idx, theme in enumerate(POPULAR_THEMES)}

    ordered: List[str] = []
    reasons: Dict[str, List[str]] = defaultdict(list)
    theme_counts: Dict[str, int] = {}

    new_sets = sorted(
        [row for row in buildable_rows if recent_release(row, now=now)],
        key=lambda row: launch_sort_key(row, now=now),
        reverse=True,
    )[: args.new_sets_limit]
    add_list(ordered, reasons, new_sets, "homepage:new_sets", max_total=args.max_total, per_theme_cap=args.per_theme_cap, theme_counts=theme_counts)

    coming_soon = sorted(
        [row for row in buildable_rows if is_coming_soon(row, now=now)],
        key=lambda row: launch_sort_key(row, now=now),
    )[: args.coming_soon_limit]
    add_list(ordered, reasons, coming_soon, "homepage:coming_soon", max_total=args.max_total, per_theme_cap=args.per_theme_cap, theme_counts=theme_counts)

    just_announced = sorted(
        [row for row in buildable_rows if is_just_announced(row)],
        key=lambda row: launch_sort_key(row, now=now),
    )[: args.just_announced_limit]
    add_list(ordered, reasons, just_announced, "homepage:just_announced", max_total=args.max_total, per_theme_cap=args.per_theme_cap, theme_counts=theme_counts)

    best_deals = sorted(
        [
            (row, best_deal_discount(row, now=now, deals_by_number=deals_by_number))
            for row in buildable_rows
        ],
        key=lambda pair: (pair[1] if pair[1] is not None else -1.0, normalize_set_code(pair[0].get("Number"), pair[0].get("Variant"))),
        reverse=True,
    )
    add_list(
        ordered,
        reasons,
        [row for row, discount in best_deals if discount is not None][: args.best_deals_limit],
        "homepage:best_deals",
        max_total=args.max_total,
        per_theme_cap=args.per_theme_cap,
        theme_counts=theme_counts,
    )

    top_performers = sorted(
        [
            (row, top_performer_tuple(row))
            for row in buildable_rows
        ],
        key=lambda pair: (
            pair[1][0] if pair[1] is not None else -1.0,
            pair[1][1] if pair[1] is not None else -1.0,
            normalize_set_code(pair[0].get("Number"), pair[0].get("Variant")),
        ),
        reverse=True,
    )
    add_list(
        ordered,
        reasons,
        [row for row, delta in top_performers if delta is not None][: args.top_performers_limit],
        "homepage:top_performers",
        max_total=args.max_total,
        per_theme_cap=args.per_theme_cap,
        theme_counts=theme_counts,
    )

    highest_value = sorted(
        [
            (row, current_value_amount(row))
            for row in buildable_rows
        ],
        key=lambda pair: (pair[1] if pair[1] is not None else -1.0, normalize_set_code(pair[0].get("Number"), pair[0].get("Variant"))),
        reverse=True,
    )
    add_list(
        ordered,
        reasons,
        [row for row, value in highest_value if value is not None][: args.highest_value_limit],
        "homepage:highest_value",
        max_total=args.max_total,
        per_theme_cap=args.per_theme_cap,
        theme_counts=theme_counts,
    )

    under_rrp = sorted(
        [
            (row, under_rrp_percent(row, now=now))
            for row in buildable_rows
        ],
        key=lambda pair: (pair[1] if pair[1] is not None else -1.0, normalize_set_code(pair[0].get("Number"), pair[0].get("Variant"))),
        reverse=True,
    )
    add_list(
        ordered,
        reasons,
        [row for row, percent in under_rrp if percent is not None][: args.under_rrp_limit],
        "homepage:below_rrp",
        max_total=args.max_total,
        per_theme_cap=args.per_theme_cap,
        theme_counts=theme_counts,
    )

    retiring = sorted(
        [row for row in buildable_rows if retiring_soon(row, now=now)],
        key=lambda row: (exit_date(row) or datetime.max.replace(tzinfo=timezone.utc), normalize_set_code(row.get("Number"), row.get("Variant"))),
    )[: args.retiring_soon_limit]
    add_list(ordered, reasons, retiring, "homepage:retiring_soon", max_total=args.max_total, per_theme_cap=args.per_theme_cap, theme_counts=theme_counts)

    for theme in POPULAR_THEMES:
        theme_rows = [
            row for row in buildable_rows
            if theme_name(row).lower() == theme.lower() and is_currently_available(row, now=now)
        ]
        theme_rows.sort(
            key=lambda row: (
                launch_sort_key(row, now=now),
                current_value_amount(row) or 0.0,
                normalize_set_code(row.get("Number"), row.get("Variant")),
            ),
            reverse=True,
        )
        add_list(
            ordered,
            reasons,
            theme_rows[: args.popular_theme_limit],
            f"popular_theme:{theme}",
            max_total=args.max_total,
            per_theme_cap=args.per_theme_cap,
            theme_counts=theme_counts,
        )
        if len(ordered) >= args.max_total:
            break

    active_fallback = [
        row for row in buildable_rows
        if is_currently_available(row, now=now)
    ]
    active_fallback.sort(
        key=lambda row: (
            recent_release(row, now=now),
            -(popular_theme_rank.get(theme_name(row).lower(), 999)),
            launch_sort_key(row, now=now),
            current_value_amount(row) or 0.0,
            normalize_set_code(row.get("Number"), row.get("Variant")),
        ),
        reverse=True,
    )
    add_list(
        ordered,
        reasons,
        active_fallback[: args.active_fallback_limit],
        "active_fallback",
        max_total=args.max_total,
        per_theme_cap=args.per_theme_cap,
        theme_counts=theme_counts,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(ordered) + ("\n" if ordered else ""), encoding="utf-8")

    rows_by_key = {normalize_set_code(row.get("Number"), row.get("Variant")): row for row in buildable_rows}
    metadata = {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "targetCount": len(ordered),
        "maxTotal": args.max_total,
        "perThemeCap": args.per_theme_cap,
        "popularThemes": POPULAR_THEMES,
        "targets": [
            {
                "itemNo": key,
                "number": collapse_ws(rows_by_key.get(key, {}).get("Number")),
                "variant": parse_int(rows_by_key.get(key, {}).get("Variant")) or 1,
                "theme": theme_name(rows_by_key.get(key, {})),
                "name": collapse_ws(rows_by_key.get(key, {}).get("Name")),
                "availability": collapse_ws(rows_by_key.get(key, {}).get("Availability")),
                "reasons": reasons.get(key, []),
            }
            for key in ordered
        ],
    }
    metadata_path = Path(args.metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[ReleaseTargets] total={len(ordered)} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
