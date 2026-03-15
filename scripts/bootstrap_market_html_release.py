#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import local
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import update_market_values as market


@dataclass(frozen=True)
class BootstrapConfig:
    timeout: float
    retries: int
    delay: float
    jitter: float
    verbose: bool
    currency_code: str
    fallback_currency_codes: Tuple[str, ...]
    allow_html_fallback: bool = True


@dataclass(frozen=True)
class Task:
    label: str
    item_type: str
    index: int
    row: Dict[str, Any]


@dataclass
class TaskResult:
    task: Task
    row: Dict[str, Any]
    ok: bool
    changed: bool
    no_price_data: bool
    fetch_failure: bool
    parse_miss: bool
    resolved_item_no: str
    resolved_item_type: str


THREAD_STATE = local()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast HTML-only BrickLink market bootstrap for release prep.")
    parser.add_argument("--sets-json", default="dist/Lego Star Wars Database.json")
    parser.add_argument("--minifigs-json", default="dist/Lego-Star-Wars-Minifigure-Database.json")
    parser.add_argument("--parts-json", default="dist/parts/parts-catalog.json")
    parser.add_argument("--market-details-dir", default="dist/market-details")
    parser.add_argument("--set-aliases-json", default="dist/bricklink-set-aliases.json")
    parser.add_argument(
        "--item-type",
        choices=["set", "minifig", "part", "both", "all"],
        default="set",
        help="Catalog item type(s) to bootstrap. Default set.",
    )
    parser.add_argument("--currency-code", default="GBP")
    parser.add_argument("--fallback-currencies", default="USD,EUR,GBP")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--jitter", type=float, default=0.03)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--missing-only", action="store_true", help="Only refresh rows missing current values or history.")
    parser.add_argument("--skip-cross-enrichment", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def get_thread_resources(cfg: BootstrapConfig) -> Tuple[market.BrickLinkClient, market.RuntimeThrottle]:
    client = getattr(THREAD_STATE, "client", None)
    throttle = getattr(THREAD_STATE, "throttle", None)
    if client is None or throttle is None:
        client = market.BrickLinkClient(
            consumer_key="HTML_ONLY",
            consumer_secret="HTML_ONLY",
            token_value="HTML_ONLY",
            token_secret="HTML_ONLY",
            timeout=cfg.timeout,
            retries=cfg.retries,
            verbose=cfg.verbose,
            request_budget=market.ApiRequestBudget(max_calls=None),
        )
        # Force apply_market_to_row down the HTML path.
        client.auth_failed = True
        throttle = market.RuntimeThrottle(min_delay=cfg.delay, jitter=cfg.jitter)
        THREAD_STATE.client = client
        THREAD_STATE.throttle = throttle
    return client, throttle


def load_set_alias_cache(path: Path) -> Dict[str, str]:
    data = market.load_json_object(path)
    output: Dict[str, str] = {}
    for raw_key, raw_value in data.items():
        key = market.normalize_set_code(raw_key, 1).lower() if market.collapse_ws(raw_key) else ""
        value = market.canonicalize_set_item_no(raw_value)
        if key and value:
            output[key] = value
    return output


def row_needs_refresh(row: Dict[str, Any], *, item_type: str) -> bool:
    if item_type == "PART":
        has_new = bool(market.collapse_ws(row.get("market_price_new")))
        has_used = bool(market.collapse_ws(row.get("market_price_used")))
        return not (has_new or has_used)

    has_new = bool(market.collapse_ws(row.get("New")))
    has_used = bool(market.collapse_ws(row.get("Used")))
    has_history_new = isinstance(row.get("BrickLinkTransactionsNew"), list) and bool(row.get("BrickLinkTransactionsNew"))
    has_history_used = isinstance(row.get("BrickLinkTransactionsUsed"), list) and bool(row.get("BrickLinkTransactionsUsed"))
    return not ((has_new or has_used) and (has_history_new or has_history_used))


def build_tasks(
    rows: List[Dict[str, Any]],
    *,
    label: str,
    item_type: str,
    start_index: int,
    limit: Optional[int],
    missing_only: bool,
) -> List[Task]:
    tasks: List[Task] = []
    for idx, row in enumerate(rows):
        if idx < start_index:
            continue
        if missing_only and not row_needs_refresh(row, item_type=item_type):
            continue
        tasks.append(Task(label=label, item_type=item_type, index=idx, row=row))
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def build_item_candidates(
    row: Dict[str, Any],
    *,
    item_type_upper: str,
    set_alias_cache: Dict[str, str],
) -> Tuple[List[Tuple[str, str]], str]:
    canonical_set_code = ""
    if item_type_upper == "SET":
        canonical_set_code = market.normalize_set_code(row.get("Number"), row.get("Variant")).lower()
        cached_alias = set_alias_cache.get(canonical_set_code) if canonical_set_code else None
        candidates = market.build_set_item_candidates(
            row.get("Number"),
            row.get("Variant"),
            row.get("link"),
            row.get("BrickLinkPriceGuideURL"),
            alias_set_code=cached_alias,
        )
        return ([("SET", value) for value in candidates], canonical_set_code)

    if item_type_upper == "MINIFIG":
        candidates = market.build_minifig_item_candidates(
            row.get("Number"),
            row.get("link"),
            row.get("BrickLinkPriceGuideURL"),
        )
        return ([("MINIFIG", value) for value in candidates], canonical_set_code)

    candidates = market.build_part_item_candidates(
        row.get("part_num"),
        row.get("link"),
        row.get("market_price_guide_url"),
        row.get("bricklink_part_num"),
    )
    return ([("PART", value) for value in candidates], canonical_set_code)


def process_task(
    task: Task,
    *,
    cfg: BootstrapConfig,
    market_details_dir: Optional[Path],
    month_key: str,
    run_started_at: datetime,
    set_alias_cache: Dict[str, str],
) -> TaskResult:
    client, throttle = get_thread_resources(cfg)
    row = copy.deepcopy(task.row)
    before = json.dumps(row, sort_keys=True, ensure_ascii=False)
    item_type_upper = market.collapse_ws(task.item_type).upper()

    item_candidates, _ = build_item_candidates(
        row,
        item_type_upper=item_type_upper,
        set_alias_cache=set_alias_cache,
    )
    if not item_candidates:
        return TaskResult(
            task=task,
            row=row,
            ok=False,
            changed=False,
            no_price_data=False,
            fetch_failure=False,
            parse_miss=True,
            resolved_item_no="",
            resolved_item_type=item_type_upper,
        )

    ok = False
    used_item_no = item_candidates[0][1]
    used_item_type = item_type_upper
    for candidate_type, candidate_no in item_candidates:
        preserved_detail_row = market.load_existing_market_detail(
            market_details_dir,
            item_type=candidate_type,
            item_no=candidate_no,
            cache={},
        )
        ok = market.apply_market_to_row(
            row,
            item_type=candidate_type,
            item_no=candidate_no,
            preserved_detail_row=preserved_detail_row,
            currency_code=cfg.currency_code,
            fallback_currency_codes=cfg.fallback_currency_codes,
            allow_html_fallback=True,
            client=client,
            throttle=throttle,
            month_key=month_key,
        )
        used_item_no = candidate_no
        used_item_type = candidate_type
        if ok:
            break

    parse_miss = False
    no_price_data = False
    fetch_failure = False
    if not ok:
        status_value = market.infer_fetch_status(client.last_error_kind, client.last_http_status)
        row["MarketFetchStatus"] = status_value
        row["MarketLastUpdatedUTC"] = run_started_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        row["MarketNoDataRetryAfterUTC"] = None
        no_price_data = client.last_error_kind in {"not_found", "no_data"}
        parse_miss = client.last_error_kind == ""
        fetch_failure = not no_price_data and not parse_miss

    after = json.dumps(row, sort_keys=True, ensure_ascii=False)
    return TaskResult(
        task=task,
        row=row,
        ok=ok,
        changed=(before != after),
        no_price_data=no_price_data,
        fetch_failure=fetch_failure,
        parse_miss=parse_miss,
        resolved_item_no=used_item_no,
        resolved_item_type=used_item_type,
    )


def run_tasks(
    rows: List[Dict[str, Any]],
    *,
    label: str,
    item_type: str,
    cfg: BootstrapConfig,
    tasks: List[Task],
    market_details_dir: Optional[Path],
    month_key: str,
    run_started_at: datetime,
    set_alias_cache: Dict[str, str],
    workers: int,
) -> market.FileUpdateStats:
    stats = market.FileUpdateStats(total_rows=len(rows))
    if not tasks:
        return stats

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [
            pool.submit(
                process_task,
                task,
                cfg=cfg,
                market_details_dir=market_details_dir,
                month_key=month_key,
                run_started_at=run_started_at,
                set_alias_cache=set_alias_cache,
            )
            for task in tasks
        ]

        for future in as_completed(futures):
            result = future.result()
            completed += 1
            rows[result.task.index] = result.row
            stats.rows_considered += 1
            stats.last_index_processed = result.task.index
            stats.processed_indices.append(result.task.index)
            if result.ok:
                stats.rows_succeeded += 1
            if result.changed:
                stats.rows_changed += 1
            if result.no_price_data:
                stats.no_price_data_skips += 1
            if result.fetch_failure:
                stats.fetch_failures += 1
            if result.parse_miss:
                stats.parse_misses += 1

            if cfg.verbose and (completed <= 10 or completed % 100 == 0 or completed == len(tasks)):
                status = "updated" if result.ok else "missed"
                resolved = result.resolved_item_no or "n/a"
                print(
                    f"[{label}] {completed}/{len(tasks)}: {result.task.index} -> {resolved} ({result.resolved_item_type}) {status}",
                    flush=True,
                )

    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    item_mode = market.collapse_ws(args.item_type).lower() or "set"
    do_sets = item_mode in {"set", "both", "all"}
    do_minifigs = item_mode in {"minifig", "both", "all"}
    do_parts = item_mode in {"part", "all"}

    sets_path = Path(args.sets_json)
    minifigs_path = Path(args.minifigs_json)
    parts_path = Path(args.parts_json)
    market_details_dir = Path(args.market_details_dir)
    set_alias_cache = load_set_alias_cache(Path(args.set_aliases_json))

    if do_sets and not sets_path.exists():
        print(f"Missing sets JSON: {sets_path}", file=sys.stderr)
        return 1
    if do_minifigs and not minifigs_path.exists():
        print(f"Missing minifigs JSON: {minifigs_path}", file=sys.stderr)
        return 1
    if do_parts and not parts_path.exists():
        print(f"Missing parts JSON: {parts_path}", file=sys.stderr)
        return 1

    cfg = BootstrapConfig(
        timeout=max(1.0, args.timeout),
        retries=max(0, args.retries),
        delay=max(0.0, args.delay),
        jitter=max(0.0, args.jitter),
        verbose=bool(args.verbose),
        currency_code=market.collapse_ws(args.currency_code).upper() or "GBP",
        fallback_currency_codes=tuple(
            code
            for code in (
                market.normalize_currency_code(part)
                for part in re.split(r"[,\s]+", str(args.fallback_currencies or ""))
            )
            if code
        ),
    )

    run_started_at = datetime.now(timezone.utc)
    month_key = run_started_at.strftime("%Y-%m")
    workers = max(1, args.workers)

    sets_rows = market.load_json_array(sets_path) if do_sets else []
    minifigs_rows = market.load_json_array(minifigs_path) if do_minifigs else []
    parts_rows = market.load_json_array(parts_path) if do_parts else []

    set_tasks = build_tasks(
        sets_rows,
        label="Sets",
        item_type="SET",
        start_index=max(0, args.start_index),
        limit=args.limit,
        missing_only=bool(args.missing_only),
    ) if do_sets else []
    minifig_tasks = build_tasks(
        minifigs_rows,
        label="Minifigs",
        item_type="MINIFIG",
        start_index=max(0, args.start_index),
        limit=args.limit,
        missing_only=bool(args.missing_only),
    ) if do_minifigs else []
    part_tasks = build_tasks(
        parts_rows,
        label="Parts",
        item_type="PART",
        start_index=max(0, args.start_index),
        limit=args.limit,
        missing_only=bool(args.missing_only),
    ) if do_parts else []

    if cfg.verbose:
        print(
            f"[BootstrapHTML] workers={workers} sets={len(set_tasks)} minifigs={len(minifig_tasks)} parts={len(part_tasks)}",
            flush=True,
        )

    sets_stats = run_tasks(
        sets_rows,
        label="Sets",
        item_type="SET",
        cfg=cfg,
        tasks=set_tasks,
        market_details_dir=market_details_dir,
        month_key=month_key,
        run_started_at=run_started_at,
        set_alias_cache=set_alias_cache,
        workers=workers,
    ) if do_sets else market.FileUpdateStats(total_rows=0)

    minifigs_stats = run_tasks(
        minifigs_rows,
        label="Minifigs",
        item_type="MINIFIG",
        cfg=cfg,
        tasks=minifig_tasks,
        market_details_dir=market_details_dir,
        month_key=month_key,
        run_started_at=run_started_at,
        set_alias_cache=set_alias_cache,
        workers=workers,
    ) if do_minifigs else market.FileUpdateStats(total_rows=0)

    parts_stats = run_tasks(
        parts_rows,
        label="Parts",
        item_type="PART",
        cfg=cfg,
        tasks=part_tasks,
        market_details_dir=None,
        month_key=month_key,
        run_started_at=run_started_at,
        set_alias_cache=set_alias_cache,
        workers=workers,
    ) if do_parts else market.FileUpdateStats(total_rows=0)

    if not args.skip_cross_enrichment and (do_sets or do_minifigs):
        set_cross, minifig_cross = market.apply_cross_catalog_enrichment(sets_rows, minifigs_rows)
        sets_stats.cross_rows_changed = set_cross
        minifigs_stats.cross_rows_changed = minifig_cross

    sets_written = market.maybe_write_json(sets_path, sets_rows) if do_sets else False
    minifigs_written = market.maybe_write_json(minifigs_path, minifigs_rows) if do_minifigs else False
    parts_written = market.maybe_write_json(parts_path, parts_rows) if do_parts else False

    if cfg.verbose:
        print(
            f"[Write] sets_written={sets_written} minifigs_written={minifigs_written} parts_written={parts_written}",
            flush=True,
        )

    market.print_summary("Sets", sets_stats)
    market.print_summary("Minifigs", minifigs_stats)
    market.print_summary("Parts", parts_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
