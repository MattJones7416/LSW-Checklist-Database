#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


PROVIDERS = ("bricklink", "brickowl", "amazon", "lego", "very", "vinted", "johnlewis")
OUTPUT_FILENAME_BY_REGION = {
    "UK": "uk.json",
    "US": "us.json",
    "EU": "eu.json",
    "ALL": "all.json",
}


def collapse_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_region(value: str) -> str:
    raw = collapse_ws(value).upper()
    if raw in {"GB", "UK"}:
        return "UK"
    if raw in {"US", "USA"}:
        return "US"
    if raw in {"EU", "DE", "FR", "ES", "IT"}:
        return "EU"
    return raw or "UK"


def normalize_model_key(value: str) -> str:
    return collapse_ws(value).casefold()


def safe_filename_token(value: str) -> str:
    token = normalize_model_key(value)
    token = re.sub(r"[^a-z0-9._-]+", "_", token)
    return token.strip("._-") or "model"


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def write_json(path: Path, data: Any, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    else:
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
    path.write_text(payload, encoding="utf-8")


def resolve_repo_relative_path(raw_path: str, repo_root: Path) -> str:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return str(path)
    candidate = repo_root / raw_path
    return str(candidate if candidate.exists() else path)


def merge_provider_region_output(existing_path: Path, replacement_deals: List[Dict[str, Any]], *, target_number: str) -> int:
    normalized_target = normalize_model_key(target_number)
    existing = load_json_array(existing_path)
    preserved = [
        deal for deal in existing
        if normalize_model_key(deal.get("number", "")) != normalized_target
    ]
    merged = preserved + replacement_deals
    merged.sort(
        key=lambda deal: (
            normalize_model_key(deal.get("number", "")),
            collapse_ws(deal.get("source")).casefold(),
            float(deal.get("priceValue") or 0.0) if str(deal.get("priceValue") or "").strip() else float("inf"),
            collapse_ws(deal.get("id")).casefold(),
        )
    )
    write_json(existing_path, merged, pretty=False)
    return len([deal for deal in merged if normalize_model_key(deal.get("number", "")) == normalized_target])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh marketplace provider deals for a single model.")
    parser.add_argument("--number", required=True)
    parser.add_argument("--item-type", required=True, choices=["set", "minifig", "part"])
    parser.add_argument("--region", default="UK")
    parser.add_argument("--request-key", default="")
    parser.add_argument("--sets-json", default="dist/Lego Star Wars Database.json")
    parser.add_argument("--minifigs-json", default="dist/Lego-Star-Wars-Minifigure-Database.json")
    parser.add_argument("--parts-json", default="dist/parts/parts-catalog.json")
    parser.add_argument("--provider-root", default="dist/provider-deals")
    parser.add_argument("--merged-output-dir", default="dist/deals")
    parser.add_argument("--merged-fallback-output", default="dist/marketplace-deals.json")
    parser.add_argument("--status-output-dir", default="dist/marketplace-refresh-status")
    parser.add_argument("--providers", default=",".join(PROVIDERS))
    parser.add_argument("--max-results-per-item", type=int, default=3)
    parser.add_argument("--max-product-pages-per-item", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    sets_json = resolve_repo_relative_path(args.sets_json, repo_root)
    minifigs_json = resolve_repo_relative_path(args.minifigs_json, repo_root)
    parts_json = resolve_repo_relative_path(args.parts_json, repo_root)
    requested_number = collapse_ws(args.number)
    requested_item_type = collapse_ws(args.item_type).lower()
    region = normalize_region(args.region)
    started_at = datetime.now(timezone.utc)
    providers = [collapse_ws(value).lower() for value in re.split(r"[,\s]+", args.providers) if collapse_ws(value)]
    providers = [provider for provider in providers if provider in PROVIDERS]
    if not providers:
        raise SystemExit("No supported providers configured")

    provider_root = Path(args.provider_root)
    status_rows: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="single-marketplace-refresh-") as temp_dir_raw:
        temp_root = Path(temp_dir_raw)
        for provider in providers:
            temp_provider_dir = temp_root / provider
            temp_provider_dir.mkdir(parents=True, exist_ok=True)
            state_path = temp_provider_dir / "state.json"
            script_cmd = [
                sys.executable,
                str(repo_root / "scripts" / "update_marketplace_provider_deals.py"),
                "--provider", provider,
                "--sets-json", sets_json,
                "--minifigs-json", minifigs_json,
                "--parts-json", parts_json,
                "--output-dir", str(temp_provider_dir),
                "--state-path", str(state_path),
                "--regions", region,
                "--sets-per-region", "1" if requested_item_type == "set" else "0",
                "--minifigs-per-region", "1" if requested_item_type == "minifig" else "0",
                "--parts-per-region", "1" if requested_item_type == "part" else "0",
                "--max-results-per-item", str(args.max_results_per_item),
                "--max-product-pages-per-item", str(args.max_product_pages_per_item),
                "--only-number", requested_number,
                "--only-item-type", requested_item_type,
            ]
            if args.verbose:
                script_cmd.append("--verbose")

            completed = subprocess.run(
                script_cmd,
                text=True,
                capture_output=True,
                check=False,
            )

            replacement_path = temp_provider_dir / OUTPUT_FILENAME_BY_REGION[region]
            replacement_deals = load_json_array(replacement_path)
            provider_output_dir = provider_root / provider
            provider_output_dir.mkdir(parents=True, exist_ok=True)
            provider_region_path = provider_output_dir / OUTPUT_FILENAME_BY_REGION[region]
            target_deals_after_merge = merge_provider_region_output(
                provider_region_path,
                replacement_deals,
                target_number=requested_number,
            )

            metadata_path = temp_provider_dir / "metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            else:
                metadata = {}
            metadata["lastSingleRefreshAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            metadata["lastSingleRefreshNumber"] = requested_number
            metadata["lastSingleRefreshItemType"] = requested_item_type
            metadata["lastSingleRefreshRegion"] = region
            write_json(provider_output_dir / "metadata.json", metadata, pretty=True)

            stdout_tail = "\n".join(line for line in completed.stdout.strip().splitlines()[-8:] if line)
            stderr_tail = "\n".join(line for line in completed.stderr.strip().splitlines()[-8:] if line)
            if completed.returncode == 0:
                status = "success"
                error_message = ""
            elif replacement_deals:
                status = "partial"
                error_message = stderr_tail or stdout_tail
            else:
                status = "failed"
                error_message = stderr_tail or stdout_tail or f"{provider} returned exit code {completed.returncode}"

            status_rows.append(
                {
                    "provider": provider,
                    "status": status,
                    "region": region,
                    "dealsFound": len(replacement_deals),
                    "dealsForModelAfterMerge": target_deals_after_merge,
                    "stdoutTail": stdout_tail,
                    "error": error_message,
                    "exitCode": completed.returncode,
                }
            )

    merge_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "merge_marketplace_deals.py"),
        "--provider-root", str(provider_root),
        "--output-dir", args.merged_output_dir,
        "--fallback-output", args.merged_fallback_output,
    ]
    merge_completed = subprocess.run(merge_cmd, text=True, capture_output=True, check=False)

    finished_at = datetime.now(timezone.utc)
    success_count = sum(1 for row in status_rows if row["status"] == "success")
    partial_count = sum(1 for row in status_rows if row["status"] == "partial")
    failed_count = sum(1 for row in status_rows if row["status"] == "failed")

    if success_count > 0 and failed_count == 0 and merge_completed.returncode == 0:
        overall_status = "success"
    elif success_count > 0 or partial_count > 0:
        overall_status = "partial"
    else:
        overall_status = "failed"

    summary = {
        "number": requested_number,
        "itemType": requested_item_type,
        "region": region,
        "requestKey": collapse_ws(args.request_key),
        "startedAt": started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finishedAt": finished_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overallStatus": overall_status,
        "providers": status_rows,
        "merge": {
            "status": "success" if merge_completed.returncode == 0 else "failed",
            "exitCode": merge_completed.returncode,
            "stdoutTail": "\n".join(line for line in merge_completed.stdout.strip().splitlines()[-8:] if line),
            "error": "\n".join(line for line in merge_completed.stderr.strip().splitlines()[-8:] if line),
        },
    }

    status_filename = f"{safe_filename_token(requested_number)}--{region.lower()}.json"
    write_json(Path(args.status_output_dir) / status_filename, summary, pretty=True)

    if overall_status == "failed":
        return 1
    if merge_completed.returncode != 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
