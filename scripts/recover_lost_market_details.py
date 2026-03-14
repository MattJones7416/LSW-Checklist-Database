#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
MARKET_DETAILS_ROOT = ROOT / 'dist' / 'market-details'
RESTORE_FIELDS = {
    'BrickLinkCurrentListingsNew',
    'BrickLinkCurrentListingsUsed',
    'BrickLinkTransactionsNew',
    'BrickLinkTransactionsUsed',
    'BrickLinkTransactionsNewCount',
    'BrickLinkTransactionsUsedCount',
    'BrickLinkMonthlySalesNew',
    'BrickLinkMonthlySalesUsed',
}
PRIMARY_SERIES_FIELDS = (
    'BrickLinkTransactionsNew',
    'BrickLinkTransactionsUsed',
    'BrickLinkMonthlySalesNew',
    'BrickLinkMonthlySalesUsed',
)
SOURCE_FIELDS = (
    'BrickLinkCurrentListingsNew',
    'BrickLinkCurrentListingsUsed',
    'BrickLinkTransactionsNew',
    'BrickLinkTransactionsUsed',
    'BrickLinkMonthlySalesNew',
    'BrickLinkMonthlySalesUsed',
)
SYNC_COMMIT_GREP = 'sync catalog from rebrickable'


def git(*args: str) -> str:
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True)


def load_json(path: Path) -> Dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def from_git(rev: str, relpath: str) -> Dict[str, Any] | None:
    try:
        text = git('show', f'{rev}:{relpath}')
    except subprocess.CalledProcessError:
        return None
    try:
        data = json.loads(text)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == '') or (isinstance(value, (list, dict)) and len(value) == 0)


def has_restore_payload(payload: Dict[str, Any] | None) -> bool:
    if not payload:
        return False
    return any(not is_empty(payload.get(field)) for field in SOURCE_FIELDS)


def needs_recovery(payload: Dict[str, Any] | None) -> bool:
    if not payload:
        return False
    latest = any(payload.get(field) is not None for field in ('BrickLinkLatestSaleNewPrice', 'BrickLinkLatestSaleUsedPrice'))
    missing_all_series = all(is_empty(payload.get(field)) for field in PRIMARY_SERIES_FIELDS)
    return latest and missing_all_series


def merge_payload(current: Dict[str, Any], historical: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    merged = dict(current)
    restored: List[str] = []
    for field in RESTORE_FIELDS:
        if is_empty(merged.get(field)) and not is_empty(historical.get(field)):
            merged[field] = historical.get(field)
            restored.append(field)

    if not is_empty(merged.get('BrickLinkTransactionsNew')):
        merged['BrickLinkTransactionsNewCount'] = len(merged['BrickLinkTransactionsNew'])
    if not is_empty(merged.get('BrickLinkTransactionsUsed')):
        merged['BrickLinkTransactionsUsedCount'] = len(merged['BrickLinkTransactionsUsed'])
    return merged, restored


def commit_chain() -> List[str]:
    lines = git('log', '--format=%H', f'--grep={SYNC_COMMIT_GREP}').splitlines()
    return [line.strip() for line in lines if line.strip()]


def changed_market_detail_paths(sync_commit: str) -> List[str]:
    try:
        lines = git('diff', '--name-only', f'{sync_commit}^', sync_commit, '--', 'dist/market-details').splitlines()
    except subprocess.CalledProcessError:
        return []
    return [line.strip() for line in lines if line.strip().endswith('.json')]


def fallback_history_payload(relpath: str) -> Dict[str, Any] | None:
    try:
        commits = git('log', '--format=%H', '--', relpath).splitlines()
    except subprocess.CalledProcessError:
        return None
    for commit in commits[1:200]:
        payload = from_git(commit, relpath)
        if has_restore_payload(payload):
            return payload
    return None


def main() -> int:
    current_payloads: Dict[str, Dict[str, Any]] = {}
    candidate_paths: set[str] = set()
    for path in MARKET_DETAILS_ROOT.rglob('*.json'):
        relpath = path.relative_to(ROOT).as_posix()
        payload = load_json(path)
        if payload is None:
            continue
        current_payloads[relpath] = payload
        if needs_recovery(payload):
            candidate_paths.add(relpath)

    print(f'[Recover] candidates={len(candidate_paths)}', flush=True)

    recovered: Dict[str, List[str]] = {}
    for sync_commit in commit_chain():
        parent = f'{sync_commit}^'
        changed = [path for path in changed_market_detail_paths(sync_commit) if path in candidate_paths and path not in recovered]
        if not changed:
            continue
        print(f'[Recover] checking sync commit {sync_commit[:10]} changed_candidates={len(changed)}', flush=True)
        for relpath in changed:
            historical = from_git(parent, relpath)
            if not has_restore_payload(historical):
                continue
            merged, restored = merge_payload(current_payloads[relpath], historical)
            if not restored:
                continue
            target = ROOT / relpath
            target.write_text(json.dumps(merged, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
            current_payloads[relpath] = merged
            recovered[relpath] = restored

    remaining = [path for path in sorted(candidate_paths) if path not in recovered]
    if remaining:
        print(f'[Recover] fallback-history-check remaining={len(remaining)}', flush=True)
    for relpath in remaining:
        historical = fallback_history_payload(relpath)
        if not has_restore_payload(historical):
            continue
        merged, restored = merge_payload(current_payloads[relpath], historical)
        if not restored:
            continue
        target = ROOT / relpath
        target.write_text(json.dumps(merged, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
        current_payloads[relpath] = merged
        recovered[relpath] = restored

    fully_recovered = 0
    for relpath in candidate_paths:
        payload = load_json(ROOT / relpath)
        if payload is not None and not needs_recovery(payload):
            fully_recovered += 1
    print(f'[Recover] restored_files={len(recovered)}', flush=True)
    print(f'[Recover] fully_recovered_candidates={fully_recovered}', flush=True)
    examples = sorted(recovered.items())[:20]
    for relpath, fields in examples:
        print(f'[Recover] sample {relpath} fields={",".join(fields)}', flush=True)
    unrecovered = len(candidate_paths) - len(recovered)
    print(f'[Recover] unrecovered_candidates={unrecovered}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
