#!/usr/bin/env python3
"""One-shot migration: normalise LLM offering details to canonical fields.

For every ``offering.json`` under ``data/parasail/services/*/`` with
``service_type == "llm"``, this script:

1. Renames legacy field names to canonical snake_case:
   - ``contextLength``     → ``context_length``
   - ``context_window``    → ``context_length``
   - ``parameterCount``    → ``parameter_count``
2. Resolves any missing canonical fields via
   ``ModelDataLookup.get_canonical_metadata`` (OpenRouter → LiteLLM →
   HuggingFace fallback chain), recording provenance under
   ``details.metadata_sources``.
3. Replaces the historical ``9999`` / ``0`` sentinels with ``null``.
4. Ensures both ``context_length`` and ``parameter_count`` keys are
   present on every LLM offering (post-PR-#863 the platform validator
   requires presence; ``null`` is the canonical "unknown" marker).

Run once, commit the resulting offering.json diffs.  Re-running is a
no-op — the script is idempotent.
"""

from __future__ import annotations

import json
from pathlib import Path

from unitysvc_sellers.model_data import ModelDataFetcher, ModelDataLookup

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICES_DIR = REPO_ROOT / "data" / "parasail" / "services"

# Legacy → canonical field-name mapping.  Keys removed from the source
# dict, value stored under the canonical name (only if the canonical
# isn't already populated, so a manual override wins).
RENAMES = {
    "contextLength": "context_length",
    "context_window": "context_length",
    "parameterCount": "parameter_count",
}

# Sentinel values from the original PR #859 — replaced with null per #863.
SENTINEL_TO_NULL = {
    "context_length": {9999},
    "parameter_count": {0},
}


def _normalise_existing(details: dict) -> None:
    """In-place: apply renames + sentinel-to-null on existing data."""
    for legacy, canonical in RENAMES.items():
        if legacy in details:
            value = details.pop(legacy)
            details.setdefault(canonical, value)

    for key, sentinels in SENTINEL_TO_NULL.items():
        if details.get(key) in sentinels:
            details[key] = None


def _backfill_canonical(
    details: dict,
    model_id: str,
    fetcher: ModelDataFetcher,
) -> None:
    """In-place: fetch canonical metadata for any field that's still missing.

    Only writes when the field is absent.  Existing values (including
    explicit ``null``) are left alone — the seller's intent wins over
    the auto-fetcher.
    """
    needs_lookup = "context_length" not in details or "parameter_count" not in details
    if not needs_lookup:
        # Make sure both keys are present even if no fetch needed.
        details.setdefault("context_length", None)
        details.setdefault("parameter_count", None)
        return

    canonical = ModelDataLookup.get_canonical_metadata(model_id, fetcher=fetcher)
    sources = details.setdefault("metadata_sources", {})

    for field in ("context_length", "parameter_count"):
        if field in details:
            continue
        details[field] = canonical[field]
        if canonical["sources"].get(field):
            sources[field] = canonical["sources"][field]

    # Drop the metadata_sources key entirely if no provenance was set —
    # avoids cluttering offerings with empty dicts.
    if not sources:
        details.pop("metadata_sources", None)


def main() -> None:
    if not SERVICES_DIR.is_dir():
        raise SystemExit(f"services directory not found: {SERVICES_DIR}")

    fetcher = ModelDataFetcher()
    try:
        offerings = sorted(SERVICES_DIR.glob("*/offering.json"))
        print(f"Found {len(offerings)} offering files")

        changed = 0
        for path in offerings:
            with path.open() as f:
                offering = json.load(f)

            if offering.get("service_type") != "llm":
                continue

            details = offering.setdefault("details", {})
            before = json.dumps(details, sort_keys=True)

            _normalise_existing(details)
            model_id = offering.get("name") or path.parent.name.replace(
                "-byok", ""
            ).replace("-byop", "")
            _backfill_canonical(details, model_id, fetcher)

            after = json.dumps(details, sort_keys=True)
            if before != after:
                with path.open("w") as f:
                    json.dump(offering, f, indent=2, ensure_ascii=False)
                    f.write("\n")
                changed += 1
                print(
                    f"  updated: {path.relative_to(REPO_ROOT)}  "
                    f"context_length={details.get('context_length')}, "
                    f"parameter_count={details.get('parameter_count')}"
                )

        print(f"\nMigration complete: {changed} of {len(offerings)} files updated.")
    finally:
        fetcher.close()


if __name__ == "__main__":
    main()
