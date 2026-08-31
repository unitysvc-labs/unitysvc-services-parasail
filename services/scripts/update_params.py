#!/usr/bin/env python3
"""
update_services.py - Extract model data from Parasail API and generate service files

This script:
1. Retrieves all models from Parasail /v1/models endpoint
2. Reads real per-model input/output rates from Parasail's published model catalog
3. Renders listing.json and offering.json from Jinja2 templates
4. Flags deprecated service directories

Usage:
  python update_services.py                                # Process all models
  python update_services.py --models model1 model2         # Process specific models
  python update_services.py custom_dir --models model1     # Custom output + specific models

The default output directory is `data/parasail/services` (resolved
relative to the script's location, not the current working
directory) so the script writes to the right place no matter where
it is invoked from.
"""

import os
import sys
import json
import requests
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import re
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from unitysvc_sellers.model_data import ModelDataFetcher, ModelDataLookup
from unitysvc_sellers.params_render import write_params_from_iterator


PROVIDER_NAME = "parasail"

PROVIDER_DISPLAY_NAME = "Parasail"
ENV_API_KEY_NAME = "PARASAIL_API_KEY"


def _sanitize_header_value(value: str) -> str:
    """Strip smart/curly quotes and any non-latin-1 chars that break HTTP headers."""
    for bad, good in [
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u2018", "'"),
        ("\u2019", "'"),
    ]:
        value = value.replace(bad, good)
    value = value.encode("latin-1", errors="ignore").decode("latin-1").strip()
    value = value.strip('"').strip("'")
    return value


def _now_iso() -> str:
    """Millisecond ISO-8601 UTC timestamp, e.g. 2025-08-17T10:55:04.976Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

# Parasail's published model catalog: one markdown table carrying, per model,
# the context window, max output, quantization and the REAL serverless rates
# (input / output / cached input, $ per 1M tokens). Parasail regenerates it
# daily from their live models endpoint, and `.md` is served as text/markdown,
# so it parses deterministically.
#
# This replaces a `derive_price()` that guessed a parameter count out of the
# model-NAME string and looked it up in Parasail's *batch* size-tier table.
# That was wrong three ways: it read the wrong table, it collapsed separate
# input/output rates into one blended number, and — because 20 of 47 model
# names carry no "<N>b" token — it silently defaulted them to 30B/$0.50.
# Kimi K3 (~1T params, $3/$15) was billed as a 30B model at $0.50 flat.
#
# /v1/models is NOT a pricing source: it returns the OpenAI-standard
# id/object/created/owned_by and nothing else.
PARASAIL_MODELS_DOC_URL = "https://docs.parasail.io/parasail-docs/products/overview/models.md"

# What the platform adds on top of the upstream rate for the MANAGED channel,
# where UnitySVC's own key pays Parasail and the customer pays UnitySVC. The
# byok channel is unaffected: the customer's key pays Parasail directly, so
# there is nothing to mark up and nothing to pay out.
#
# Matches the crofai catalog, which is the reference implementation for this
# shape. Kept as a marked-up `list_price` + raw `payout_price` pair rather than
# a `revenue_share` payout, so the payout does not silently follow a change to
# the markup, an override, or a promotion (unitysvc/unitysvc#1892).
PLATFORM_MARKUP = Decimal("1.15")

# Rounding for the marked-up rate, matching crofai. 3dp measured across every
# rate this catalog carries ($0.03 - $15) lands the effective markup in
# 15.0% - 16.7%; the top of that range is the $0.03 floor, where a third
# decimal is the finest granularity available and $0.03 -> $0.035 overshoots
# by half a cent. 2dp would be far worse there ($0.03 -> $0.03, no markup at
# all, or $0.04 at a third). `_fmt_price` drops trailing zeros afterwards, so
# a rate that needs no third decimal does not show one.
PRICE_PLACES = Decimal("0.001")


def _fmt_price(value: Decimal) -> str:
    """Render a rate without trailing zeros ($0.250 -> "0.25", $1.000 -> "1")."""
    return str(value.normalize().quantize(Decimal(1)) if value == value.to_integral_value() else value.normalize())


def _parse_money(cell: str) -> Optional[Decimal]:
    cell = cell.replace("$", "").replace(",", "").strip()
    if cell in ("", "—", "-", "–", "N/A"):
        return None
    try:
        return Decimal(cell)
    except Exception:
        return None


def _parse_int(cell: str) -> Optional[int]:
    cell = cell.replace(",", "").strip()
    if not cell or not cell.replace(".", "").isdigit():
        return None
    try:
        return int(float(cell))
    except Exception:
        return None


def fetch_upstream_catalog(url: str = PARASAIL_MODELS_DOC_URL) -> Dict[str, Dict]:
    """Parse Parasail's published model table into {model_id: {...}}.

    Columns: Model | Model ID | Context window | Max output | Quantization
             | Input ($/1M) | Output ($/1M) | Cached input ($/1M)

    Rows without BOTH an input and an output rate are dropped rather than
    half-filled: a model we cannot price is a model we must not publish.
    """
    print(f"🔍 Fetching upstream pricing catalog: {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    catalog: Dict[str, Dict] = {}
    for line in response.text.splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 8:
            continue
        model_id = cells[1].strip().strip("`")
        if not model_id.startswith(f"{PROVIDER_NAME}-"):
            continue  # skips the header row and any stray table
        up_in, up_out = _parse_money(cells[5]), _parse_money(cells[6])
        if up_in is None or up_out is None:
            print(f"  ⚠️  {model_id}: no input/output rate published, skipping")
            continue
        catalog[model_id] = {
            "input": up_in,
            "output": up_out,
            "cached_input": _parse_money(cells[7]),
            "context_length": _parse_int(cells[2]),
            "max_output": _parse_int(cells[3]),
            "quantization": cells[4] or None,
        }

    if not catalog:
        raise RuntimeError(
            f"Parsed 0 priced models from {url}. The table layout probably "
            f"changed — refusing to run rather than republish stale prices."
        )
    print(f"✅ Parsed {len(catalog)} priced models from the upstream catalog")
    return catalog


def build_price(model_id: str, catalog: Dict[str, Dict]) -> Optional[Dict]:
    """Upstream and marked-up rates for one model, or None if unpriced.

    Returns ``{"upstream": {...}, "managed": {...}}`` — two distinct
    ``one_million_tokens`` prices that must never be aliased to one object:

    * ``upstream`` is what Parasail charges. It becomes ``payout_price``:
      what the platform owes the seller, which must not move when we change
      what we charge.
    * ``managed`` is upstream x ``PLATFORM_MARKUP``. It becomes ``list_price``:
      what the customer pays UnitySVC.

    Returning None is deliberate and load-bearing — see the caller, which drops
    the model from the iterator rather than falling back to a default rate. That
    drop is also what retires it: `write_params_from_iterator(deprecate_missing=
    True)` marks every committed service the iterator did not yield as
    `status="deprecated"`. No published rate means Parasail stopped selling it.
    """
    entry = catalog.get(model_id)
    if entry is None:
        return None

    up_in, up_out = entry["input"], entry["output"]
    up_cached = entry.get("cached_input")

    def marked(value: Decimal) -> Decimal:
        return (value * PLATFORM_MARKUP).quantize(PRICE_PLACES, rounding=ROUND_HALF_UP)

    mk_in, mk_out = marked(up_in), marked(up_out)

    upstream: Dict[str, str] = {
        "input": _fmt_price(up_in),
        "output": _fmt_price(up_out),
        "type": "one_million_tokens",
        "reference": PARASAIL_MODELS_DOC_URL,
    }
    managed: Dict[str, str] = {
        "description": f"${_fmt_price(mk_in)}/${_fmt_price(mk_out)} / 1M input/output tokens",
        "input": _fmt_price(mk_in),
        "output": _fmt_price(mk_out),
        "type": "one_million_tokens",
        "reference": PARASAIL_MODELS_DOC_URL,
    }
    if up_cached is not None:
        upstream["cached_input"] = _fmt_price(up_cached)
        managed["cached_input"] = _fmt_price(marked(up_cached))
    return {"upstream": upstream, "managed": managed}


def derive_service_type(model_id: str) -> str:
    mid = model_id.lower()
    if any(k in mid for k in ["embed", "embedding"]):
        return "embedding"
    if any(k in mid for k in ["flux", "stable-diffusion", "sdxl"]):
        return "image_generation"
    # vision_language_model and prerecorded_transcription are not valid server-side;
    # fall back to llm for all text-based models including vision and TTS
    return "llm"


#: Embedding model families whose names contain no "embed" substring, so
#: ``derive_service_type`` files them as ``llm``.  BAAI/bge-m3 is the live
#: case: it serves /v1/embeddings only, and a chat call against it 400s.
_EMBEDDING_FAMILIES = ("bge-", "bge_", "gte-", "e5-", "jina-embeddings", "nomic-embed")


def derive_capability(model_id: str, service_type: str) -> str:
    """The platform capability this offering provides.

    From the platform vocabulary (unitysvc ``docs/capabilities.yml``): what
    the caller GETS from a call, which is a different axis from
    ``service_type``.

    Cannot simply map ``service_type``, because ``derive_service_type``
    deliberately collapses TTS and transcription models into ``llm`` — those
    values are not valid server-side. So the modality keywords are re-tested
    here, where the collapse does not apply. Vision models stay ``chat``: an
    image in the request is an attribute of a chat call.
    """
    mid = model_id.lower()
    if "rerank" in mid:
        return "rerank"
    if any(k in mid for k in ["tts", "text-to-speech"]):
        return "text-to-speech"
    if any(k in mid for k in ["whisper", "transcribe"]):
        return "speech-to-text"
    if service_type == "embedding" or any(k in mid for k in _EMBEDDING_FAMILIES):
        return "embed"
    if service_type == "image_generation":
        return "image-generate"
    return "chat"


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class ParasailModelExtractor:
    def __init__(self, api_key: str, api_base_url: str, templates_dir: Path):
        api_key = _sanitize_header_value(api_key)
        self.api_key = api_key
        self.api_base_url = (api_base_url or "https://api.parasail.io/v1").strip()
        self.templates_dir = templates_dir
        self.fetcher = ModelDataFetcher()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            }
        )
        self.summary = {
            "total_models": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "deprecated_models": 0,
            "new_models": 0,
            "extraction_date": datetime.now().isoformat(),
            "processing_limit": None,
        }

        # Set up Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )
        self.jinja_env.filters["tojson"] = lambda v: json.dumps(v)

    # ------------------------------------------------------------------
    # Model listing
    # ------------------------------------------------------------------

    def get_all_models(self) -> List[Dict]:
        """Retrieve all models from Parasail /v1/models endpoint."""
        print("🔍 Fetching all models from Parasail API...")
        url = f"{self.api_base_url}/models"
        try:
            print("📄 Fetching models...")
            response = self.session.get(url, params={"limit": 1000})
            response.raise_for_status()
            data = response.json()
            all_models = data.get("data", data) if isinstance(data, dict) else data
            if not isinstance(all_models, list):
                print(f"❌ Unexpected /models response shape: {type(all_models)}")
                return []
            self.summary["total_models"] = len(all_models)
            print(f"✅ Found {len(all_models)} models total")
            all_models.sort(key=lambda x: x.get("id", ""))
            return all_models
        except requests.RequestException as e:
            print(f"❌ Error fetching models: {e}")
            return []

    def get_model_details(self, model_id: str) -> Optional[Dict]:
        """Attempt per-model detail fetch (Parasail may not support this)."""
        endpoint = f"{self.api_base_url}/models/{model_id}"
        try:
            response = self.session.get(endpoint, timeout=10)
            if response.status_code == 200:
                print("  ✅ Retrieved API details")
                return response.json()
            elif response.status_code == 404:
                return None
            else:
                response.raise_for_status()
        except requests.RequestException:
            return None
        return None

    # ------------------------------------------------------------------
    # Template rendering
    # ------------------------------------------------------------------

    def _render_template(self, template_name: str, context: Dict) -> str:
        template = self.jinja_env.get_template(template_name)
        return template.render(**context)

    def build_listing_context(
        self, model_id: str, price: Dict, time_created: Optional[str] = None
    ) -> Dict:
        return {
            "provider_name": PROVIDER_NAME,
            "offering_name": model_id,
            "env_api_key_name": ENV_API_KEY_NAME,
            "time_created": time_created or _now_iso(),
            "status": "ready",
            # The MARKED-UP rate: what the customer pays UnitySVC on the
            # managed channel. `description` is set in build_price() because
            # the listing template falls back to it whenever `price` is absent,
            # which is always now that we carry input/output separately.
            "list_price": price["managed"],
        }

    def build_offering_context(
        self,
        model_id: str,
        model_data: Dict,
        price: Dict,
        time_created: Optional[str] = None,
    ) -> Dict:
        service_type = derive_service_type(model_id)
        display_name = (
            model_data.get("display_name")
            or model_data.get("name")
            or model_id.split("/")[-1]
        )
        description = model_data.get("description", "")

        details: Dict[str, Any] = {"model_name": model_id}
        for field in [
            "context_length",
            "context_window",
            "max_tokens",
            "parameter_count",
            "supports_tools",
            "supports_vision",
        ]:
            if field in model_data:
                details[field] = model_data[field]

        # Canonical metadata fallback (PR unitysvc/unitysvc#863 requires
        # both context_length and parameter_count keys to be present on
        # every LLM offering — null is the sentinel for "unknown").
        # Uses ModelDataLookup.get_canonical_metadata which chains
        # OpenRouter → LiteLLM → HuggingFace.
        if service_type == "llm":
            canonical = ModelDataLookup.get_canonical_metadata(
                model_id, fetcher=self.fetcher
            )
            sources: Dict[str, Any] = {}
            if details.get("context_length") is None:
                details["context_length"] = canonical["context_length"]
                if canonical["sources"].get("context_length"):
                    sources["context_length"] = canonical["sources"]["context_length"]
            # Parasail's API never reports parameter_count — always pull
            # from canonical so the validator-required key is populated.
            details["parameter_count"] = canonical["parameter_count"]
            if canonical["sources"].get("parameter_count"):
                sources["parameter_count"] = canonical["sources"]["parameter_count"]
            if sources:
                details["metadata_sources"] = sources
            # Ensure both required keys are present even when canonical
            # lookup returned nothing — null marks "unknown".
            details.setdefault("context_length", None)
            details.setdefault("parameter_count", None)

        # Gate for the function-calling code example. Parasail's model API
        # reports supports_tools=null for every model, so there is no positive
        # signal to key on; default to attaching the example and correct with
        # the observed-failure denylist (deployments without a tool-call
        # parser 400 — or crash with a 500 — on any `tools` request).
        # Parasail's model API reports supports_tools=null for everything;
        # corrections live in the per-model <name>.override.json companions
        # (merged at render time), so this script never changes for one.
        supports_tools = details.get("supports_tools") is not False

        return {
            "provider_name": PROVIDER_NAME,
            "provider_display_name": PROVIDER_DISPLAY_NAME,
            "env_api_key_name": ENV_API_KEY_NAME,
            "time_created": time_created or _now_iso(),
            "offering_name": model_id,
            "display_name": display_name,
            "description": description,
            "service_type": service_type,
            "capability": derive_capability(model_id, service_type),
            "status": "ready",
            "api_base_url": "https://api.parasail.io",
            "supports_tools": supports_tools,
            "details": details,
            # The RAW upstream rate: what the platform owes the seller. It must
            # not track list_price — that is the whole point of storing both.
            "payout_price": price["upstream"],
        }

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _write_file(self, content: str, output_file: Path):
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  ✅ Written: {output_file}")
        except Exception as e:
            print(f"  ❌ Error writing {output_file}: {e}")

    @staticmethod
    def _existing_time_created(path: Path) -> Optional[str]:
        """Return the ``time_created`` already recorded in a spec file, if any.

        Regenerating a service reuses its original creation timestamp so an
        unchanged service produces no diff; only brand-new services get a fresh
        timestamp.
        """
        if path.is_file():
            try:
                return json.loads(path.read_text()).get("time_created")
            except Exception:
                return None
        return None

    @staticmethod
    def _param_time_created(path: Path) -> Optional[str]:
        """time_created recorded inside a committed param file's ``parameters``
        (the post-migration home of the field), so re-runs stay idempotent."""
        if path.is_file():
            try:
                data = json.loads(path.read_text())
                return (data.get("parameters") or {}).get("time_created")
            except Exception:
                return None
        return None

    def write_listing(self, model_id: str, price: str, output_dir: Path):
        created = self._existing_time_created(output_dir / "listing.json")
        context = self.build_listing_context(model_id, price, time_created=created)
        content = self._render_template("listing.json.j2", context)
        self._write_file(content, output_dir / "listing.json")

    def write_offering(
        self, model_id: str, model_data: Dict, price: str, output_dir: Path
    ):
        created = self._existing_time_created(output_dir / "offering.json")
        context = self.build_offering_context(
            model_id, model_data, price, time_created=created
        )
        content = self._render_template("offering.json.j2", context)
        self._write_file(content, output_dir / "offering.json")

    def write_provider(self, output_dir: Path):
        """Copy the static templates/provider.json into the service folder so
        each folder is self-contained. provider.json is a pure provider
        definition — the populator config lives in templates/config.json."""
        prov = json.loads((self.templates_dir / "provider.json").read_text())
        content = json.dumps(prov, sort_keys=True, indent=2) + "\n"
        self._write_file(content, output_dir / "provider.json")

    def write_summary(self):
        try:
            print(f"   Total models: {self.summary['total_models']}")
            print(
                f"   Successful extractions: {self.summary['successful_extractions']}"
            )
            print(f"   New models: {self.summary.get('new_models', 0)}")
            print(f"   Deprecated models: {self.summary.get('deprecated_models', 0)}")
            if self.summary["processing_limit"]:
                print(f"   Processing limit: {self.summary['processing_limit']}")
        except Exception as e:
            print(f"❌ Error writing summary: {e}")

    # ------------------------------------------------------------------
    # Model list hygiene
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup_case_variant_ids(models: List[Dict]) -> List[Dict]:
        """Drop case-variant duplicate model IDs returned by the API.

        Parasail's catalog sometimes lists the same model twice under IDs that
        differ only by letter case (e.g. ``MiniMaxAI/MiniMax-M3`` and
        ``MiniMaxAI/Minimax-M3``). They render to service paths that collide on a
        case-insensitive filesystem (default macOS, much CI), so keep exactly one
        per case-folded ID — the most properly-cased variant (most uppercase
        letters; ties broken lexicographically) — and skip the rest. Original API
        order is preserved for the survivors.
        """
        by_fold: Dict[str, List[Dict]] = {}
        for m in models:
            mid = m.get("id", "")
            if mid:
                by_fold.setdefault(mid.casefold(), []).append(m)
        keep_ids: set = set()
        for group in by_fold.values():
            if len(group) > 1:
                group.sort(key=lambda m: (-sum(c.isupper() for c in m["id"]), m["id"]))
                keep, *drop = group
                print(
                    f"  ⚠️  case-variant duplicate(s) of '{keep['id']}': dropping "
                    + ", ".join(repr(m["id"]) for m in drop)
                )
            else:
                keep = group[0]
            keep_ids.add(id(keep))
        return [m for m in models if id(m) in keep_ids]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_all_models(
        self,
        output_dir: str = "services",
        specific_models: Optional[List[str]] = None,
        limit: Optional[int] = None,
        dry_run: bool = False,
    ):
        print("🚀 Starting Parasail model extraction...\n")
        self.summary["processing_limit"] = limit

        if dry_run:
            print(
                "🔍 Dry-run mode enabled - will show what would be done without writing files"
            )

        if specific_models:
            print(f"🎯 Processing specific models: {', '.join(specific_models)}")
            models = [{"id": model_id} for model_id in specific_models]
            self.summary["total_models"] = len(models)
        else:
            models = self.get_all_models()
            if not models:
                # Exit non-zero, not quietly. An empty enumeration means the
                # upstream call failed (bad credential, endpoint, outage) — and
                # a silent success here writes nothing, produces no PR, and
                # looks exactly like "no changes today".
                print("❌ No models retrieved — treating as a failed run, not an empty catalog.")
                sys.exit(1)
            models = self._dedup_case_variant_ids(models)
            self.summary["total_models"] = len(models)

        processed_count = 0
        param_contexts: List[Dict] = []

        # Fetched once per run, not per model. A failure here raises rather
        # than degrading to a default: republishing 47 services at a guessed
        # rate is strictly worse than not republishing them at all.
        upstream_catalog = fetch_upstream_catalog()
        unpriced: List[str] = []

        for i, model_data in enumerate(models, start=1):
            model_id = model_data.get("id", "")
            if not model_id:
                continue

            print(f"\n[{i}/{len(models)}] Processing: {model_id}")

            if limit and processed_count >= limit:
                print(f"🔢 Reached processing limit of {limit} models, stopping...")
                break

            base_path = Path(output_dir)
            # Folder path = the listing name = "<provider>/<model_id>" (#1263).
            # The full model id (incl. any org segment) becomes the nested path
            # under specs/<provider>/, so the folder matches listing.name.
            data_dir = base_path / PROVIDER_NAME / model_id

            processed_count += 1

            try:
                # Get API details
                details = self.get_model_details(model_id)
                if details:
                    model_data = model_data | details
                time.sleep(0.1)

                # Real upstream rates, looked up by model id. A model absent
                # from Parasail's published catalog is dropped from the
                # iterator, never priced from a fallback: the old code's
                # `else 30` default is exactly what put Kimi K3 on the shelf
                # at $0.50. Dropping it also retires it, via deprecate_missing.
                price = build_price(model_id, upstream_catalog)
                if price is None:
                    print(f"  ⏭️  No published upstream rate for {model_id} — will deprecate")
                    unpriced.append(model_id)
                    self.summary["skipped_unpriced"] = len(unpriced)
                    continue
                print(
                    f"  💰 upstream ${price['upstream']['input']}/${price['upstream']['output']}"
                    f"  →  managed (×{PLATFORM_MARKUP}) "
                    f"${price['managed']['input']}/${price['managed']['output']} per 1M in/out"
                )

                if dry_run:
                    print(
                        f"  📝 [DRY-RUN] Would write offering.json + listing.json to {data_dir}"
                    )
                    self.summary["successful_extractions"] += 1
                    continue

                # Preserve time_created (committed param file first, then the
                # legacy expanded offering.json) so unchanged services don't
                # churn; the merged offering + listing render context becomes
                # one param file.
                created = self._param_time_created(
                    base_path / PROVIDER_NAME / f"{model_id}.json"
                ) or self._existing_time_created(data_dir / "offering.json")
                offering = self.build_offering_context(model_id, model_data, price, time_created=created)
                listing = self.build_listing_context(model_id, price, time_created=created)
                param_contexts.append({**offering, **listing, "service_name": f"{PROVIDER_NAME}/{model_id}"})

                self.summary["successful_extractions"] += 1
                print(f"  ✅ Successfully processed {model_id}")

            except Exception as e:
                print(f"  ❌ Error processing {model_id}: {e}")
                self.summary["failed_extractions"] += 1

        if not dry_run:
            # Deprecating on absence is only sound when this run saw the whole
            # catalog. A truncated run (--limit) yields a handful of models, and
            # a run that dropped models to per-model errors cannot tell "retired
            # upstream" from "we failed to fetch it" — either would retire live
            # services. Skip deprecation rather than guess.
            complete_run = limit is None and self.summary["failed_extractions"] == 0
            if unpriced:
                # Dropped from the iterator on purpose, so `deprecate_missing`
                # retires them. A model Parasail no longer publishes a rate for
                # is a model Parasail no longer sells; keeping it listed at its
                # last known price is the worse failure. write_params_from_iterator
                # still raises UpstreamEnumerationError if the iterator matched
                # NOTHING, which is the case this must not be confused with.
                print(
                    f"\n⚠️  {len(unpriced)} service(s) have no published upstream "
                    f"rate and will be DEPRECATED: {', '.join(sorted(unpriced))}"
                )
            if not complete_run:
                print(
                    "⚠️  Incomplete run "
                    f"(limit={limit}, failures={self.summary['failed_extractions']})"
                    " — skipping deprecation of absent services."
                )
            stats = write_params_from_iterator(
                iter(param_contexts), output_dir, deprecate_missing=complete_run
            )
            # Report what the writer actually did. The old counter tracked this
            # script's own deprecation pass, which scanned for listing.json —
            # a shape this repo has not had since the params migration — so it
            # printed 0 no matter what.
            self.summary["deprecated_models"] = stats["deprecated"]
            self.summary["new_models"] = stats["new"]

        self.write_summary()
        print(f"\n🎉 Extraction complete! Check {output_dir}/ for results.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Resolve the default output directory relative to this script so the
    # behaviour matches the other 12 unitysvc-services-* repos and is
    # independent of the current working directory.
    DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent.parent / "specs")

    parser = argparse.ArgumentParser(
        description="Extract model data from Parasail API and generate service files"
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for service files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model IDs to process",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of models to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files",
    )

    args = parser.parse_args()

    api_key = os.environ.get("PARASAIL_API_KEY")
    api_base_url = os.environ.get("PARASAIL_API_BASE_URL")

    if api_key:
        api_key = _sanitize_header_value(api_key)

    if not api_key:
        print(
            "❌ Error: No API key provided. Set the PARASAIL_API_KEY environment variable."
        )
        sys.exit(1)

    # Templates live at ../templates relative to this script
    script_dir = Path(__file__).parent
    templates_dir = script_dir.parent / "templates"

    if not templates_dir.exists():
        print(f"❌ Templates directory not found: {templates_dir}")
        sys.exit(1)

    extractor = ParasailModelExtractor(api_key, api_base_url, templates_dir)
    extractor.process_all_models(
        args.output_dir,
        specific_models=args.models,
        limit=args.limit,
        dry_run=args.dry_run,
    )
