#!/usr/bin/env python3
"""
update_services.py - Extract model data from Parasail API and generate service files

This script:
1. Retrieves all models from Parasail /v1/models endpoint
2. Derives pricing from Parasail's parameter-size pricing table
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
from datetime import datetime, timezone

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from unitysvc_sellers.model_data import ModelDataFetcher, ModelDataLookup


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

# Parasail prices serverless models by parameter size.
# Table from https://docs.parasail.io/parasail-docs/billing/pricing
PRICING_TIERS = [
    (4, "0.05"),
    (8, "0.08"),
    (16, "0.11"),
    (21, "0.45"),
    (41, "0.50"),
    (80, "0.70"),
    (404, "0.80"),
    (float("inf"), "1.75"),
]


def derive_price(model_id: str) -> str:
    """Return price-per-1M-tokens string for a model based on its parameter count."""
    model_lower = model_id.lower()

    # MoE pattern: NxMb (e.g. 8x7b = 56b total)
    moe_match = re.search(r"(\d+)x(\d+\.?\d*)b", model_lower)
    if moe_match:
        params_b = int(moe_match.group(1)) * float(moe_match.group(2))
    else:
        # Largest number followed by b (handles 70b, 3.3-70b, 235b-a22b → 235)
        size_matches = re.findall(r"(\d+\.?\d*)b", model_lower)
        params_b = max(float(x) for x in size_matches) if size_matches else 30

    for max_b, price in PRICING_TIERS:
        if params_b <= max_b:
            return price
    return PRICING_TIERS[-1][1]


def derive_service_type(model_id: str) -> str:
    mid = model_id.lower()
    if any(k in mid for k in ["embed", "embedding"]):
        return "embedding"
    if any(k in mid for k in ["flux", "stable-diffusion", "sdxl"]):
        return "image_generation"
    # vision_language_model and prerecorded_transcription are not valid server-side;
    # fall back to llm for all text-based models including vision and TTS
    return "llm"


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
            "deprecated_models": [],
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
        self, model_id: str, price: str, time_created: Optional[str] = None
    ) -> Dict:
        return {
            "provider_name": PROVIDER_NAME,
            "offering_name": model_id,
            "env_api_key_name": ENV_API_KEY_NAME,
            "time_created": time_created or _now_iso(),
            "status": "ready",
            "list_price": {
                "description": "Pricing Per 1M Tokens",
                "price": price,
                "type": "one_million_tokens",
                "reference": "https://docs.parasail.io/parasail-docs/billing/pricing",
            },
        }

    def build_offering_context(
        self,
        model_id: str,
        model_data: Dict,
        price: str,
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

        return {
            "provider_name": PROVIDER_NAME,
            "provider_display_name": PROVIDER_DISPLAY_NAME,
            "env_api_key_name": ENV_API_KEY_NAME,
            "time_created": time_created or _now_iso(),
            "offering_name": model_id,
            "display_name": display_name,
            "description": description,
            "service_type": service_type,
            "status": "ready",
            "api_base_url": "https://api.parasail.io",
            "details": details,
            "payout_price": {
                "description": "Pricing Per 1M Tokens",
                "price": price,
                "type": "one_million_tokens",
                "reference": "https://docs.parasail.io/parasail-docs/billing/pricing",
            },
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
            print(f"   Deprecated models: {len(self.summary['deprecated_models'])}")
            if self.summary["processing_limit"]:
                print(f"   Processing limit: {self.summary['processing_limit']}")
        except Exception as e:
            print(f"❌ Error writing summary: {e}")

    # ------------------------------------------------------------------
    # Deprecation
    # ------------------------------------------------------------------

    def mark_deprecated_services(
        self, output_dir: str, active_models: List[str], dry_run: bool = False
    ):
        """Mark service files as deprecated for models no longer active."""
        print("🔍 Checking for deprecated services...")
        # Service folders live at specs/<provider>/<model_id>/ (model_id may be
        # nested, e.g. deepseek-ai/DeepSeek-V3.2). File TYPE is the filename now.
        base_path = Path(output_dir) / PROVIDER_NAME
        if not base_path.exists():
            return

        active = set(active_models)  # full model ids
        deprecated_count = 0

        for listing_file in base_path.rglob("listing.json"):
            svc_dir = listing_file.parent
            model_id = svc_dir.relative_to(base_path).as_posix()
            if model_id in active:
                continue

            deprecated_count += 1
            print(f"  🗑️  Processing deprecated service: {model_id}")

            for fname in ("offering.json", "listing.json"):
                json_file = svc_dir / fname
                if not json_file.exists():
                    continue
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("status") == "deprecated":
                        continue
                    data["status"] = "deprecated"
                    if dry_run:
                        print(f"    📝 [DRY-RUN] Would deprecate {json_file.name}")
                    else:
                        with open(json_file, "w", encoding="utf-8") as f:
                            json.dump(
                                data,
                                f,
                                sort_keys=True,
                                indent=2,
                                separators=(",", ": "),
                            )
                            f.write("\n")
                        print(f"    ✅ Deprecated {json_file.name}")
                except Exception as e:
                    print(f"    ❌ Error updating {json_file}: {e}")

        if deprecated_count == 0:
            print("  ✅ No deprecated services found")
        else:
            print(f"  🗑️  Processed {deprecated_count} deprecated services")

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
                print("❌ No models retrieved. Exiting.")
                return
            # Full sync: deprecate any local service no longer offered upstream.
            # (Skipped when --limit is set, since the model list is truncated.)
            if limit is None:
                active_ids = [m.get("id", "") for m in models if m.get("id")]
                self.mark_deprecated_services(output_dir, active_ids, dry_run)

        processed_count = 0

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

                # Derive pricing from parameter count
                price = derive_price(model_id)
                print(f"  💰 Price: ${price}/1M tokens")

                if dry_run:
                    print(
                        f"  📝 [DRY-RUN] Would write offering.json + listing.json to {data_dir}"
                    )
                    self.summary["successful_extractions"] += 1
                    continue

                print(f"  📝 Writing files to {data_dir}...")
                # Regenerate every file. time_created is preserved per-file
                # (see write_offering/write_listing) so unchanged services are
                # byte-identical and the daily cron produces no churn.
                self.write_offering(model_id, model_data, price, data_dir)
                # Self-contained service folder: copy the provider in too.
                self.write_provider(data_dir)
                self.write_listing(model_id, price, data_dir)

                self.summary["successful_extractions"] += 1
                print(f"  ✅ Successfully processed {model_id}")

            except Exception as e:
                print(f"  ❌ Error processing {model_id}: {e}")
                self.summary["failed_extractions"] += 1

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
