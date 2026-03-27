#!/usr/bin/env python3
"""
update_services.py - Extract model data from Parasail API and generate service files

This script:
1. Retrieves all models from Parasail /v1/models endpoint
2. Derives pricing from Parasail's parameter-size pricing table
3. Renders listing.json and offering.json from Jinja2 templates
4. Flags deprecated service directories

Usage:
  python update_services.py [output_dir]                    # Process all models
  python update_services.py --models model1 model2         # Process specific models
  python update_services.py custom_dir --models model1     # Custom output + specific models
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


PROVIDER_NAME = "parasail"
PROVIDER_DISPLAY_NAME = "Parasail"
ENV_API_KEY_NAME = "PARASAIL_API_KEY"


def _sanitize_header_value(value: str) -> str:
    """Strip smart/curly quotes and any non-latin-1 chars that break HTTP headers."""
    for bad, good in [("\u201c", '"'), ("\u201d", '"'), ("\u2018", "'"), ("\u2019", "'")]:
        value = value.replace(bad, good)
    value = value.encode("latin-1", errors="ignore").decode("latin-1").strip()
    value = value.strip('"').strip("'")
    return value


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

# Parasail prices serverless models by parameter size.
# Table from https://docs.parasail.io/parasail-docs/billing/pricing
PRICING_TIERS = [
    (4,           "0.05"),
    (8,           "0.08"),
    (16,          "0.11"),
    (21,          "0.45"),
    (41,          "0.50"),
    (80,          "0.70"),
    (404,         "0.80"),
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
    if any(k in mid for k in ["vision", "vl-", "-vl", "llava", "minicpm"]):
        return "vision_language_model"
    if any(k in mid for k in ["whisper", "audio", "speech", "tts"]):
        return "prerecorded_transcription"
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
            "skipped_models": 0,
            "deprecated_models": [],
            "extraction_date": datetime.now().isoformat(),
            "force_mode": False,
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

    def build_listing_context(self, model_id: str, price: str) -> Dict:
        return {
            "provider_name": PROVIDER_NAME,
            "offering_name": model_id,
            "env_api_key_name": ENV_API_KEY_NAME,
            "status": "ready",
            "list_price": {
                "description": "Pricing Per 1M Tokens",
                "price": price,
                "type": "one_million_tokens",
                "reference": "https://docs.parasail.io/parasail-docs/billing/pricing",
            },
        }

    def build_offering_context(self, model_id: str, model_data: Dict, price: str) -> Dict:
        service_type = derive_service_type(model_id)
        display_name = (
            model_data.get("display_name")
            or model_data.get("name")
            or model_id.split("/")[-1]
        )
        description = model_data.get("description", "")

        details: Dict[str, Any] = {"model_name": model_id}
        for field in ["context_length", "context_window", "max_tokens",
                      "supports_tools", "supports_vision"]:
            if field in model_data:
                details[field] = model_data[field]

        return {
            "provider_name": PROVIDER_NAME,
            "provider_display_name": PROVIDER_DISPLAY_NAME,
            "offering_name": model_id,
            "display_name": display_name,
            "description": description,
            "service_type": service_type,
            "status": "ready",
            "api_base_url": "https://api.parasail.io/v1",
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

    def write_listing(self, model_id: str, price: str, output_dir: Path):
        context = self.build_listing_context(model_id, price)
        content = self._render_template("listing.json.j2", context)
        self._write_file(content, output_dir / "listing.json")

    def write_offering(self, model_id: str, model_data: Dict, price: str, output_dir: Path):
        context = self.build_offering_context(model_id, model_data, price)
        content = self._render_template("offering.json.j2", context)
        self._write_file(content, output_dir / "offering.json")

    def write_summary(self):
        try:
            print(f"   Total models: {self.summary['total_models']}")
            print(f"   Successful extractions: {self.summary['successful_extractions']}")
            print(f"   Skipped models: {self.summary['skipped_models']}")
            print(f"   Deprecated models: {len(self.summary['deprecated_models'])}")
            if self.summary["force_mode"]:
                print("   Force mode: Enabled")
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
        base_path = Path(output_dir)
        if not base_path.exists():
            return

        active_dirs = {m.split("/")[-1].replace(":", "_") for m in active_models}
        deprecated_count = 0

        for item in base_path.iterdir():
            if not item.is_dir() or item.name in active_dirs:
                continue

            deprecated_count += 1
            print(f"  🗑️  Processing deprecated service: {item.name}")

            for json_file in item.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    schema = data.get("schema", "")
                    updated = False

                    if schema == "offering_v1" and data.get("status") != "deprecated":
                        data["status"] = "deprecated"
                        updated = True
                        msg = "offering status to deprecated"
                    elif schema == "listing_v1" and data.get("status") != "upstream_deprecated":
                        data["status"] = "upstream_deprecated"
                        updated = True
                        msg = "listing status to upstream_deprecated"

                    if updated:
                        if dry_run:
                            print(f"    📝 [DRY-RUN] Would update {json_file.name} {msg}")
                        else:
                            with open(json_file, "w", encoding="utf-8") as f:
                                json.dump(data, f, sort_keys=True, indent=2, separators=(",", ": "))
                                f.write("\n")
                            print(f"    ✅ Updated {json_file.name} {msg}")
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
        force: bool = False,
        limit: Optional[int] = None,
        dry_run: bool = False,
    ):
        print("🚀 Starting Parasail model extraction...\n")
        self.summary["force_mode"] = force
        self.summary["processing_limit"] = limit

        if dry_run:
            print("🔍 Dry-run mode enabled - will show what would be done without writing files")
        if force:
            print("💪 Force mode enabled - will overwrite all existing data files")

        if specific_models:
            print(f"🎯 Processing specific models: {', '.join(specific_models)}")
            models = [{"id": model_id} for model_id in specific_models]
            self.summary["total_models"] = len(models)
        else:
            models = self.get_all_models()
            if not models:
                print("❌ No models retrieved. Exiting.")
                return
            if force and limit is None:
                active_ids = [m.get("id", "") for m in models if m.get("id")]
                self.mark_deprecated_services(output_dir, active_ids, dry_run)

        skipped_count = 0
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
            dir_name = model_id.split("/")[-1].replace(":", "_")
            data_dir = base_path / dir_name
            offering_file = data_dir / "offering.json"

            if not force and data_dir.exists() and offering_file.exists():
                print(f"  ⏭️  Skipping {model_id} - files already exist (use --force to overwrite)")
                skipped_count += 1
                self.summary["skipped_models"] += 1
                continue

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
                    print(f"  📝 [DRY-RUN] Would write offering.json + listing.json to {data_dir}")
                    self.summary["successful_extractions"] += 1
                    continue

                print(f"  📝 Writing files to {data_dir}...")
                self.write_offering(model_id, model_data, price, data_dir)

                listing_file = data_dir / "listing.json"
                if listing_file.exists() and not force:
                    print("  ⏭️  Skipping existing listing.json (use --force to overwrite)")
                    print("      💡 Manual customizations can be preserved in listing.override.json")
                else:
                    self.write_listing(model_id, price, data_dir)

                self.summary["successful_extractions"] += 1
                print(f"  ✅ Successfully processed {model_id}")

            except Exception as e:
                print(f"  ❌ Error processing {model_id}: {e}")
                self.summary["failed_extractions"] += 1

        self.write_summary()
        print(f"\n🎉 Extraction complete! Check {output_dir}/ for results.")
        if skipped_count > 0:
            print(f"   ⏭️  Skipped {skipped_count} existing models (use --force to overwrite)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract model data from Parasail API and generate service files"
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="services",
        help="Output directory for service files (default: services)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model IDs to process",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite all existing files",
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
        print("❌ Error: No API key provided. Set the PARASAIL_API_KEY environment variable.")
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
        force=args.force,
        limit=args.limit,
        dry_run=args.dry_run,
    )
