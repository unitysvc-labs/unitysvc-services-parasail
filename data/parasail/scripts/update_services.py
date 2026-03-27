#!/usr/bin/env python3
"""
update_services.py - Extract model data from Parasail API and pricing pages

This script:
1. Retrieves all models from Parasail API (or processes specific models if --models is used)
2. Extracts pricing information from web pages using BeautifulSoup
3. Gets detailed model information from API endpoints
4. Writes organized data to service.json and listing.json files
5. Creates summary and flags deprecated directories

Usage:
  python update_services.py [output_dir]                    # Process all models
  python update_services.py --models model1 model2         # Process specific models
  python update_services.py custom_dir --models model1     # Custom output directory + specific models
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

from bs4 import BeautifulSoup


def _sanitize_header_value(value: str) -> str:
    """Strip smart/curly quotes and any non-latin-1 chars that break HTTP headers."""
    for bad, good in [("\u201c", '"'), ("\u201d", '"'), ("\u2018", "'"), ("\u2019", "'")]:
        value = value.replace(bad, good)
    value = value.encode("latin-1", errors="ignore").decode("latin-1").strip()
    # Strip any surrounding straight quotes that usvc may wrap values with
    value = value.strip('"').strip("'")
    return value


class ParasailModelExtractor:
    def __init__(self, api_key: str, api_base_url: str, model_base_url: str):
        api_key = _sanitize_header_value(api_key)
        self.api_key = api_key
        self.api_base_url = (api_base_url or "https://api.parasail.io/v1").strip()
        # e.g. https://www.saas.parasail.io/serverless
        self.model_base_url = (model_base_url or "https://www.saas.parasail.io/serverless").strip()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            }
        )
        self.extracted_data = {}
        self.summary = {
            "total_models": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "skipped_models": 0,
            "pricing_found": 0,
            "deprecated_models": [],
            "extraction_date": datetime.now().isoformat(),
            "force_mode": False,
            "processing_limit": None,
        }

    def get_all_models(self) -> List[Dict]:
        """Retrieve all models from Parasail API"""
        print("🔍 Fetching all models from Parasail API...")

        url = f"{self.api_base_url}/models"
        try:
            print(f"📄 Fetching models...")
            response = self.session.get(url, params={"limit": 1000})
            response.raise_for_status()

            data = response.json()
            # OpenAI-compatible: {"object": "list", "data": [...]}
            all_models = data.get("data", data) if isinstance(data, dict) else data

            if not isinstance(all_models, list):
                print(f"❌ Unexpected /models response shape: {type(all_models)}")
                return []

            self.summary["total_models"] = len(all_models)
            print(f"✅ Found {len(all_models)} models total")

            # Sort models by id for easier debugging
            all_models.sort(key=lambda x: x.get("id", ""))
            return all_models

        except requests.RequestException as e:
            print(f"❌ Error fetching models: {e}")
            return []

    def extract_pricing_from_page(self, model_id: str) -> Optional[Dict]:
        """Derive pricing from Parasail's parameter-size-based pricing table.

        Parasail prices serverless models based on parameter count (not per-model pages).
        Table from https://docs.parasail.io/parasail-docs/billing/pricing:
            0-4B:       $0.05/1M tokens
            4.1-8B:     $0.08/1M tokens
            8.1-16B:    $0.11/1M tokens
            16.1-21B:   $0.45/1M tokens
            21.1-41B:   $0.50/1M tokens
            41.1-80B:   $0.70/1M tokens
            80.1-404B:  $0.80/1M tokens
            405B+:      $1.75/1M tokens
        """
        # Pricing tiers: (max_params_billions, price_per_1M)
        PRICING_TIERS = [
            (4,    "0.05"),
            (8,    "0.08"),
            (16,   "0.11"),
            (21,   "0.45"),
            (41,   "0.50"),
            (80,   "0.70"),
            (404,  "0.80"),
            (float("inf"), "1.75"),
        ]

        # Extract parameter count in billions from model id
        # Patterns: 7b, 70b, 8x7b (MoE -> sum), 235b-a22b (active params), 72b, 3.3-70b etc.
        model_lower = model_id.lower()

        # MoE pattern: NxMb (e.g. 8x7b = 56b total)
        moe_match = re.search(r"(\d+)x(\d+\.?\d*)b", model_lower)
        if moe_match:
            params_b = int(moe_match.group(1)) * float(moe_match.group(2))
        else:
            # Standard pattern: largest number followed by b (e.g. 70b, 3.3-70b, 235b)
            # For MoE with active params like 235b-a22b, use total (235)
            size_matches = re.findall(r"(\d+\.?\d*)b", model_lower)
            if size_matches:
                params_b = max(float(x) for x in size_matches)
            else:
                # No size found — use mid-range default ($0.50)
                print(f"  ⚠️  Could not determine parameter count from {model_id}, using default pricing")
                params_b = 30

        print(f"  📊 Estimated params: {params_b}B for {model_id}")

        # Look up price tier
        price = PRICING_TIERS[-1][1]
        for max_b, tier_price in PRICING_TIERS:
            if params_b <= max_b:
                price = tier_price
                break

        pricing_info = {
            "unit": "Pricing Per 1M Tokens",
            "price": f"${price}",
            "reference": "https://docs.parasail.io/parasail-docs/billing/pricing",
        }
        self.summary["pricing_found"] += 1
        print(f"  ✅ Derived pricing: {pricing_info['price']} (params: {params_b}B)")
        return pricing_info
    def get_model_details(self, model_id: str) -> Optional[Dict]:
        """Get detailed model information from API endpoint"""
        endpoint = f"{self.api_base_url}/models/{model_id}"

        try:
            response = self.session.get(endpoint, timeout=10)

            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Retrieved API details")
                return data
            elif response.status_code == 404:
                # Parasail does not expose per-model detail endpoints
                return None
            else:
                response.raise_for_status()

        except requests.RequestException:
            return None

        print(f"  ⚠️  No API details available")
        return None

    def parse_pricing_string(
        self, price_string: str, pricing_unit: str
    ) -> Dict[str, Any]:
        """Parse pricing string and return structured pricing data"""
        pricing_data = {}

        if pricing_unit == "Pricing Per 1M Tokens Input/Output":
            if " / " in price_string:
                input_price, output_price = price_string.split(" / ")
                pricing_data["price_input"] = str(input_price.strip().replace("$", ""))
                pricing_data["price_output"] = str(output_price.strip().replace("$", ""))
            else:
                price = str(price_string.strip().replace("$", ""))
                pricing_data["price_input"] = price
                pricing_data["price_output"] = price
        else:
            pricing_data["price"] = str(price_string.strip().replace("$", ""))

        return pricing_data

    def create_pricing_info_structure(self, pricing_data: Dict) -> Dict[str, Any]:
        """Create pricing info structure based on pricing data"""
        if not pricing_data:
            return {}

        parsed_pricing = self.parse_pricing_string(
            pricing_data["price"], pricing_data["unit"]
        )

        match pricing_data["unit"]:
            case "Pricing Per 1M Tokens Input/Output":
                return {
                    "description": pricing_data["unit"],
                    "input": parsed_pricing["price_input"],
                    "output": parsed_pricing["price_output"],
                    "type": "one_million_tokens",
                    "reference": pricing_data.get("reference", None),
                }
            case "Pricing Per 1M Tokens":
                return {
                    "description": pricing_data["unit"],
                    "price": parsed_pricing["price"],
                    "type": "one_million_tokens",
                    "reference": pricing_data.get("reference", None),
                }
            case "Pricing Per Image":
                return {
                    "description": pricing_data["unit"],
                    "price": parsed_pricing["price"],
                    "unit": "image",
                    "reference": pricing_data.get("reference", None),
                }
            case "Pricing Per Step":
                return {
                    "description": pricing_data["unit"],
                    "price": parsed_pricing["price"],
                    "type": "step",
                    "reference": pricing_data.get("reference", None),
                }
            case _:
                raise RuntimeError(f"Unknown pricing unit {pricing_data['unit']}")

    def determine_service_type(
        self, model_id: str, pricing_data: Optional[Dict] = None
    ) -> str:
        """Determine service type based on model id and pricing unit"""
        model_lower = model_id.lower()

        # First, check pricing unit which is often a reliable indicator
        if pricing_data and "unit" in pricing_data:
            pricing_unit = pricing_data["unit"]
            if "image" in pricing_unit.lower():
                return "image_generation"
            elif "step" in pricing_unit.lower():
                return "image_generation"
            elif "token" in pricing_unit.lower():
                if any(keyword in model_lower for keyword in ["embedding", "embed"]):
                    return "embedding"
                elif any(
                    keyword in model_lower
                    for keyword in ["vision", "vl-", "-vl", "llava", "minicpm"]
                ):
                    return "vision_language_model"
                else:
                    return "llm"

        # Fallback to model name analysis
        if any(keyword in model_lower for keyword in ["embedding", "embed"]):
            return "embedding"

        if any(
            keyword in model_lower
            for keyword in ["flux", "dalle", "stable-diffusion", "sdxl", "controlnet"]
        ):
            return "image_generation"

        if any(
            keyword in model_lower
            for keyword in ["vision", "vl-", "-vl", "llava", "minicpm"]
        ):
            return "vision_language_model"

        if any(
            keyword in model_lower
            for keyword in ["whisper", "audio", "speech", "transcri", "tts"]
        ):
            return "prerecorded_transcription"

        return "llm"

    def create_service_data_structure(
        self,
        model_id: str,
        model_data: Dict,
        pricing_data: Optional[Dict],
        api_key: str,
    ) -> Dict:
        """Create a structured service configuration for the model"""

        service_type = self.determine_service_type(model_id, pricing_data)

        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        handled_model_fields = [
            "id",
            "object",
            "created",
            "owned_by",
        ]

        top_level_model_fields = [
            "context_length",
            "context_window",
            "max_tokens",
            "supports_tools",
            "supports_vision",
            "supports_function_calling",
            "input_modalities",
            "output_modalities",
            "pricing",
        ]

        service_config = {
            "schema": "service_v1",
            "time_created": timestamp,
            "name": model_id.split("/")[-1],
            "currency": "USD",
            # type of service to group services
            "service_type": service_type,
            # common display name for the service, allowing across provider linking
            "display_name": model_data.get("display_name") or model_data.get("name") or model_id.split("/")[-1],
            "version": "",
            "description": model_data.get("description", ""),
            "upstream_status": "ready",
            "details": {"model_name": model_id},
            "upstream_access_interface": {},
            "seller_price": {
                "type": "revenue_share",
                "percentage": "100.00",
                "description": "Pricing Per 1M Tokens",
            },
        }

        # top level details
        for field in top_level_model_fields:
            if field in model_data:
                service_config["details"][field] = model_data[field]

        for field in model_data.keys():
            if field not in top_level_model_fields + handled_model_fields:
                print(f" {field} for model {model_id} is not processed.")

        # Add pricing information if available
        if service_config["upstream_status"] == "ready":
            # if no pricing information, the service cannot be ready
            service_config["upstream_status"] == "uploading"

        service_config["upstream_access_interface"] = {
            "name": "Parasail API",
            "api_key": api_key,
            "base_url": "https://api.parasail.io/v1",
            "access_method": "http",
            "rate_limits": [
                {
                    "description": "Requests per minute",
                    "limit": 60,
                    "unit": "requests",
                    "window": "minute",
                },
                {
                    "description": "Input tokens per minute",
                    "limit": 100000,
                    "unit": "input_tokens",
                    "window": "minute",
                },
                {
                    "description": "Output tokens per minute",
                    "limit": 10000,
                    "unit": "output_tokens",
                    "window": "minute",
                },
            ],
        }
        return service_config

    def create_operation_data_structure(
        self,
        model_id: str,
        pricing_data: Optional[Dict],
        ready: bool,
    ) -> Dict:
        """Create a structured operation configuration for the model"""

        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        operation_config = {
            "schema": "listing_v1",
            "seller_name": "svcreseller",
            "currency": "USD",
            "time_created": timestamp,
            "listing_status": "ready" if ready else "unknown",
            # type of service to group services
            "user_access_interfaces": [],
            # common display name for the service, allowing across provider linking
            "customer_price": {},
        }

        # Add pricing information if available
        if pricing_data is not None:
            pricing_info = self.create_pricing_info_structure(pricing_data)
            operation_config["customer_price"] = pricing_info

        operation_config["user_access_interfaces"] = [
            {
                "name": "Provider API",
                "base_url": "${GATEWAY_BASE_URL}/p/parasail.io",
                "access_method": "http",
                "documents": [
                    {
                        "title": "Python code example",
                        "description": "Example code to use the model",
                        "mime_type": "python",
                        "category": "code_example",
                        "file_path": "../../docs/code_example.py.j2",
                        "is_active": True,
                        "is_public": True,
                    },
                    {
                        "title": "Python function calling code example",
                        "description": "Example code to use the model",
                        "mime_type": "python",
                        "category": "code_example",
                        "file_path": "../../docs/code_example_1.py.j2",
                        "is_active": True,
                        "is_public": True,
                    },
                    {
                        "title": "JavaScript code example",
                        "description": "Example code to use the model",
                        "mime_type": "javascript",
                        "category": "code_example",
                        "file_path": "../../docs/code_example.js.j2",
                        "is_active": True,
                        "is_public": True,
                    },
                    {
                        "title": "cURL code example",
                        "description": "Example code to use the model",
                        "mime_type": "bash",
                        "category": "code_example",
                        "file_path": "../../docs/code_example.sh.j2",
                        "is_active": True,
                        "is_public": True,
                    },
                ],
            }
        ]
        return operation_config

    def write_service_files(self, service_data, output_dir):
        """Write service.json file"""
        base_path = Path(output_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        output_file = base_path / "service.json"

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    service_data, f, sort_keys=True, indent=2, separators=(",", ": ")
                )
                f.write("\n")
            print(f"  ✅ Written: {output_file}")
        except Exception as e:
            print(f"  ❌ Error writing {output_file}: {e}")

    def write_listing_files(self, operation_data, output_dir):
        """Write listing.json file"""
        base_path = Path(output_dir)
        base_path.mkdir(parents=True, exist_ok=True)

        # Create shared examples directory
        shared_path = base_path / ".." / ".." / "docs"
        shared_path.mkdir(parents=True, exist_ok=True)

        output_file = base_path / "listing.json"

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    operation_data, f, sort_keys=True, indent=2, separators=(",", ": ")
                )
                f.write("\n")
            print(f"  ✅ Written: {output_file}")
        except Exception as e:
            print(f"  ❌ Error writing {output_file}: {e}")

    def write_summary(self):
        """Write extraction summary"""
        try:
            print(f"   Total models: {self.summary['total_models']}")
            print(
                f"   Successful extractions: {self.summary['successful_extractions']}"
            )
            print(f"   Skipped models: {self.summary['skipped_models']}")
            print(f"   With pricing data: {self.summary['pricing_found']}")
            print(f"   Deprecated models: {len(self.summary['deprecated_models'])}")
            if self.summary["force_mode"]:
                print(f"   Force mode: Enabled")
            if self.summary["processing_limit"]:
                print(f"   Processing limit: {self.summary['processing_limit']}")
        except Exception as e:
            print(f"❌ Error writing summary: {e}")

    def mark_deprecated_services(
        self, output_dir: str, active_models: List[str], dry_run: bool = False
    ):
        """Mark services as deprecated if they no longer exist in active_models"""
        print("🔍 Checking for deprecated services...")

        base_path = Path(output_dir)
        if not base_path.exists():
            print(f"  ⚠️  Output directory {output_dir} does not exist")
            return

        active_service_dirs = {
            model_id.split("/")[-1].replace(":", "_") for model_id in active_models
        }

        print(f"  Found {len(active_service_dirs)} active models")

        deprecated_count = 0
        for item in base_path.iterdir():
            if not item.is_dir():
                continue

            service_dir = item.name

            if service_dir in active_service_dirs:
                continue

            deprecated_count += 1
            print(f"  🗑️  Processing deprecated service: {service_dir}")

            for json_file in item.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    schema = data.get("schema")
                    updated = False

                    if schema == "service_v1":
                        current_status = data.get("upstream_status", "unknown")
                        if current_status != "deprecated":
                            data["upstream_status"] = "deprecated"
                            updated = True
                            status_msg = f"service upstream_status to deprecated"
                        else:
                            print(
                                f"    ⏭️  {json_file.name} service already marked as deprecated"
                            )

                    elif schema == "listing_v1":
                        current_op_status = data.get("listing_status", "unknown")
                        if current_op_status != "upstream_deprecated":
                            data["listing_status"] = "upstream_deprecated"
                            updated = True
                            status_msg = (
                                f"operation listing_status to upstream_deprecated"
                            )
                        else:
                            print(
                                f"    ⏭️  {json_file.name} operation already marked as upstream_deprecated"
                            )

                    if updated:
                        if dry_run:
                            print(
                                f"    📝 [DRY-RUN] Would update {json_file.name} {status_msg}"
                            )
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
                            print(f"    ✅ Updated {json_file.name} {status_msg}")

                except Exception as e:
                    print(f"    ❌ Error updating {json_file}: {e}")

        if deprecated_count == 0:
            print("  ✅ No deprecated services found")
        else:
            print(f"  🗑️  Processed {deprecated_count} deprecated services")

    def process_all_models(
        self,
        output_dir: str = "services",
        specific_models: Optional[List[str]] = None,
        force: bool = False,
        limit: Optional[int] = None,
        dry_run: bool = False,
    ):
        """Main processing function"""
        print("🚀 Starting Parasail model extraction...\n")

        self.summary["force_mode"] = force
        self.summary["processing_limit"] = limit
        if dry_run:
            print(
                "🔍 Dry-run mode enabled - will show what would be done without writing files"
            )
        if force:
            print(
                "💪 Force mode enabled - will overwrite all existing data files (service.json and listing.json)"
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

            if force and limit is None:
                active_model_ids = [
                    model.get("id", "") for model in models if model.get("id")
                ]
                self.mark_deprecated_services(output_dir, active_model_ids, dry_run)

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
            data_file = data_dir / "service.json"

            if not force and data_dir.exists() and data_file.exists():
                print(
                    f"  ⏭️  Skipping {model_id} - service file already exists (use --force to overwrite)"
                )
                skipped_count += 1
                self.summary["skipped_models"] += 1
                continue

            processed_count += 1

            try:
                # Get API details
                details = self.get_model_details(model_id)
                if details:
                    model_data = model_data | details
                time.sleep(0.1)  # Rate limiting

                # Get pricing data
                try:
                    pricing_data = self.extract_pricing_from_page(model_id)
                    time.sleep(0.5)  # Rate limiting
                    if not pricing_data:
                        print(f"  ⚠️  No pricing data found for {model_id}")
                        continue
                except Exception as e:
                    sys.exit(f"  ❌ Error parsing pricing page: {e}")

                # Create service configuration
                service_config = self.create_service_data_structure(
                    model_id,
                    model_data,
                    pricing_data,
                    api_key,
                )

                # Create operation configuration
                operation_config = self.create_operation_data_structure(
                    model_id,
                    pricing_data,
                    service_config["upstream_status"] == "ready",
                )

                print(f"  📝 Generated service data")
                self.extracted_data[model_id] = service_config
                self.summary["successful_extractions"] += 1

                # Write service file
                if dry_run:
                    print(f"  📝 [DRY-RUN] Would write service files to {data_dir}")
                else:
                    print(f"  📝 Writing service files to {data_dir}...")
                    self.write_service_files(service_config, data_dir)

                # Write listing file
                listing_file = data_dir / "listing.json"
                if listing_file.exists() and not force:
                    print(
                        "  ⏭️  Skipping existing listing.json (use --force to overwrite)"
                    )
                    print(
                        "      💡 Manual customizations can be preserved in listing.override.json"
                    )
                else:
                    if dry_run:
                        action = "overwrite" if listing_file.exists() else "write"
                        print(
                            f"  📝 [DRY-RUN] Would {action} listing files to {data_dir}"
                        )
                    else:
                        action = "Overwriting" if listing_file.exists() else "Writing"
                        print(f"  📝 {action} listing files to {data_dir}...")
                        self.write_listing_files(operation_config, data_dir)

                print(f"  ✅ Successfully processed {model_id}")

            except Exception as e:
                print(f"  ❌ Error processing {model_id}: {e}")
                self.summary["failed_extractions"] += 1

        self.write_summary()

        print(f"\n🎉 Extraction complete! Check {output_dir}/ for results.")
        if skipped_count > 0:
            print(
                f"   ⏭️  Skipped {skipped_count} existing models (use --force to overwrite)"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract model data from Parasail API and pricing pages"
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
        help="Specific model IDs to process (e.g., --models parasail-deepseek-r1)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite all existing data files (service.json and listing.json). Without this flag, existing files will be skipped. Manual customizations can be preserved in .override.json files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of models to process. Skipped models (when directories already exist) are not counted towards this limit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually writing files.",
    )

    args = parser.parse_args()

    # Get API key from environment variables
    api_key = os.environ.get("PARASAIL_API_KEY")
    api_base_url = os.environ.get("PARASAIL_API_BASE_URL")
    model_base_url = os.environ.get("PARASAIL_MODEL_BASE_URL")

    # Sanitize env vars — strip smart/curly quotes that usvc may inject
    if api_key:
        api_key = _sanitize_header_value(api_key)

    if not api_key:
        print(
            "❌ Error: No API key provided. Set the PARASAIL_API_KEY environment variable."
        )
        sys.exit(1)

    # Initialize extractor
    extractor = ParasailModelExtractor(api_key, api_base_url, model_base_url)

    # Process models (all or specific ones)
    extractor.process_all_models(
        args.output_dir,
        specific_models=args.models,
        force=args.force,
        limit=args.limit,
        dry_run=args.dry_run,
    )
