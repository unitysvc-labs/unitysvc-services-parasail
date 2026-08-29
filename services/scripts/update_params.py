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
            # Tool-calling probe outcomes (see _probe_tools).
            "tools_supported": 0,
            "tools_unsupported": 0,
            "tools_unknown": 0,
            "tools_skipped_non_chat": 0,
        }
        #: (model_id, override_value, probed_value) where a human override in
        #: <model>.override.json contradicts what the probe measured. The
        #: override still wins — it is merged at render time and this script
        #: never writes one — but a disagreement is evidence about the probe,
        #: so it is collected and reported rather than silently reconciled.
        self.tools_override_conflicts: List[tuple] = []

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
    # Tool-calling probe
    # ------------------------------------------------------------------

    #: Pause between probe requests, and the base for the bounded 429 backoff.
    #: ~99 models means ~99 POSTs against a shared seller key; pacing them keeps
    #: the run from rate-limiting itself into a wall of UNKNOWNs.
    _PROBE_PACE_SECONDS = 0.5
    _PROBE_MAX_RETRIES = 2

    #: The probe payload mirrors the catalog's canonical tools code example, so
    #: a pass here means the example a customer copies will also pass.
    _TOOL_PROBE_TOOL = {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    @staticmethod
    def _error_message(response) -> str:
        """Best-effort human-readable error text from a provider response."""
        try:
            payload = response.json()
        except Exception:
            return (response.text or "")[:200]
        err = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(err, dict):
            return str(err.get("message", ""))
        if isinstance(err, str):
            return err
        return str(payload)[:200]

    def _chat_probe(self, model_id: str, *, tools: bool):
        """One /v1/chat/completions POST, with a bounded retry on 429.

        Returns the response, or None if the request never completed. The body
        carries no token cap or sampling knob on purpose: a parameter one model
        family rejects would read as a refusal of the thing being measured.
        """
        body: Dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": "What time is it?"}],
        }
        if tools:
            body["tools"] = [self._TOOL_PROBE_TOOL]
        url = f"{self.api_base_url}/chat/completions"
        for attempt in range(self._PROBE_MAX_RETRIES + 1):
            try:
                response = self.session.post(url, json=body, timeout=60)
            except requests.RequestException as exc:
                print(f"  ⚠️  probe transport error ({exc})")
                return None
            if response.status_code == 429 and attempt < self._PROBE_MAX_RETRIES:
                wait = self._PROBE_PACE_SECONDS * (2 ** (attempt + 1))
                print(f"  ⏳ 429 rate-limited — backing off {wait:.1f}s")
                time.sleep(wait)
                continue
            return response
        return None

    def _probe_tools(self, model_id: str) -> Optional[bool]:
        """Does this deployment accept a ``tools`` request?

        Returns True (accepted), False (answered the identical call WITHOUT
        tools but explicitly refused it WITH tools), or None for UNKNOWN.

        **None is not False.** Parasail's /v1/models reports
        ``supports_tools: null`` for every model, so this probe is the only
        positive signal there is — but a probe that cannot reach a verdict must
        never be recorded as a negative. Writing False on a rate limit, an
        exhausted key or an upstream blip would strip ``feature:func-call``
        from a working model and delete its tools code example, which is how a
        wave of false rejections happened here before. Callers keep whatever
        the param file already holds when this returns None.

        The refusal is confirmed *differentially* rather than by matching error
        text. A 400 on a ``tools`` request can mean "this deployment has no
        tool-call parser" or it can mean the model is unserviceable right now;
        only the second request — the identical call minus ``tools`` —
        separates the two. If the model answers without tools and refuses with
        them, the ``tools`` parameter is the only difference and the refusal is
        real. Anything else is UNKNOWN, including the 500 that some
        parser-less deployments throw instead of a 400: a 5xx is
        indistinguishable from an outage, so those genuine negatives stay in
        the .override.json denylist rather than being guessed at here.
        """
        response = self._chat_probe(model_id, tools=True)
        if response is None:
            return None
        if response.status_code == 200:
            print("  🔧 tools supported")
            return True
        if response.status_code not in (400, 404):
            print(
                f"  ❓ tools probe HTTP {response.status_code} — UNKNOWN "
                "(not a refusal; keeping the committed value)"
            )
            return None

        message = self._error_message(response)
        control = self._chat_probe(model_id, tools=False)
        if control is None or control.status_code != 200:
            observed = (
                "transport error" if control is None else f"HTTP {control.status_code}"
            )
            print(
                f"  ❓ tools probe {response.status_code} but the control call "
                f"(same request, no tools) also failed ({observed}) — UNKNOWN, "
                "the model itself is unserviceable"
            )
            return None
        print(f"  🚫 tools NOT supported ({message[:70]})")
        return False

    def resolve_supports_tools(
        self, model_id: str, capability: str, param_file: Path, override_file: Path
    ) -> Optional[bool]:
        """Measured tool support for one model, or the committed value.

        Never returns a value derived from a failed measurement: on UNKNOWN it
        falls back to what is already committed, and if nothing is committed it
        returns None so the caller omits the key entirely. A null is never
        written — ``supports_tools: null`` would render ``"tools": null`` into
        the listing's example collection.
        """
        committed = self._param_field(param_file, "supports_tools")

        if capability != "chat":
            # /v1/chat/completions is the wrong endpoint for an embedding, TTS
            # or rerank deployment, so a refusal there measures nothing. The
            # offering template gates feature:func-call on the chat capability
            # anyway, so there is nothing here worth a request.
            print("  ⏭️  non-chat capability — skipping tools probe")
            self.summary["tools_skipped_non_chat"] += 1
            return committed

        probed = self._probe_tools(model_id)
        time.sleep(self._PROBE_PACE_SECONDS)

        if probed is None:
            self.summary["tools_unknown"] += 1
            if committed is None:
                print("  ❓ no committed value either — leaving supports_tools unset")
            else:
                print(f"  ↩️  keeping committed supports_tools={committed}")
            return committed

        self.summary["tools_supported" if probed else "tools_unsupported"] += 1

        override = self._param_field(override_file, "supports_tools")
        if override is not None and override != probed:
            # The override wins regardless — it is merged at render time and
            # this script never writes override files — but record it.
            self.tools_override_conflicts.append((model_id, override, probed))
            print(
                f"  ⚠️  override says supports_tools={override} but the probe "
                f"measured {probed}; the override still wins at render time"
            )
        return probed

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
        supports_tools: Optional[bool] = None,
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

        # Gate for the function-calling code example and for the
        # `feature:func-call` tag. Measured per model by `_probe_tools` and
        # passed in by the caller — Parasail's model API still reports
        # supports_tools=null for everything, so a live request is the only
        # positive signal available. `None` means the probe reached no verdict
        # AND nothing is committed, in which case the key is left out of the
        # param file entirely: writing null would render `"tools": null` into
        # the listing's example collection. Corrections still live in the
        # per-model <name>.override.json companions (merged at render time),
        # which this script only ever reads.
        context = {
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
            "details": details,
            "payout_price": {
                "description": "Pricing Per 1M Tokens",
                "price": price,
                "type": "one_million_tokens",
                "reference": "https://docs.parasail.io/parasail-docs/billing/pricing",
            },
        }
        if supports_tools is not None:
            context["supports_tools"] = supports_tools
        return context

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
    def _param_field(path: Path, field: str) -> Any:
        """Read one field out of a committed param (or override) file's
        ``parameters`` block, or None if the file, the block or the field is
        missing or unreadable.

        This is how a run that could not measure something keeps what is
        already committed instead of overwriting it — an unreadable file must
        read as "unknown", never as a new value.
        """
        if path.is_file():
            try:
                data = json.loads(path.read_text())
                return (data.get("parameters") or {}).get(field)
            except Exception:
                return None
        return None

    @classmethod
    def _param_time_created(cls, path: Path) -> Optional[str]:
        """time_created recorded inside a committed param file's ``parameters``
        (the post-migration home of the field), so re-runs stay idempotent."""
        return cls._param_field(path, "time_created")

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
            print(
                "   Tool probe: "
                f"{self.summary['tools_supported']} supported, "
                f"{self.summary['tools_unsupported']} refused, "
                f"{self.summary['tools_unknown']} UNKNOWN (committed value kept), "
                f"{self.summary['tools_skipped_non_chat']} non-chat (not probed)"
            )
            # An UNKNOWN is not a failure, but a run that is mostly UNKNOWN
            # measured nothing — say so rather than letting the counts imply
            # the catalog was verified.
            probed = self.summary["tools_supported"] + self.summary["tools_unsupported"]
            if self.summary["tools_unknown"] > probed:
                print(
                    "   ⚠️  more models were UNKNOWN than were measured — treat "
                    "supports_tools in this run as carried over, not verified."
                )
            for model_id, override, measured in self.tools_override_conflicts:
                print(
                    f"   ⚠️  override conflict: {model_id} "
                    f"override={override} probe={measured} (override wins)"
                )
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

                # Preserve time_created (committed param file first, then the
                # legacy expanded offering.json) so unchanged services don't
                # churn; the merged offering + listing render context becomes
                # one param file.
                param_file = base_path / PROVIDER_NAME / f"{model_id}.json"
                created = self._param_time_created(
                    param_file
                ) or self._existing_time_created(data_dir / "offering.json")
                supports_tools = self.resolve_supports_tools(
                    model_id,
                    derive_capability(model_id, derive_service_type(model_id)),
                    param_file,
                    base_path / PROVIDER_NAME / f"{model_id}.override.json",
                )
                offering = self.build_offering_context(
                    model_id,
                    model_data,
                    price,
                    time_created=created,
                    supports_tools=supports_tools,
                )
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
