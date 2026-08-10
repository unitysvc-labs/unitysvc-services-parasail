# CLAUDE.md — unitysvc-services-parasail

This repository holds **UnitySVC service data** — the declarative files (plus
tests) that define services published to the UnitySVC gateway/marketplace. It is
*data*, not application code; there is no server here. **`unitysvc-sellers`
(repo + `docs/`) is the authoritative standard** for this format and the CLI;
this file is the quick, repo-local reference and defers to those docs.

## Where the data lives and in what shape

**All service data lives under `specs/`** (files are JSON or TOML — role is fixed
by filename: `offering.json`, `listing.json`, `provider.json`, `service.json`).
Author each service one of **two ways**:

**(a) Full spec** — a concrete folder per service:

    specs/<provider>/<service-name>/
      ├── offering.json      # exactly one — technical spec (upstream, auth, channels, payout)
      ├── listing.json       # one or more — customer-facing (base_url, price, docs, params)
      ├── provider.json      # the provider record
      ├── service.json       # { "service_id": "…" } — identity sidecar; commit it, minimal form
      └── <docs>             # connectivity.*.j2, description.md, code-example.*.j2

**(b) Templates + param data** — one template renders many services (large
catalogs, notification-channel fleets):

    templates/<template>/{provider.json, offering.json.j2, listing.json.j2}
    specs/<provider>/<name>.json          # { "template": "<template>", "parameters": {…} }
    specs/<provider>/<name>.service.json  # identity sidecar
    # Commit the param file + its .service.json — NEVER the rendered folder.
    # specs commands render param files into a temp folder ephemerally.

Either way, the folder (or param-file stem) under `specs/<provider>/` **is** the
service name. The backend derives identity: `service.name ← listing.name`,
`display_name ← listing/offering display_name`, `status ← worst-of components`.

## Offering `description` convention — three paragraphs

The `offering.json` `description` (the service's marketplace description) MUST be
**exactly three paragraphs, separated by `\n\n`**:

1. **Brief** — a one-line summary of what the service provides. Keep it under
   **200 characters**. This is the preview/teaser line.
2. **Detail** — what the service does in depth: the mechanism, the upstream it
   fronts, how bytes/requests flow, billing basis.
3. **How to use** — how a customer actually consumes it: the channels (e.g.
   `byok` vs `plus`), where each is reached, what config each needs. Point at the
   longer "How to use this service" doc for full setup.

The richer, table-driven walkthrough still lives in the service's
`*-description.md` doc; the three-paragraph `description` is the concise summary.

## Environment (required for tests and anything touching staging)

Load the repo's committed secrets manifest first — it provides the
`customer_secrets` values local tests resolve (and is the CI seed source):

    set -a; . ./seller.secrets.txt; set +a      # == .env.example (a symlink to it)

You also need the seller credentials in your environment —
`UNITYSVC_SELLER_API_KEY`, `UNITYSVC_SELLER_API_URL`
(`https://seller.staging.unitysvc.com/v1/`), and `UNITYSVC_API_KEY` (the svcpass
key used as a gateway customer). A 401 / "Missing svcpass API key" means the
environment isn't loaded.

Invoke the CLI as `usvc_seller …` if on PATH, else
`uvx --from unitysvc-sellers usvc_seller …`, else from a local checkout
`uv run --project ~/unitysvc/unitysvc-sellers usvc_seller …`.

## Naming rules (the validator rejects violations)

- `listing.name` MUST equal `<provider>/<service-name>` (its path under `specs/`),
  or be a bare top-level handle. `offering.name` stays the **bare** service name.
- If `listing.name` contains `/`, the first segment MUST equal `provider.name`.
- `user_access_interfaces.<iface>.base_url`: use
  `${API_GATEWAY_BASE_URL}/{{ service_name }}` (resolved from `listing.name` at
  request time) — literal `<provider>/<service>` paths are rejected. Name the
  single static interface `canonical`.
- Segments ≥ 2 chars, start alphanumeric, allowed `[A-Za-z0-9._-]`, at most one
  `@variant`. Single-letter first segments are reserved (`a b c d f g l m r t …`);
  only `a/` is a permitted movable-pointer carve-out.

## seller.secrets.txt / .env.example — the secrets manifest

Every `${ customer_secrets.<NAME> }` referenced in a listing/offering needs a
same-named **seller** secret on the platform (the gateway-side test plugs in a
real value). Seed them from the repo-committed manifest, never GitHub variables:

- Commit `seller.secrets.txt` (symlinked as `.env.example`) at the repo root:
  `export NAME="value"` lines, each preceded by a `#` comment block that is the
  **customer-facing** description (what the customer sets + how to obtain it). It
  is both shell-sourceable for local tests and the upload workflow's seed source
  (`usvc_seller secrets upload seller.secrets.txt`).
- Seed the **mock** value; keep the manifest exhaustive —
  `grep -rho '\${ customer_secrets\.[A-Z_]*' specs/ | sort -u` should have no
  name missing from it. A missing name is silently skipped ⇒ the gateway test
  fails with no obvious cause.
- **Namespace** secret names by service (`HTTP_RELAY_BASE_URL`,
  `SMTP_RELAY_PASSWORD`, `<PROVIDER>_API_KEY`) — bare names collide.
- Sensitive (key/password/token) → `customer_secrets`. Operational
  (host/port/url/flag) → a direct `{{ params.X }}` with a default, not a secret.

## Verification pipeline — all four green before "ready"

    usvc_seller specs validate                     # schema + cross-file, no network
    usvc_seller specs format                       # canonical formatting; commit result
    usvc_seller specs run-tests <name>             # UPSTREAM-side (docs vs upstream directly)
    usvc_seller specs upload <name>                # push to staging (re-upload ⇒ revision)
    usvc_seller services run-tests <name> --force  # GATEWAY-side (route + svcpass + upstream)

- **Selector NAME** fnmatches `listing.name`; `%` is a shell-safe synonym for `*`
  (`cohere/%`, `%-byok`, `%command%`). Wildcards only at start/end. If a name
  matches an active row **and** a pending revision, single-row commands need
  `--id <prefix>`.
- Testing does NOT require the service to be public/active. `set-visibility` +
  `submit` is **publishing** — a separate explicit step, not part of testing.

### Persistent `pending` for on-wire testing without the CLI runner

`services run-tests` flips the service to `pending` only for the duration of the
run. To test a service **directly with your own scripts** (curl, the rendered
code examples, a REST client) against the gateway, make `pending` persistent
first, then revert when done:

    usvc_seller services enable-testing <name>   # draft|rejected|suspended → pending (routable)
    # ... run your own scripts against the gateway ...
    usvc_seller services withdraw <name>         # pending|rejected → draft (revert)

`enable-testing` is a **pure status change** — it does NOT run the activation
pipeline, does NOT progress the service to `active`, and is NOT a submission for
review (use `submit` for that). It just makes the service routable so on-wire
tests resolve. Same `NAME` / `--id` / `--local-ids` / `--all` selectors as the
other `services` commands.

## Testing gotchas (these have bitten us repeatedly)

- **Draft status is NOT a cause of test failures.** `run-tests` temporarily
  flips the service to `pending` so the gateway routes to it. Do not chase draft
  status — a service in `draft`/`rejected` while testing is expected and fine.
- A gateway **550 "No active enrollment found"** is about enrollment →
  AccessInterface routing (the enrollment-scoped AI + bridge + rendered
  `routing_key.username`), NOT status. If a clean re-provision doesn't fix it,
  suspect stale/conflicting staging data.
- Tests run in **two modes**: `specs run-tests` renders docs against the upstream
  directly (`local_testing` true branch, seller credential); `services run-tests`
  renders the same docs against the gateway (customer svcpass key). A failure
  only in gateway mode ⇒ wrong base_url shape (`{{ service_name }}`), wrong
  `api_key` disposition, or not re-uploaded since the data changed.
- An "empty host `:587`" / connectivity failure when running **locally** is
  usually a missing env var (e.g. `SMTP_GATEWAY_HOST`), not a service defect.
- `specs run-tests` Python `ModuleNotFoundError: requests` ⇒ wrong Python;
  `source ~/unitysvc/.venv/bin/activate`. Environmental, not your data.

## service.json sidecar

- Auto-written by `specs upload` (and the CI upload workflow); carries the
  backend `service_id` so later uploads target the existing service. **Commit
  whatever the tool writes — don't hand-edit.** `service_id` is the only
  load-bearing field.
- Two committed shapes, both valid: a **concrete** `service.json` carries the CI
  write-back's richer form (`service_id` + derived `name` / `display_name` /
  `status` / `time_created`); a **param-file** `<name>.service.json` stays
  minimal `{ "service_id": "…" }`.
- Delete the sidecar only to deliberately re-upload as a brand-new service. If a
  rename leaves the derived fields stale (`name` of the old service), correct
  them or let the next upload/CI write-back refresh them.

## Tests: connectivity is mandatory; presets come from unitysvc-data

Every service needs at least one connectivity-test document. Two forms:

- A **`$doc_preset`** from the **`unitysvc-data`** package
  (`~/unitysvc/unitysvc-data`) — e.g. `llm_connectivity`, `api_connectivity`,
  `smtp_connectivity_v2`. That repo is where test/connectivity presets live and
  are added.
- A **local Jinja file** (`connectivity.*.j2`, `code-example.*.j2`) under the
  service dir for custom probes — handle both modes via the `local_testing` flag.

Without a connectivity test the service is untestable and cannot activate.

## Standards & references

- **`unitysvc-sellers`** — the authoritative standard. Read
  `~/unitysvc/unitysvc-sellers/docs/`: `file-schemas.md`, `pricing.md`,
  `naming-conventions.md`, `secrets-and-variables.md`, `service-templates.md`,
  `documenting-services.md`, `cli-reference.md`, `seller-lifecycle.md`. When this
  file conflicts with those docs, the docs win.
- **`unitysvc-data`** — connectivity-test / code-example presets (`$doc_preset`).
- **`unitysvc-services-demo`** — closest working examples of every pattern:
  `~/unitysvc/unitysvc-services-demo/specs/unitysvc-demo/`.
