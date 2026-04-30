# UnitySVC Services Template

A template repository for digital service providers to manage their services on the UnitySVC platform.

Use this template to create your own service data repository with automated validation and upload workflows.

**[Full Documentation](https://unitysvc-services.readthedocs.io)** | **[Getting Started Guide](https://unitysvc-services.readthedocs.io/en/latest/getting-started/)** | **[CLI Reference](https://unitysvc-services.readthedocs.io/en/latest/cli-reference/)**

## Quick Start

### 1. Create Your Repository from Template

1. Click the **"Use this template"** button at the top of this repository
2. Choose a name for your repository (e.g., `unitysvc-services-yourcompany`)
3. Clone your new repository locally

### 2. Install unitysvc-services CLI

```bash
pip install unitysvc-services
```

### 3. Customize Your Data

Replace the example data in the `data/` directory with your actual service information:

**Required changes:**

- [ ] Update `data/${provider}/provider.toml` with your company information
- [ ] Update `data/${provider}/README.md` with your company description
- [ ] Update `data/${provider}/docs/` with your documentation and code examples
- [ ] Replace example services in `data/${provider}/services/` with your actual services

**Recommended naming:**

- Provider name: Use your company name (e.g., "acme-corp")
- Service names: Use descriptive names (e.g., "gpt-4-turbo", "claude-3-opus")
- Keep names lowercase with hyphens (e.g., "my-service-name")

See [Data Structure Documentation](https://unitysvc-services.readthedocs.io/en/latest/data-structure/) for complete details.

### 4. Validate Your Data

Before committing changes:

```bash
# Validate all files
usvc data validate

# Format files (optional)
usvc data format
```

## Repository Structure

```
data/
└── ${provider_name}/                # Provider directory
    ├── provider.toml                # Provider metadata
    ├── README.md                    # Provider documentation
    ├── docs/                        # Shared code examples and descriptions
    │   ├── code-example.py.j2
    │   ├── code-example.js.j2
    │   ├── code-example.sh.j2
    │   └── description.md
    └── services/                    # Service definitions
        └── ${service_name}/
            ├── offering.json        # Service offering (technical specs)
            └── listing.json         # Service listing (user-facing info)
```

**Key points:**

- Each service directory contains one `offering.json` and one or more `listing*.json` files
- The provider directory name should match the `name` field in `provider.toml`
- Code examples can use Jinja2 templates (`.j2` extension) for dynamic content

See [Data Structure Documentation](https://unitysvc-services.readthedocs.io/en/latest/data-structure/) for complete details.

## GitHub Actions Workflows

This template includes automated workflows:

### 1. Validate Data Workflow

**File**: `.github/workflows/validate-data.yml`
**Triggers**: Every pull request and push to main

Validates all data files for schema compliance, file references, and directory consistency.

### 2. Format Check Workflow

**File**: `.github/workflows/format-check.yml`
**Triggers**: Every pull request and push to main

Checks JSON and TOML formatting. Run `usvc data format` locally to fix issues.

### 3. Upload Data Workflow

**File**: `.github/workflows/upload-data.yml`
**Triggers**: PR merged to main

Uploads services to the UnitySVC backend. Services are uploaded atomically (provider + offering + listing together).

**Setup Required**: Configure GitHub secrets (see below).

### 4. Populate Services Workflow (Optional)

**File**: `.github/workflows/populate-services.yml`
**Triggers**: Daily at 2 AM UTC / Manual

For dynamic service catalogs, automatically fetches data from upstream provider APIs and creates PRs with updates.

**To use this workflow:**

1. Add a `services_populator` section to your `provider.toml`:

```toml
[services_populator]
command = ['scripts/update_services.py']
requirements = ['requests']

[services_populator.envs]
# Non-sensitive config goes here
API_BASE_URL = "https://api.provider.com/v1"
```

2. Create a populate script that generates service files
3. Add provider API key as a GitHub secret (e.g., `PROVIDER_API_KEY`)
4. Update the workflow to pass the secret as an environment variable

See [Automated Workflow Documentation](https://unitysvc-services.readthedocs.io/en/latest/workflows/#automated-workflow) for details.

## GitHub Secrets Configuration

Configure the following secrets in your repository settings (**Settings** -> **Secrets and variables** -> **Actions**):

### Required Secrets

#### `SERVICE_BASE_URL`

- **Description**: The UnitySVC backend API URL
- **Example**: `https://api.unitysvc.com/v1`

#### `UNITYSVC_API_KEY`

- **Description**: API key for authenticating with the UnitySVC backend
- **How to obtain**:
  1. Log in to the UnitySVC platform
  2. Navigate to **Settings** -> **API Keys**
  3. Generate a new API key
  4. Copy the key (you won't be able to see it again)

### Optional Secrets (for Populate Workflow)

#### `${PROVIDER}_API_KEY`

- **Description**: API key for your upstream provider (e.g., `FIREWORKS_API_KEY`, `OPENAI_API_KEY`)
- **Used by**: Populate Services workflow to fetch service data from provider APIs

## Development Workflow

### Common Commands

```bash
# Validate data locally
usvc data validate

# Format data files
usvc data format

# List local services
usvc data list

# Populate services (if configured)
usvc data populate

# Upload manually (usually done via CI/CD)
export UNITYSVC_API_URL="https://api.unitysvc.com/v1"
export UNITYSVC_API_KEY="your-api-key"
usvc services upload
```

### Service Lifecycle

After uploading, manage service status:

```bash
# List uploaded services
usvc services list

# Submit draft service for review
usvc services submit <service-id>

# Deprecate an active service
usvc services deprecate <service-id>

# Withdraw a submitted service
usvc services withdraw <service-id>
```

### Pre-commit Hooks (Recommended)

```bash
pip install pre-commit
pre-commit install
```

## Contributing

1. Create a new branch for your changes
2. Make changes to files in the `data/` directory
3. Run `usvc data validate` to check your changes
4. Commit and push (pre-commit hooks run automatically)
5. Create a pull request
6. Once merged, data is automatically uploaded

## Documentation

- **[Getting Started](https://unitysvc-services.readthedocs.io/en/latest/getting-started/)** - Installation and first steps
- **[Data Structure](https://unitysvc-services.readthedocs.io/en/latest/data-structure/)** - File organization rules
- **[Workflows](https://unitysvc-services.readthedocs.io/en/latest/workflows/)** - Manual and automated patterns
- **[CLI Reference](https://unitysvc-services.readthedocs.io/en/latest/cli-reference/)** - All commands and options
- **[File Schemas](https://unitysvc-services.readthedocs.io/en/latest/file-schemas/)** - Schema specifications

## Support

- **UnitySVC Services SDK**: https://github.com/unitysvc/unitysvc-services
- **Documentation**: https://unitysvc-services.readthedocs.io
- **Issues**: Open an issue in this repository

## License

This template is provided under the MIT License. Service data you add is subject to your own licensing terms.
