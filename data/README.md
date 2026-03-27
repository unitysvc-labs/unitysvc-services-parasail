# Service Data Directory

This directory contains the service provider data for UnitySVC.

## Getting Started

### 1. Install the UnitySVC Services CLI

```bash
pip install unitysvc-services
```

### 2. Create Your Data

Replace the example data with your actual service information. The typical structure is:

```
data/
└── <provider-name>/
    ├── provider.toml                # Provider configuration
    ├── README.md                    # Provider documentation
    ├── docs/                        # Shared code examples
    │   ├── description.md
    │   └── code-example.*.j2
    └── services/
        └── <service-name>/
            ├── offering.json        # Service offering (technical specs)
            └── listing.json         # Service listing (user-facing)
```

### 3. Validate and Format

```bash
# Validate your data
usvc data validate

# Format files
usvc data format
```

### 4. Upload

Commit and push your changes. The GitHub workflow will automatically upload when PRs are merged.

Or upload manually:

```bash
export UNITYSVC_API_URL="https://api.unitysvc.com/v1"
export UNITYSVC_API_KEY="your-api-key"
usvc services upload
```

## Notes

- Keep your data files properly formatted and validated
- Follow the schema specifications for compatibility
- Use version control to track changes
- For multiple services, consider using `usvc data populate` with a custom script
- See the [documentation](https://unitysvc-services.readthedocs.io) for complete details
