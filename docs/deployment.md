# Documentation Deployment

This page is for maintainers who need to build or deploy the documentation site.

## Automatic deployment

Documentation is built and deployed to GitHub Pages via
`.github/workflows/docs.yml` on every push to `main`.

## Local preview

```bash
pip install -r docs/requirements.txt
mkdocs serve
# Open http://127.0.0.1:8000
```

## Manual build

```bash
mkdocs build
# Output in site/
```
