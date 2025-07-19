# Documentation Deployment Guide

This guide explains how the WarpConvNet documentation is automatically deployed to GitHub Pages.

## Automatic Deployment

The documentation is automatically built and deployed using GitHub Actions whenever you push to the `main` or `master` branch.

### How it Works

1. **GitHub Actions Workflow**: `.github/workflows/docs.yml`
   - Triggers on pushes to main/master branch
   - Installs Python dependencies
   - Builds documentation using MkDocs
   - Deploys to GitHub Pages

2. **Configuration**: `mkdocs-readthedocs.yml`
   - Uses ReadTheDocs theme
   - Configures site metadata
   - Sets up navigation structure

3. **Dependencies**: `docs/requirements.txt`
   - Lists all required Python packages
   - Ensures consistent builds

## Manual Deployment

If you need to deploy manually:

```bash
# Install dependencies
uv pip install -r docs/requirements.txt

# Build documentation
mkdocs build -f mkdocs-readthedocs.yml

# The site/ directory contains the built documentation
```

## Local Development

For local development and testing:

```bash
# Serve documentation locally
mkdocs serve -f mkdocs-readthedocs.yml

# This will start a local server at http://127.0.0.1:8000
```

## GitHub Pages Settings

To enable GitHub Pages:

1. Go to your repository settings
2. Navigate to "Pages" section
3. Set source to "GitHub Actions"
4. The workflow will automatically deploy to `https://username.github.io/repository-name/`

## Troubleshooting

### Build Failures
- Check GitHub Actions logs for error details
- Ensure all dependencies are listed in `docs/requirements.txt`
- Verify MkDocs configuration syntax

### Missing Pages
- Check that all referenced markdown files exist
- Verify navigation structure in `mkdocs-readthedocs.yml`
- Ensure files are committed to the repository

### Theme Issues
- ReadTheDocs theme is automatically installed by the workflow
- Check theme configuration in `mkdocs-readthedocs.yml`
- Verify JavaScript and CSS assets are loading correctly 