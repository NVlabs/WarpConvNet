#!/bin/bash

# Build WarpConvNet Documentation
# This script builds the documentation using MkDocs with ReadTheDocs theme

set -e

echo "🔨 Building WarpConvNet Documentation..."

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: Not in a virtual environment. Consider activating one."
fi

# Install documentation dependencies if not already installed
echo "📦 Installing documentation dependencies..."
uv pip install -r docs/requirements.txt

# Build the documentation
echo "🏗️  Building documentation with ReadTheDocs theme..."
mkdocs build -f mkdocs-readthedocs.yml

echo "✅ Documentation built successfully!"
echo "📁 Output directory: site/"
echo "🌐 To serve locally: mkdocs serve -f mkdocs-readthedocs.yml"
echo "🚀 To deploy: push to main branch (GitHub Actions will handle deployment)" 