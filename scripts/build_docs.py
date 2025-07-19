# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Build documentation for WarpConvNet."""

import os
import subprocess
import sys
from pathlib import Path


def build_docs():
    """Build documentation using MkDocs."""
    try:
        # Ensure we're in the project root
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)

        # Add project root to PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

        # Generate API documentation
        print("Generating API documentation...")
        subprocess.run([sys.executable, "scripts/generate_api_docs.py"], check=True, env=env)

        # Generate diagrams
        print("Generating diagrams...")
        subprocess.run([sys.executable, "scripts/generate_diagrams.py"], check=True, env=env)

        # Build the documentation
        print("Building documentation...")
        result = subprocess.run(
            ["mkdocs", "build", "--clean"], check=True, capture_output=True, text=True, env=env
        )
        print(result.stdout)

        if result.stderr:
            print("Warnings:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)

        print("\nDocumentation built successfully!")
        return 0

    except subprocess.CalledProcessError as e:
        print("Error building documentation:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(build_docs())
