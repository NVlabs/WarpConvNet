#!/usr/bin/env python3
"""Generate a PEP 503 simple repository index from a directory of wheel files.

Usage:
    python scripts/generate_index.py <wheel_dir> <output_dir>
    python scripts/generate_index.py <wheel_dir> <output_dir> --base-url <url>

Without --base-url: copies wheels into output_dir and links to local files.
With --base-url: links point to <url>/<wheel_name> (e.g. GitHub Release assets).
                 No wheel files are copied — only HTML index is generated.
"""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_index(wheel_dir: Path, output_dir: Path, base_url: str = "") -> None:
    pkg_dir = output_dir / "warpconvnet"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        print(f"No .whl files found in {wheel_dir}", file=sys.stderr)
        sys.exit(1)

    links = []
    for whl in wheels:
        digest = sha256(whl)
        if base_url:
            # Link to external URL (e.g. GitHub Release asset)
            url = f"{base_url.rstrip('/')}/{whl.name}"
        else:
            # Copy wheel locally and link to it
            shutil.copy2(whl, pkg_dir / whl.name)
            url = whl.name
        links.append(f'    <a href="{url}#sha256={digest}">{whl.name}</a>')
        print(f"  {whl.name}")

    index_html = "<!DOCTYPE html>\n<html><body>\n" + "\n".join(links) + "\n</body></html>\n"
    (pkg_dir / "index.html").write_text(index_html)

    # Root index pointing to the package
    root_index = (
        "<!DOCTYPE html>\n"
        "<html><body>\n"
        '    <a href="warpconvnet/">warpconvnet</a>\n'
        "</body></html>\n"
    )
    (output_dir / "index.html").write_text(root_index)

    mode = f"linking to {base_url}" if base_url else "with local copies"
    print(f"\nGenerated index with {len(wheels)} wheels ({mode})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PEP 503 wheel index")
    parser.add_argument("wheel_dir", type=Path, help="Directory containing .whl files")
    parser.add_argument("output_dir", type=Path, help="Output directory for index")
    parser.add_argument(
        "--base-url",
        default="",
        help="Base URL for wheel links (e.g. GitHub Release download URL)",
    )
    args = parser.parse_args()
    generate_index(args.wheel_dir, args.output_dir, args.base_url)
