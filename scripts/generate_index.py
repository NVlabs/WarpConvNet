#!/usr/bin/env python3
"""Generate a PEP 503 simple repository index from a directory of wheel files.

Usage:
    python scripts/generate_index.py <wheel_dir> <output_dir>

Creates:
    <output_dir>/warpconvnet/index.html   - package index with links to all wheels
    <output_dir>/warpconvnet/<wheel>.whl   - copies of the wheel files
"""

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


def generate_index(wheel_dir: Path, output_dir: Path) -> None:
    pkg_dir = output_dir / "warpconvnet"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        print(f"No .whl files found in {wheel_dir}", file=sys.stderr)
        sys.exit(1)

    links = []
    for whl in wheels:
        dest = pkg_dir / whl.name
        shutil.copy2(whl, dest)
        digest = sha256(dest)
        links.append(f'    <a href="{whl.name}#sha256={digest}">{whl.name}</a>')
        print(f"  {whl.name}")

    index_html = "<!DOCTYPE html>\n" "<html><body>\n" + "\n".join(links) + "\n</body></html>\n"

    (pkg_dir / "index.html").write_text(index_html)

    # Root index pointing to the package
    root_index = (
        "<!DOCTYPE html>\n"
        "<html><body>\n"
        '    <a href="warpconvnet/">warpconvnet</a>\n'
        "</body></html>\n"
    )
    (output_dir / "index.html").write_text(root_index)

    print(f"\nGenerated index with {len(wheels)} wheels at {pkg_dir / 'index.html'}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <wheel_dir> <output_dir>", file=sys.stderr)
        sys.exit(1)

    generate_index(Path(sys.argv[1]), Path(sys.argv[2]))
