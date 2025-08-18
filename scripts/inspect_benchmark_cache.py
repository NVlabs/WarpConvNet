#!/usr/bin/env python3
"""
Utility script to inspect and pretty print the stored benchmark cache.

New generic cache workflow:
  - By default, show a tree-like summary of all namespaces and their entry counts
  - To view details for a specific namespace, pass: namespace=<name>

Usage:
    python inspect_benchmark_cache.py                          # List namespaces (tree summary)
    python inspect_benchmark_cache.py namespace=<name>         # Show details for a namespace
    python inspect_benchmark_cache.py namespace=<name> --top-k K   # Show top K per-config results
    python inspect_benchmark_cache.py namespace=<name> --best-only  # Show only best per-config

Examples:
    python inspect_benchmark_cache.py
    python inspect_benchmark_cache.py namespace=sparse_conv_forward
    python inspect_benchmark_cache.py namespace=sparse_conv_forward_implicit --best-only
    python inspect_benchmark_cache.py namespace=sparse_conv_backward --top-k 3

The script loads benchmark results from the generic cache
(~/.cache/warpconvnet/benchmark_cache_generic.pkl) and formats them
for human-readable inspection.
"""

import sys
from datetime import datetime
from typing import Dict, Any

from warpconvnet.utils.benchmark_cache import (
    get_generic_benchmark_cache,
    load_generic_benchmark_cache,
    generic_benchmark_get_namespace,
)


def format_value(value: Any, indent: int = 0, top_k: int = None) -> str:
    """Format a value for pretty printing with proper indentation."""
    spaces = "  " * indent

    if isinstance(value, dict):
        if not value:
            return "{}"

        lines = ["{"]
        for k, v in value.items():
            formatted_value = format_value(v, indent + 1, top_k)
            lines.append(f"{spaces}  {k}: {formatted_value}")
        lines.append(f"{spaces}}}")
        return "\n".join(lines)

    elif isinstance(value, (list, tuple)):
        if not value:
            return "[]"

        # If top_k is specified and this looks like benchmark results, show only the top K results
        # Case 1: legacy tuple/list results
        if (
            top_k is not None
            and len(value) > 0
            and isinstance(value[0], (list, tuple))
            and len(value[0]) >= 3
        ):
            top_results = value[:top_k]
            if len(top_results) == 1:
                formatted_item = format_value(top_results[0], indent + 1, top_k)
                return f"[\n{spaces}  {formatted_item}\n{spaces}]"
            else:
                lines = ["["]
                for result in top_results:
                    formatted_item = format_value(result, indent + 1, top_k)
                    lines.append(f"{spaces}  {formatted_item}")
                lines.append(f"{spaces}]")
                return "\n".join(lines)

        # Case 2: new generic cache format: list of dicts with 'params'/'metric'
        if (
            top_k is not None
            and len(value) > 0
            and isinstance(value[0], dict)
            and ("params" in value[0] and "metric" in value[0])
        ):
            top_results = value[:top_k]
            lines = ["["]
            for result in top_results:
                formatted_item = format_value(result, indent + 1, top_k)
                lines.append(f"{spaces}  {formatted_item}")
            lines.append(f"{spaces}]")
            return "\n".join(lines)

        # Keep short, simple lists on one line. Allow primitives and empty dicts.
        if len(value) <= 3 and all(
            isinstance(x, (int, float, str)) or (isinstance(x, dict) and not x) for x in value
        ):
            inner = ", ".join(format_value(x, indent, top_k) for x in value)
            return f"[{inner}]"

        lines = ["["]
        for item in value:
            formatted_item = format_value(item, indent + 1, top_k)
            lines.append(f"{spaces}  {formatted_item}")
        lines.append(f"{spaces}]")
        return "\n".join(lines)

    elif isinstance(value, str):
        return f'"{value}"'

    elif isinstance(value, float):
        # Format floats nicely
        if value < 0.001:
            return f"{value:.6f}"
        elif value < 1:
            return f"{value:.4f}"
        else:
            return f"{value:.3f}"

    else:
        return str(value)


def pretty_print_benchmark_results(results: Dict, title: str, top_k: int = None) -> None:
    """Pretty print benchmark results with clear formatting."""
    print(f"\n{'='*60}")
    if top_k == 1:
        print(f"{title.upper()} - BEST RESULTS ONLY")
    elif top_k is not None:
        print(f"{title.upper()} - TOP {top_k} RESULTS")
    else:
        print(f"{title.upper()}")
    print(f"{'='*60}")

    if not results:
        print("No cached results found.")
        return

    print(f"Total configurations: {len(results)}")
    if top_k == 1:
        print("(Showing only the best performing algorithm for each configuration)")
    elif top_k is not None:
        print(f"(Showing top {top_k} performing algorithms for each configuration)")
    print()

    # Sort configurations by in_channels (primary) and out_channels (secondary) when possible
    def get_sort_key(item):
        config_key, result = item
        config_str = str(config_key)

        # Default values if parsing fails
        in_channels = 999999
        out_channels = 999999

        if "SpatiallySparseConvConfig" in config_str:
            # Extract parameters from the config string
            try:
                # Remove the class name and parentheses
                params_str = config_str.replace("SpatiallySparseConvConfig(", "").replace(")", "")
                parts = params_str.split(", ")

                for part in parts:
                    if "=" in part:
                        key, value = part.split("=", 1)
                        key = key.strip()
                        value = value.strip()

                        if key == "in_channels":
                            in_channels = int(value)
                        elif key == "out_channels":
                            out_channels = int(value)
            except (ValueError, IndexError):
                # If parsing fails, use default values which will sort last
                pass

        return (in_channels, out_channels)

    # Sort the results
    sorted_results = sorted(results.items(), key=get_sort_key)

    for i, (config_key, result) in enumerate(sorted_results, 1):
        print(f"{'-'*40}")
        print(f"Configuration {i}:")
        print(f"{'-'*40}")

        # Format the configuration key more readably
        config_str = str(config_key)
        if "SpatiallySparseConvConfig" in config_str:
            # Extract key parameters for a more readable format
            print("Config Parameters:")
            parts = (
                config_str.replace("SpatiallySparseConvConfig(", "").replace(")", "").split(", ")
            )
            for part in parts:
                if "=" in part:
                    key, value = part.split("=", 1)
                    print(f"  {key.strip()}: {value.strip()}")
        else:
            print(f"Config Key: {config_str}")
        print()

        # Print the result
        if top_k == 1:
            print("Best Result:")
        elif top_k is not None:
            print(f"Top {top_k} Results:")
        else:
            print("Results:")
        formatted_result = format_value(result, 1, top_k)
        print(f"  {formatted_result}")
        print()


def print_cache_tree(namespaces: Dict[str, Dict[Any, Any]]) -> None:
    """Print a tree-like summary of namespaces and their entry counts."""
    print(f"\n{'='*60}")
    print("NAMESPACE TREE")
    print(f"{'='*60}")
    if not namespaces:
        print("No namespaces found.")
        return
    names = sorted(namespaces.keys())
    print(f"Total namespaces: {len(names)}\n")
    for name in names:
        try:
            count = len(namespaces.get(name, {}))
        except Exception:
            count = 0
        print(f"- {name}: {count} entry(ies)")


def load_and_inspect_cache(top_k: int = None, namespace: str | None = None) -> None:
    """Load and display the generic benchmark cache summary, and optional namespace details."""
    print("Loading benchmark cache...")

    # Get cache file info
    cache = get_generic_benchmark_cache()
    cache_file = cache.cache_file

    print(f"Cache file location: {cache_file}")

    if cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        size = cache_file.stat().st_size
        print(f"Cache file size: {size:,} bytes")
        print(f"Last modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("Cache file does not exist.")
        return

    try:
        namespaces = load_generic_benchmark_cache()
        print_cache_tree(namespaces)

        if namespace is None:
            print("\nTip: pass namespace=<name> to view details for a specific namespace.")
            print(f"{'='*60}")
            return

        if namespace not in namespaces:
            print(f"\nNamespace not found: '{namespace}'")
            print("Available namespaces:")
            for name in sorted(namespaces.keys()):
                print(f"  - {name}")
            print(f"{'='*60}")
            return

        # Show details for selected namespace
        results = generic_benchmark_get_namespace(namespace)
        pretty_print_benchmark_results(results, f"Namespace: {namespace}", top_k)
        print(f"\n{'='*60}")
        print("Inspection complete.")
        print(f"{'='*60}")
    except Exception as e:
        print(f"Error loading cache: {e}")


def search_cache(pattern: str, namespace: str | None = None) -> None:
    """Search namespaces or configurations matching a pattern."""
    print(f"\nSearching for: '{pattern}'")
    print(f"{'='*50}")

    if namespace is None:
        # Search namespace names
        namespaces = load_generic_benchmark_cache()
        matched = [name for name in namespaces.keys() if pattern.lower() in name.lower()]
        if matched:
            print(f"Matching namespaces ({len(matched)}):")
            for i, name in enumerate(sorted(matched), 1):
                print(f"  {i}. {name} ({len(namespaces.get(name, {}))} entry(ies))")
        else:
            print("No matching namespaces found.")
        return

    # Search within a specific namespace's keys
    ns_map = generic_benchmark_get_namespace(namespace)
    matches = [k for k in ns_map.keys() if pattern.lower() in str(k).lower()]
    if matches:
        print(f"Matches in '{namespace}' ({len(matches)}):")
        for i, k in enumerate(matches, 1):
            k_str = str(k)
            if len(k_str) > 100:
                k_str = k_str[:97] + "..."
            print(f"  {i}. {k_str}")
    else:
        print(f"No matches found in namespace '{namespace}'.")


if __name__ == "__main__":
    # Parse command line arguments
    top_k = None

    # Handle --best-only (equivalent to --top-k 1)
    best_only = "--best-only" in sys.argv
    if best_only:
        sys.argv.remove("--best-only")
        top_k = 1

    # Handle --top-k K
    if "--top-k" in sys.argv:
        try:
            idx = sys.argv.index("--top-k")
            if idx + 1 >= len(sys.argv):
                print("Error: --top-k requires an integer argument")
                sys.exit(1)

            top_k_value = int(sys.argv[idx + 1])
            if top_k_value <= 0:
                print("Error: --top-k argument must be a positive integer")
                sys.exit(1)

            if top_k is not None:
                print("Error: Cannot specify both --best-only and --top-k")
                sys.exit(1)

            top_k = top_k_value
            # Remove both --top-k and its argument
            sys.argv.pop(idx)  # Remove --top-k
            sys.argv.pop(idx)  # Remove the argument (now at the same index)
        except (ValueError, IndexError):
            print("Error: --top-k requires a valid integer argument")
            sys.exit(1)

    # Extract optional namespace=<name>
    namespace = None
    for arg in list(sys.argv[1:]):
        if arg.startswith("namespace="):
            namespace = arg.split("=", 1)[1].strip()
            sys.argv.remove(arg)
            break

    # If there are leftover args after removing known flags, treat them as a search query
    leftover_args = sys.argv[1:]
    if leftover_args:
        # First, show tree and then search either namespaces or within provided namespace
        load_and_inspect_cache(top_k=None, namespace=None)
        pattern = " ".join(leftover_args)
        search_cache(pattern, namespace)
    else:
        # Show tree, and optionally the selected namespace details
        load_and_inspect_cache(top_k, namespace)
