#!/usr/bin/env python3
"""
Utility script to inspect and pretty print the stored benchmark cache.

Usage:
    python inspect_benchmark_cache.py                    # Full inspection of all cached benchmarks
    python inspect_benchmark_cache.py --best-only        # Show only the best algorithm for each config
    python inspect_benchmark_cache.py --top-k K          # Show top K fastest algorithms for each config
    python inspect_benchmark_cache.py --forward-only     # Show only forward pass results
    python inspect_benchmark_cache.py --backward-only    # Show only backward pass results
    python inspect_benchmark_cache.py <search_term>      # Search for configurations containing the term
    python inspect_benchmark_cache.py --best-only <term> # Search and show only best results

Examples:
    python inspect_benchmark_cache.py                           # Show all cached benchmarks
    python inspect_benchmark_cache.py --best-only               # Show only best performing algorithms
    python inspect_benchmark_cache.py --top-k 3                 # Show top 3 fastest algorithms for each config
    python inspect_benchmark_cache.py --forward-only            # Show only forward pass results
    python inspect_benchmark_cache.py --backward-only           # Show only backward pass results
    python inspect_benchmark_cache.py --best-only --forward-only # Show only best forward results
    python inspect_benchmark_cache.py --top-k 2 --forward-only  # Show top 2 fastest forward algorithms
    python inspect_benchmark_cache.py "float16"                 # Find configurations with float16
    python inspect_benchmark_cache.py --best-only "32"          # Find 32-channel configs, show only best
    python inspect_benchmark_cache.py --top-k 3 "EXPLICIT_GEMM" # Find EXPLICIT_GEMM, show top 3 results

The script loads benchmark results from ~/.cache/warpconvnet/benchmark_cache.pkl and formats them
for human-readable inspection. Each configuration shows:
- Configuration parameters (input/output sizes, channels, kernel volume, etc.)
- Algorithm performance results (execution times in milliseconds)
- Failed algorithms (shown as 'inf' for infinite time)
"""

import sys
from datetime import datetime
from typing import Dict, Any

from warpconvnet.utils.benchmark_cache import load_sparse_conv_benchmark_cache, get_benchmark_cache


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
        if (
            top_k is not None
            and len(value) > 0
            and isinstance(value[0], (list, tuple))
            and len(value[0]) >= 3
        ):
            # This looks like benchmark results - show only the top K results
            top_results = value[:top_k]
            if len(top_results) == 1:
                # Single result, format inline
                formatted_item = format_value(top_results[0], indent + 1, top_k)
                return f"[\n{spaces}  {formatted_item}\n{spaces}]"
            else:
                # Multiple results, format each
                lines = ["["]
                for i, result in enumerate(top_results):
                    formatted_item = format_value(result, indent + 1, top_k)
                    lines.append(f"{spaces}  {formatted_item}")
                lines.append(f"{spaces}]")
                return "\n".join(lines)

        if len(value) <= 3 and all(isinstance(x, (int, float, str)) for x in value):
            # Keep short lists on one line
            return str(value)

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

    # Sort configurations by in_channels (primary) and out_channels (secondary)
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
            print("Best Algorithm:")
        elif top_k is not None:
            print(f"Top {top_k} Algorithms:")
        else:
            print("Benchmark Results:")
        formatted_result = format_value(result, 1, top_k)
        print(f"  {formatted_result}")
        print()


def load_and_inspect_cache(
    top_k: int = None, show_forward: bool = True, show_backward: bool = True
) -> None:
    """Load and display the benchmark cache in a human-readable format."""
    print("Loading benchmark cache...")

    # Get cache file info
    cache = get_benchmark_cache()
    cache_file = cache.cache_file

    print(f"Cache file location: {cache_file}")

    if cache_file.exists():
        # Get file modification time
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        size = cache_file.stat().st_size
        print(f"Cache file size: {size:,} bytes")
        print(f"Last modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("Cache file does not exist.")
        return

    # Load the cache
    try:
        forward_results, backward_results = load_sparse_conv_benchmark_cache()

        # Print summary
        print("\nCache Summary:")
        if show_forward:
            print(f"  Forward configurations: {len(forward_results)}")
        if show_backward:
            print(f"  Backward configurations: {len(backward_results)}")

        # Pretty print forward results
        if show_forward:
            pretty_print_benchmark_results(
                forward_results, "🔄 Forward Pass Benchmark Results", top_k
            )

        # Pretty print backward results
        if show_backward:
            pretty_print_benchmark_results(
                backward_results, "⏪ Backward Pass Benchmark Results", top_k
            )

        print(f"\n{'='*60}")
        print("Inspection complete.")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Error loading cache: {e}")


def search_cache(
    pattern: str, top_k: int = None, show_forward: bool = True, show_backward: bool = True
) -> None:
    """Search for configurations matching a pattern."""
    forward_results, backward_results = load_sparse_conv_benchmark_cache()

    print(f"\nSearching for configurations containing: '{pattern}'")
    if top_k == 1:
        print("(Showing only best results)")
    elif top_k is not None:
        print(f"(Showing top {top_k} results)")
    print(f"{'='*50}")

    # Helper function to sort configurations
    def get_sort_key(config_key):
        config_str = str(config_key)

        # Default values if parsing fails
        in_channels = 999999
        out_channels = 999999

        if "SpatiallySparseConvConfig" in config_str:
            try:
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
                pass

        return (in_channels, out_channels)

    # Search forward results
    if show_forward:
        forward_matches = {
            k: v for k, v in forward_results.items() if pattern.lower() in str(k).lower()
        }
        if forward_matches:
            print(f"\n🔄 Forward Pass Matches ({len(forward_matches)}):")
            # Sort the matches
            sorted_forward_matches = sorted(forward_matches.keys(), key=get_sort_key)
            for i, config_key in enumerate(sorted_forward_matches, 1):
                config_str = str(config_key)
                if len(config_str) > 80:
                    # Truncate long config strings for search results
                    config_str = config_str[:77] + "..."
                print(f"  {i}. {config_str}")

    # Search backward results
    if show_backward:
        backward_matches = {
            k: v for k, v in backward_results.items() if pattern.lower() in str(k).lower()
        }
        if backward_matches:
            print(f"\n⏪ Backward Pass Matches ({len(backward_matches)}):")
            # Sort the matches
            sorted_backward_matches = sorted(backward_matches.keys(), key=get_sort_key)
            for i, config_key in enumerate(sorted_backward_matches, 1):
                config_str = str(config_key)
                if len(config_str) > 80:
                    # Truncate long config strings for search results
                    config_str = config_str[:77] + "..."
                print(f"  {i}. {config_str}")

    if show_forward and show_backward:
        if not forward_matches and not backward_matches:
            print("No matches found.")
    elif show_forward:
        if not forward_matches:
            print("No forward matches found.")
    elif show_backward:
        if not backward_matches:
            print("No backward matches found.")


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

    forward_only = "--forward-only" in sys.argv
    if forward_only:
        sys.argv.remove("--forward-only")

    backward_only = "--backward-only" in sys.argv
    if backward_only:
        sys.argv.remove("--backward-only")

    # Determine what to show
    show_forward = True
    show_backward = True

    if forward_only and backward_only:
        print("Error: Cannot specify both --forward-only and --backward-only")
        sys.exit(1)
    elif forward_only:
        show_backward = False
    elif backward_only:
        show_forward = False

    if len(sys.argv) > 1:
        # Search mode
        pattern = " ".join(sys.argv[1:])
        search_cache(pattern, top_k, show_forward, show_backward)
    else:
        # Full inspection mode
        load_and_inspect_cache(top_k, show_forward, show_backward)
