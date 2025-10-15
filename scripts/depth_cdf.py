import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def read_bed_depths(bed_file, depth_column=3):
    """Read depths from BED file (0-indexed column)."""
    depths = []

    with open(bed_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")

            if len(parts) <= depth_column:
                print(
                    f"Warning: Line {line_num} has only {len(parts)} columns, skipping",
                    file=sys.stderr,
                )
                continue

            try:
                depth = int(parts[depth_column])
                depths.append(depth)
            except ValueError:
                print(
                    f"Warning: Line {line_num} has invalid depth value '{parts[depth_column]}', skipping",
                    file=sys.stderr,
                )
                continue

    return depths


def create_cdf(depths):
    """Create CDF from depth values."""
    if not depths:
        raise ValueError("No depth values provided")

    # Count frequencies
    counts = Counter(depths)
    total = len(depths)

    # Sort by depth value
    sorted_depths = sorted(counts.items())

    # Build CDF
    cdf = []
    cumsum = 0.0

    for depth, count in sorted_depths:
        cumsum += count / total
        cdf.append([depth, cumsum])

    # Calculate statistics
    sorted_vals = sorted(depths)
    stats = {
        "mean": sum(depths) / len(depths),
        "median": sorted_vals[len(sorted_vals) // 2],
        "min": min(depths),
        "max": max(depths),
        "total_sites": len(depths),
        "unique_depths": len(counts),
    }

    return cdf, stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert BED file to CDF JSON format for depth distribution"
    )
    parser.add_argument(
        "input", type=Path, help="Input BED file with depth in one column"
    )
    parser.add_argument("output", type=Path, help="Output JSON file (.json)")
    parser.add_argument(
        "-c",
        "--depth-column",
        type=int,
        default=3,
        help="0-indexed column containing depth values (default: 3)",
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print JSON output"
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Reading depths from {args.input}...")
    print(f"Using column {args.depth_column} for depth values")

    depths = read_bed_depths(args.input, args.depth_column)

    if not depths:
        print("Error: No valid depth values found", file=sys.stderr)
        sys.exit(1)

    print(f"Read {len(depths)} depth values")

    print("Creating CDF...")
    cdf, stats = create_cdf(depths)

    # Print statistics
    print("\nDepth Distribution Statistics:")
    print(f"  Total sites: {stats['total_sites']}")
    print(f"  Mean depth: {stats['mean']:.2f}")
    print(f"  Median depth: {stats['median']}")
    print(f"  Min depth: {stats['min']}")
    print(f"  Max depth: {stats['max']}")
    print(f"  Unique depth values: {stats['unique_depths']}")

    # Create output structure
    output_data = {"cdf": cdf, "stats": stats}

    # Write JSON
    print(f"\nWriting to {args.output}...")
    with open(args.output, "w") as f:
        if args.pretty:
            json.dump(output_data, f, indent=2)
        else:
            json.dump(output_data, f)

    print("✓ Done!")


if __name__ == "__main__":
    main()
