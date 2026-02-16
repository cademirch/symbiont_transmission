#!/usr/bin/env python3
"""
Aggregate shared mutation CSVs from multiple parameter sets.
Creates a single consolidated CSV with parameter metadata joined.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import sys


def load_parameter_sets(params_file):
    """Load parameter sets from JSON file"""
    with open(params_file, 'r') as f:
        data = json.load(f)
    
    # Create a dictionary mapping param_set ID to full parameter info
    param_dict = {p["param_set"]: p for p in data["parameter_sets"]}
    print(f"Loaded {len(param_dict)} parameter sets from {params_file}")
    
    return param_dict


def aggregate_shared_mutations(results_dir, param_dict, output_file):
    """Aggregate all shared mutation CSVs into one file with parameter metadata"""
    results_dir = Path(results_dir)
    
    all_data = []
    missing_files = []
    
    print(f"\nSearching for shared mutation files in {results_dir}...")
    
    for param_set, params in param_dict.items():
        # Look for the shared CSV file
        shared_file = results_dir / f"paramset_{param_set}_shared.csv"
        
        if not shared_file.exists():
            missing_files.append(param_set)
            continue
        
        # Read the CSV
        try:
            df = pd.read_csv(shared_file)
            
            if df.empty:
                print(f"  Warning: {shared_file.name} is empty")
                df = pd.DataFrame(
                    [
                        {
                            "run": 0,
                            "circles_depth": 1,
                            "circles_alt": 0,
                            "duplex_depth": 1,
                            "duplex_alt": 0,
                        }
                    ]
                )
            
            # Add all parameter metadata as columns
            for key, value in params.items():
                df[key] = value
            
            all_data.append(df)
            
        except Exception as e:
            print(f"  Error reading {shared_file.name}: {e}")
            continue
    
    if missing_files:
        print(f"\nWarning: {len(missing_files)} parameter sets have no output files:")
        print(f"  Missing param_sets: {sorted(missing_files)[:10]}...")
        if len(missing_files) > 10:
            print(f"  (and {len(missing_files) - 10} more)")
    
    if not all_data:
        print("\nError: No data was successfully loaded")
        sys.exit(1)
    
    # Concatenate all dataframes
    print(f"\nConcatenating {len(all_data)} files...")
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns: parameter metadata first, then mutation data
    param_cols = ['param_set', 'N', 'u_per_site', 'u_total', 'genome_size', 'Nb', 'Gs']
    mutation_cols = ['run', 'circles_depth', 'circles_alt', 'duplex_depth', 'duplex_alt']
    
    # Only include columns that exist
    param_cols = [c for c in param_cols if c in final_df.columns]
    mutation_cols = [c for c in mutation_cols if c in final_df.columns]
    
    final_df = final_df[param_cols + mutation_cols]
    
    # Save to output file
    final_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Success!")
    print(f"  Total parameter sets processed: {len(all_data)}")
    print(f"  Total shared mutations: {len(final_df):,}")
    print(f"  Output saved to: {output_file}")
    print(f"\nColumns: {', '.join(final_df.columns)}")
    
    # Print summary statistics
    print(f"\nSummary statistics:")
    print(f"  Mutations per parameter set (mean): {len(final_df) / len(all_data):.1f}")
    print(f"  Nb range: [{final_df['Nb'].min()}, {final_df['Nb'].max()}]")
    print(f"  N range: [{final_df['N'].min():.2e}, {final_df['N'].max():.2e}]")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate shared mutation CSVs from simulation results"
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing paramset_*_shared.csv files"
    )
    parser.add_argument(
        "-p", "--params-file",
        type=Path,
        default="parameter_sets.json",
        help="Parameter sets JSON file (default: parameter_sets.json)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default="aggregated_shared_mutations.csv",
        help="Output CSV file (default: aggregated_shared_mutations.csv)"
    )
    
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"Error: Results directory {args.results_dir} does not exist")
        sys.exit(1)
    
    if not args.params_file.exists():
        print(f"Error: Parameter file {args.params_file} does not exist")
        sys.exit(1)
    
    # Load parameter sets
    param_dict = load_parameter_sets(args.params_file)
    
    # Aggregate shared mutations
    aggregate_shared_mutations(args.results_dir, param_dict, args.output)


if __name__ == "__main__":
    main()