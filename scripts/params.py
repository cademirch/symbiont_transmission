#!/usr/bin/env python3
import json

import argparse
from pathlib import Path
from scipy.stats.qmc import Sobol, scale


def generate_sobol_params(
    n_samples=625, genome_size=1.2e6, seed=42, skip=0, start_id=0
):
    """Generate parameter sets using Sobol sampling with log-uniform distributions"""

    # Define bounds in LOG space for log-uniform parameters, linear for others
    # Parameters: [log10(N), log10(u), log10(Nb), Gs]
    l_bounds = [8, -9, 0, 100]  # [log10(1e8), log10(1e-9), log10(1), 100]
    u_bounds = [10, -8, 3, 10000]  # [log10(1e10), log10(1e-8), log10(1000), 10000]

    # Generate Sobol sequence
    sampler = Sobol(d=4, scramble=True, rng=seed)

    # Skip previously generated samples if needed
    if skip > 0:
        sampler.fast_forward(skip)

    # Generate samples in [0, 1]^4
    samples = sampler.random(n_samples)

    # Scale to parameter bounds (in log space for log-uniform params)
    scaled_samples = scale(samples, l_bounds, u_bounds)

    # Convert to parameter sets
    param_sets = []

    for i, sample in enumerate(scaled_samples):
        # Exponentiate log-scaled parameters to get actual values
        N = int(10 ** sample[0])  # log-uniform: 10^8 to 10^10
        u_per_site = float(10 ** sample[1])  # log-uniform: 10^-9 to 10^-8
        Nb = int(10 ** sample[2])  # log-uniform: 1 to 1000
        Gs = int(sample[3])  # linear: 100 to 10000

        param_sets.append(
            {
                "param_set": start_id + i,
                "N": N,
                "u_per_site": u_per_site,
                "u_total": u_per_site * genome_size,
                "genome_size": int(genome_size),
                "Nb": Nb,
                "Gs": Gs,
            }
        )

    return param_sets


def main():
    parser = argparse.ArgumentParser(
        description="Generate parameter sets using Sobol sampling"
    )
    parser.add_argument(
        "-n",
        "--n-samples",
        type=int,
        default=625,
        help="Number of parameter sets to generate (default: 625)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="parameter_sets.json",
        help="Output JSON file (default: parameter_sets.json)",
    )
    parser.add_argument(
        "-g",
        "--genome-size",
        type=float,
        default=1.2e6,
        help="Genome size in bp (default: 1.2e6)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Sobol sequence (default: 42)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing parameter file instead of overwriting",
    )

    args = parser.parse_args()

    # Check for existing file if appending
    existing_params = []
    skip = 0
    start_id = 0

    if args.append and Path(args.output).exists():
        print(f"Appending to existing file: {args.output}")
        with open(args.output) as f:
            existing_data = json.load(f)
        existing_params = existing_data["parameter_sets"]

        # Figure out how many samples to skip from the last param_set index
        if existing_params:
            last_id = existing_params[-1]["param_set"]
            skip = last_id + 1
            start_id = skip

        print(f"  Existing parameters: {len(existing_params)}")
        print(f"  Starting new IDs from: {start_id}")
        print(f"  Skipping first {skip} Sobol samples")

    print(f"Generating {args.n_samples} new parameter sets using Sobol sampling...")

    new_params = generate_sobol_params(
        n_samples=args.n_samples,
        genome_size=args.genome_size,
        seed=args.seed,
        skip=skip,
        start_id=start_id,
    )

    all_params = existing_params + new_params

    output_data = {
        "n_param_sets": len(all_params),
        "genome_size": int(args.genome_size),
        "sampling_method": "sobol",
        "sobol_seed": args.seed,
        "parameter_sets": all_params,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Wrote {len(all_params)} total parameter sets to {args.output}")
    print(f"  ({len(new_params)} new parameters added)")
    print(f"\nParameter ranges:")
    print(f"  N (population): 10^8 to 10^10 (log-uniform)")
    print(f"  u (mutation rate): 10^-9 to 10^-8 per site (log-uniform)")
    print(f"  Nb (bottleneck): 1 to 1000 (log-uniform)")
    print(f"  Gs (generations): 100 to 10000 (linear)")
    print(f"\nSampling method: Sobol quasi-random sequence")


if __name__ == "__main__":
    main()
