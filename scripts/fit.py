#!/usr/bin/env python3
"""
Predict bottleneck size (Nb) from empirical allele frequency data.
"""

import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path


def compute_af_spectrum_empirical(df, bins):
    """
    Compute AF spectrum from empirical shared mutations.
    Returns feature vector matching training format.
    """
    if len(df) == 0:
        n_bins = len(bins) - 1
        return np.zeros(2 * n_bins)

    # Calculate allele frequencies
    circles_af = df["circles_alt"] / df["circles_depth"]
    duplex_af = df["duplex_alt"] / df["duplex_depth"]

    # Bin the allele frequencies
    circles_counts, _ = np.histogram(circles_af, bins=bins)
    duplex_counts, _ = np.histogram(duplex_af, bins=bins)

    # Concatenate into feature vector
    features = np.concatenate([circles_counts, duplex_counts])

    return features


def predict_nb_with_confidence(model, X, n_bootstrap=100):
    """
    Predict Nb with confidence interval using Random Forest tree predictions.

    Random Forest gives us a distribution of predictions from each tree,
    which we can use to estimate uncertainty.
    """
    # Get predictions from all trees
    tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])

    # Mean prediction
    mean_pred = tree_predictions.mean(axis=0)

    # Confidence intervals from tree predictions
    lower_95 = np.percentile(tree_predictions, 2.5, axis=0)
    upper_95 = np.percentile(tree_predictions, 97.5, axis=0)
    lower_68 = np.percentile(tree_predictions, 16, axis=0)
    upper_68 = np.percentile(tree_predictions, 84, axis=0)
    print(f"Median: {np.median(tree_predictions):.1f}")
    print(
        f"IQR: [{np.percentile(tree_predictions, 25):.1f}, {np.percentile(tree_predictions, 75):.1f}]"
    )
    # Standard deviation across trees
    std_pred = tree_predictions.std(axis=0)

    return {
        "mean": mean_pred,
        "std": std_pred,
        "lower_95": lower_95,
        "upper_95": upper_95,
        "lower_68": lower_68,
        "upper_68": upper_68,
    }


def plot_empirical_vs_test(
    predictions_csv, empirical_pred, empirical_name, output_path
):
    """
    Plot test set predictions with empirical prediction overlaid.
    Uses pre-computed test predictions from training.
    """
    # Load test set predictions
    test_df = pd.read_csv(predictions_csv)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Get empirical prediction values
    emp_mean = empirical_pred["mean"][0]
    emp_lower_95 = empirical_pred["lower_95"][0]
    emp_upper_95 = empirical_pred["upper_95"][0]
    emp_lower_68 = empirical_pred["lower_68"][0]
    emp_upper_68 = empirical_pred["upper_68"][0]

    # Plot test set as scatter
    ax.scatter(
        test_df["y_true"],
        test_df["y_pred"],
        alpha=0.3,
        s=30,
        color="steelblue",
        edgecolors="none",
        label="Test set",
        zorder=1,
    )

    # Determine plot range
    min_val = min(test_df["y_true"].min(), test_df["y_pred"].min(), emp_lower_95)
    max_val = max(test_df["y_true"].max(), test_df["y_pred"].max(), emp_upper_95)

    # Add perfect prediction line
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        lw=1.5,
        alpha=0.4,
        label="Perfect prediction",
        zorder=2,
    )

    # Plot 95% CI as shaded rectangle (in log space, use geometric mean for height)
    rect_height_95 = emp_mean * 1.5 / (emp_mean * 0.5)  # Proportional height
    rect_95 = plt.Rectangle(
        (emp_lower_95, emp_mean * 0.5),
        emp_upper_95 - emp_lower_95,
        emp_mean * 1.5 - emp_mean * 0.5,
        facecolor="crimson",
        alpha=0.15,
        edgecolor="none",
        zorder=3,
    )
    ax.add_patch(rect_95)

    # Plot 68% CI as shaded rectangle
    rect_68 = plt.Rectangle(
        (emp_lower_68, emp_mean * 0.7),
        emp_upper_68 - emp_lower_68,
        emp_mean * 1.3 - emp_mean * 0.7,
        facecolor="crimson",
        alpha=0.25,
        edgecolor="none",
        zorder=4,
    )
    ax.add_patch(rect_68)

    # Plot error bars
    # 95% CI
    ax.plot(
        [emp_lower_95, emp_upper_95],
        [emp_mean, emp_mean],
        color="crimson",
        linewidth=3,
        alpha=0.6,
        solid_capstyle="round",
        zorder=5,
    )
    ax.plot(
        [emp_lower_95, emp_lower_95],
        [emp_mean * 0.8, emp_mean * 1.2],
        color="crimson",
        linewidth=3,
        alpha=0.6,
        zorder=5,
    )
    ax.plot(
        [emp_upper_95, emp_upper_95],
        [emp_mean * 0.8, emp_mean * 1.2],
        color="crimson",
        linewidth=3,
        alpha=0.6,
        zorder=5,
    )

    # 68% CI (thicker)
    ax.plot(
        [emp_lower_68, emp_upper_68],
        [emp_mean, emp_mean],
        color="darkred",
        linewidth=6,
        alpha=0.8,
        solid_capstyle="round",
        zorder=6,
    )

    # Center point
    ax.scatter(
        [emp_mean],
        [emp_mean],
        s=200,
        color="darkred",
        marker="o",
        edgecolors="white",
        linewidths=2,
        label=empirical_name,
        zorder=7,
    )

    # Formatting
    ax.set_xlabel("True Nb / Predicted Nb ", fontsize=13, fontweight="bold")
    ax.set_ylabel("Predicted Nb", fontsize=13, fontweight="bold")

    # LOG SCALE
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Set limits with some margin in log space
    ax.set_xlim(min_val * 0.8, max_val * 1.2)
    ax.set_ylim(min_val * 0.8, max_val * 1.2)

    # Grid
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.8, which="both")

    # Legend in lower right
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="steelblue",
            markersize=8,
            alpha=0.5,
            label="Test set",
            linestyle="none",
        ),
        plt.Line2D(
            [0],
            [0],
            color="k",
            linestyle="--",
            linewidth=1.5,
            alpha=0.4,
            label="Perfect prediction",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="darkred",
            markeredgecolor="white",
            markersize=12,
            markeredgewidth=2,
            label=f"{empirical_name}\nNb = {emp_mean:.1f} [{emp_lower_95:.1f}, {emp_upper_95:.1f}]",
            linestyle="none",
        ),
        plt.Line2D([0], [0], color="darkred", linewidth=6, alpha=0.8, label="68% CI"),
        plt.Line2D([0], [0], color="crimson", linewidth=3, alpha=0.6, label="95% CI"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=10,
        framealpha=0.95,
        edgecolor="gray",
    )

    # Improve tick labels
    ax.tick_params(axis="both", which="major", labelsize=11)

    # Make square aspect ratio
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Predict Nb from empirical shared mutation data"
    )
    parser.add_argument(
        "empirical_csv",
        type=Path,
        help="CSV file with empirical shared mutations (columns: run,circles_depth,circles_alt,duplex_depth,duplex_alt)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default="model_output/model.joblib",
        help="Path to trained model (default: model_output/model.joblib)",
    )
    parser.add_argument(
        "--bins",
        type=Path,
        default="model_output/af_bins.npy",
        help="Path to AF bins file (default: model_output/af_bins.npy)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="Optional: save predictions to CSV"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Create plot comparing to test set"
    )
    parser.add_argument(
        "--test-predictions",
        type=Path,
        default="model_output/test_predictions.csv",
        help="CSV with test set predictions (default: model_output/test_predictions.csv)",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default="empirical_vs_test.png",
        help="Output filename for plot (default: empirical_vs_test.png)",
    )
    parser.add_argument(
        "--empirical-name",
        type=str,
        default="Empirical data",
        help="Label for empirical data point in plot",
    )

    args = parser.parse_args()

    # Load model and bins
    print(f"Loading model from {args.model}...")
    model = joblib.load(args.model)
    bins = np.load(args.bins)
    print(f"  Model loaded: {model.n_estimators} trees")
    print(f"  AF bins: {len(bins) - 1} bins")

    # Load empirical data
    print(f"\nLoading empirical data from {args.empirical_csv}...")
    df = pd.read_csv(args.empirical_csv)
    print(f"  Loaded {len(df)} shared mutations")

    # Check required columns
    required_cols = ["circles_depth", "circles_alt", "duplex_depth", "duplex_alt"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Compute AF spectrum
    print("\nComputing allele frequency spectrum...")
    X = compute_af_spectrum_empirical(df, bins).reshape(1, -1)

    print(f"  Feature vector shape: {X.shape}")
    print(f"  Total mutations in spectrum: {X.sum():.0f}")

    # Make prediction with confidence intervals
    print("\nPredicting Nb with confidence intervals...")
    predictions = predict_nb_with_confidence(model, X)

    nb_mean = predictions["mean"][0]
    nb_std = predictions["std"][0]
    nb_lower_95 = predictions["lower_95"][0]
    nb_upper_95 = predictions["upper_95"][0]
    nb_lower_68 = predictions["lower_68"][0]
    nb_upper_68 = predictions["upper_68"][0]

    # Display results
    print("\n" + "=" * 60)
    print("BOTTLENECK SIZE PREDICTION")
    print("=" * 60)
    print(f"  Point estimate:    Nb = {nb_mean:.1f}")
    print(f"  Std deviation:     ± {nb_std:.1f}")
    print(f"  68% CI:            [{nb_lower_68:.1f}, {nb_upper_68:.1f}]")
    print(f"  95% CI:            [{nb_lower_95:.1f}, {nb_upper_95:.1f}]")
    print("=" * 60)

    # Save to file if requested
    if args.output:
        results_df = pd.DataFrame(
            [
                {
                    "nb_mean": nb_mean,
                    "nb_std": nb_std,
                    "nb_lower_95": nb_lower_95,
                    "nb_upper_95": nb_upper_95,
                    "nb_lower_68": nb_lower_68,
                    "nb_upper_68": nb_upper_68,
                    "n_shared_mutations": len(df),
                }
            ]
        )
        results_df.to_csv(args.output, index=False)
        print(f"\n✓ Saved predictions to {args.output}")

    # Create plot if requested
    if args.plot:
        print(f"\nCreating plot vs test set...")
        if not args.test_predictions.exists():
            print(f"ERROR: Test predictions file not found: {args.test_predictions}")
            print("You need to save test predictions during training.")
            print("See instructions below.")
        else:
            plot_empirical_vs_test(
                args.test_predictions,
                predictions,
                args.empirical_name,
                args.plot_output,
            )


if __name__ == "__main__":
    main()
