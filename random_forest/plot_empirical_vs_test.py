#!/usr/bin/env python3
"""
Visualize empirical prediction against test set with confidence intervals.
"""

import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path


def compute_af_spectrum_empirical(df, bins):
    """Compute AF spectrum from empirical shared mutations."""
    if len(df) == 0:
        n_bins = len(bins) - 1
        return np.zeros(2 * n_bins)

    circles_af = df["circles_alt"] / df["circles_depth"]
    duplex_af = df["duplex_alt"] / df["duplex_depth"]

    circles_counts, _ = np.histogram(circles_af, bins=bins)
    duplex_counts, _ = np.histogram(duplex_af, bins=bins)

    features = np.concatenate([circles_counts, duplex_counts])
    return features


def predict_nb_with_confidence(model, X):
    """Predict Nb with confidence interval using Random Forest tree predictions."""
    tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])

    mean_pred = tree_predictions.mean(axis=0)
    std_pred = tree_predictions.std(axis=0)
    lower_95 = np.percentile(tree_predictions, 2.5, axis=0)
    upper_95 = np.percentile(tree_predictions, 97.5, axis=0)

    return {
        "mean": mean_pred,
        "std": std_pred,
        "lower_95": lower_95,
        "upper_95": upper_95,
    }


def plot_empirical_vs_test(
    y_test, y_test_pred, empirical_pred, empirical_name, output_path
):
    """
    Plot test set predictions with empirical prediction overlaid.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot test set as scatter
    ax.scatter(
        y_test, y_test_pred, alpha=0.5, s=50, color="black", label="Test set", zorder=1
    )

    # Add perfect prediction line
    min_val = min(y_test.min(), y_test_pred.min(), empirical_pred["lower_95"][0])
    max_val = max(y_test.max(), y_test_pred.max(), empirical_pred["upper_95"][0])

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        lw=1.5,
        alpha=0.5,
        label="Perfect prediction",
        zorder=2,
    )

    # Plot empirical prediction with 95% CI error bar
    emp_mean = empirical_pred["mean"][0]
    emp_lower_95 = empirical_pred["lower_95"][0]
    emp_upper_95 = empirical_pred["upper_95"][0]

    ax.errorbar(
        emp_mean,
        emp_mean,
        xerr=[[emp_mean - emp_lower_95], [emp_upper_95 - emp_mean]],
        fmt="o",
        markersize=10,
        color="red",
        elinewidth=2,
        capsize=6,
        capthick=2,
        label=f"{empirical_name} (95% CI)",
        zorder=4,
    )

    # Formatting
    ax.set_xlabel("True Nb (Test Set) / Predicted Nb (Empirical)", fontsize=12)
    ax.set_ylabel("Predicted Nb", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize empirical prediction against test set with confidence intervals"
    )
    parser.add_argument(
        "test_predictions_csv",
        type=Path,
        help="CSV with test set predictions (columns: y_true, y_pred)",
    )
    parser.add_argument(
        "empirical_csv", type=Path, help="CSV file with empirical shared mutations"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default="model_output/random_forest_model.joblib",
        help="Path to trained model",
    )
    parser.add_argument(
        "--bins",
        type=Path,
        default="model_output/af_bins.npy",
        help="Path to AF bins file",
    )
    parser.add_argument(
        "--empirical-name",
        type=str,
        default="Empirical data",
        help="Label for empirical data point (default: 'Empirical data')",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="empirical_vs_test.png",
        help="Output plot filename (default: empirical_vs_test.png)",
    )

    args = parser.parse_args()

    # Load model and bins
    print(f"Loading model from {args.model}...")
    model = joblib.load(args.model)
    bins = np.load(args.bins)
    print(f"  ✓ Model loaded: {model.n_estimators} trees")

    # Load test predictions
    print(f"\nLoading test predictions from {args.test_predictions_csv}...")
    df_test = pd.read_csv(args.test_predictions_csv)
    y_test = df_test["y_true"].values
    y_test_pred = df_test["y_pred"].values
    print(f"  ✓ Loaded {len(df_test)} test samples")

    # Load empirical data
    print(f"\nLoading empirical data from {args.empirical_csv}...")
    df_empirical = pd.read_csv(args.empirical_csv)
    print(f"  ✓ Loaded {len(df_empirical)} shared mutations")

    # Compute empirical prediction
    print("\nComputing empirical prediction...")
    X_empirical = compute_af_spectrum_empirical(df_empirical, bins).reshape(1, -1)
    empirical_pred = predict_nb_with_confidence(model, X_empirical)

    print(f"\nEmpirical prediction:")
    print(f"  Mean: {empirical_pred['mean'][0]:.1f}")
    print(
        f"  95% CI: [{empirical_pred['lower_95'][0]:.1f}, {empirical_pred['upper_95'][0]:.1f}]"
    )

    # Create plot
    print("\nCreating visualization...")
    plot_empirical_vs_test(
        y_test, y_test_pred, empirical_pred, args.empirical_name, args.output
    )

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
