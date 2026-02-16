#!/usr/bin/env python3
"""
Plot parameter effects and predictions from saved analysis results.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


def plot_correlations(analysis_df, output_dir):
    """
    Plot correlation heatmap
    """
    output_dir = Path(output_dir)

    # Select relevant columns for correlation matrix
    corr_cols = [
        "N",
        "u_per_site",
        "Gs",
        "Nb",
        "predicted_Nb",
        "residual",
        "relative_error",
    ]
    corr_df = analysis_df[corr_cols].copy()

    # Compute correlation matrix
    corr_matrix = corr_df.corr()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # Plot heatmap with proper formatting
    im = ax.imshow(
        corr_matrix,
        cmap="coolwarm",
        aspect="auto",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )

    # Set ticks and labels
    n_vars = len(corr_cols)
    ax.set_xticks(np.arange(n_vars))
    ax.set_yticks(np.arange(n_vars))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(corr_cols, fontsize=12)

    # Add correlation values as text
    for i in range(n_vars):
        for j in range(n_vars):
            if not mask[i, j]:  # Only show lower triangle
                text_color = "black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white"
                ax.text(
                    j,
                    i,
                    f"{corr_matrix.iloc[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=11,
                    fontweight="bold",
                )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation Coefficient", fontsize=12, fontweight="bold")

    ax.set_title("Parameter Correlation Matrix", fontsize=14, fontweight="bold", pad=20)

    # Add grid lines
    ax.set_xticks(np.arange(n_vars + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_vars + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", size=0)

    plt.tight_layout()
    plt.savefig(output_dir / "parameter_correlations.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved correlation plot to {output_dir / 'parameter_correlations.png'}")
    plt.close()


def plot_predictions_colored_by_params(analysis_df, output_dir):
    """
    Plot predicted vs true Nb, colored by each parameter
    """
    output_dir = Path(output_dir)

    params = ["N", "u_per_site", "Gs"]
    param_labels = {
        "N": "Population Size (N)",
        "u_per_site": "Mutation Rate (μ)",
        "Gs": "Generations",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, param in enumerate(params):
        ax = axes[i]

        # Create scatter plot
        scatter = ax.scatter(
            analysis_df["Nb"],
            analysis_df["predicted_Nb"],
            c=analysis_df[param],
            cmap="viridis",
            alpha=0.6,
            s=30,
            edgecolors="none",
        )

        # Add perfect prediction line
        min_val = min(analysis_df["Nb"].min(), analysis_df["predicted_Nb"].min())
        max_val = max(analysis_df["Nb"].max(), analysis_df["predicted_Nb"].max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=2,
            alpha=0.7,
            label="Perfect prediction",
        )

        # Formatting
        ax.set_xlabel("True Nb", fontsize=12, fontweight="bold")
        ax.set_ylabel("Predicted Nb", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Colored by {param_labels[param]}", fontsize=13, fontweight="bold"
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=10)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(param_labels[param], fontsize=11, fontweight="bold")

        # Add R² to plot
        from sklearn.metrics import r2_score

        r2 = r2_score(analysis_df["Nb"], analysis_df["predicted_Nb"])
        ax.text(
            0.98,
            0.02,
            f"R² = {r2:.4f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round",
                facecolor="wheat",
                alpha=0.8,
                edgecolor="black",
                linewidth=1.5,
            ),
        )

    plt.tight_layout()
    plt.savefig(
        output_dir / "predictions_colored_by_params.png", dpi=300, bbox_inches="tight"
    )
    print(
        f"✓ Saved predictions colored by parameters to {output_dir / 'predictions_colored_by_params.png'}"
    )
    plt.close()


def plot_residuals_vs_params(analysis_df, output_dir):
    """
    Plot residuals vs each parameter to check for systematic bias
    """
    output_dir = Path(output_dir)

    params = ["N", "u_per_site", "Gs"]
    param_labels = {
        "N": "Population Size (N)",
        "u_per_site": "Mutation Rate (μ)",
        "Gs": "Generations",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, param in enumerate(params):
        # Top row: Residuals vs parameter
        ax = axes[0, i]
        scatter = ax.scatter(
            analysis_df[param],
            analysis_df["residual"],
            c=analysis_df["Nb"],
            cmap="plasma",
            alpha=0.5,
            s=20,
            edgecolors="none",
        )

        ax.axhline(y=0, color="red", linestyle="--", linewidth=2, alpha=0.7)
        ax.set_xlabel(param_labels[param], fontsize=12, fontweight="bold")
        ax.set_ylabel("Residual (True - Predicted)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Residuals vs {param_labels[param]}", fontsize=13, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("True Nb", fontsize=10)

        # Add correlation value
        corr = np.corrcoef(analysis_df[param], analysis_df["residual"])[0, 1]
        ax.text(
            0.02,
            0.98,
            f"r = {corr:.4f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round",
                facecolor="wheat",
                alpha=0.8,
                edgecolor="black",
                linewidth=1.5,
            ),
        )

        # Bottom row: Absolute residuals vs parameter
        ax = axes[1, i]
        scatter = ax.scatter(
            analysis_df[param],
            np.abs(analysis_df["residual"]),
            c=analysis_df["Nb"],
            cmap="plasma",
            alpha=0.5,
            s=20,
            edgecolors="none",
        )

        ax.set_xlabel(param_labels[param], fontsize=12, fontweight="bold")
        ax.set_ylabel("Absolute Error", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Absolute Error vs {param_labels[param]}", fontsize=13, fontweight="bold"
        )
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("True Nb", fontsize=10)

        # Add correlation value
        corr = np.corrcoef(analysis_df[param], np.abs(analysis_df["residual"]))[0, 1]
        ax.text(
            0.02,
            0.98,
            f"r = {corr:.4f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round",
                facecolor="wheat",
                alpha=0.8,
                edgecolor="black",
                linewidth=1.5,
            ),
        )

    plt.tight_layout()
    plt.savefig(output_dir / "residuals_vs_params.png", dpi=300, bbox_inches="tight")
    print(
        f"✓ Saved residuals vs parameters to {output_dir / 'residuals_vs_params.png'}"
    )
    plt.close()


def plot_performance_by_param_bins(analysis_df, output_dir):
    """
    Plot model performance (R², MAE) across parameter bins
    """
    output_dir = Path(output_dir)

    params = ["N", "u_per_site", "Gs"]
    param_labels = {
        "N": "Population Size",
        "u_per_site": "Mutation Rate",
        "Gs": "Generations",
    }

    from sklearn.metrics import r2_score, mean_absolute_error

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, param in enumerate(params):
        # Get quartiles
        param_values = analysis_df[param].values
        quartiles = np.percentile(param_values, [0, 25, 50, 75, 100])

        r2_scores = []
        mae_scores = []
        labels = []

        for j in range(len(quartiles) - 1):
            mask = (param_values >= quartiles[j]) & (param_values < quartiles[j + 1])
            if j == len(quartiles) - 2:
                mask = (param_values >= quartiles[j]) & (
                    param_values <= quartiles[j + 1]
                )

            if mask.sum() > 0:
                subset = analysis_df[mask]
                r2 = r2_score(subset["Nb"], subset["predicted_Nb"])
                mae = mean_absolute_error(subset["Nb"], subset["predicted_Nb"])

                r2_scores.append(r2)
                mae_scores.append(mae)
                labels.append(f"Q{j + 1}")

        # Plot R² scores
        ax = axes[0, i]
        colors = ["#3498DB", "#E74C3C", "#F39C12", "#2ECC71"]
        bars = ax.bar(
            range(len(r2_scores)),
            r2_scores,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.set_xlabel("Quartile", fontsize=12, fontweight="bold")
        ax.set_ylabel("R² Score", fontsize=11, fontweight="bold")
        ax.set_title(
            f"R² across {param_labels[param]} Quartiles", fontsize=13, fontweight="bold"
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylim(0.8, 1.0)
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.5, linewidth=1)

        # Add values on bars
        for bar, val in zip(bars, r2_scores):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        # Plot MAE scores
        ax = axes[1, i]
        bars = ax.bar(
            range(len(mae_scores)),
            mae_scores,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.set_xlabel("Quartile", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Absolute Error", fontsize=11, fontweight="bold")
        ax.set_title(
            f"MAE across {param_labels[param]} Quartiles",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.grid(True, alpha=0.3, axis="y")

        # Add values on bars
        for bar, val in zip(bars, mae_scores):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(
        output_dir / "performance_by_param_bins.png", dpi=300, bbox_inches="tight"
    )
    print(
        f"✓ Saved performance by parameter bins to {output_dir / 'performance_by_param_bins.png'}"
    )
    plt.close()


def print_summary_stats(analysis_df):
    """Print summary statistics"""
    from sklearn.metrics import r2_score, mean_absolute_error

    print("\n" + "=" * 70)
    print("PARAMETER EFFECT SUMMARY")
    print("=" * 70)

    # Overall performance
    r2 = r2_score(analysis_df["Nb"], analysis_df["predicted_Nb"])
    mae = mean_absolute_error(analysis_df["Nb"], analysis_df["predicted_Nb"])

    print(f"\nOverall Model Performance:")
    print(f"  R² = {r2:.4f}")
    print(f"  MAE = {mae:.2f}")
    print(f"  Total samples = {len(analysis_df)}")

    # Parameter correlations with error
    print(f"\nCorrelation with Absolute Error:")
    params = ["N", "u_per_site", "Gs"]
    for param in params:
        corr = np.corrcoef(analysis_df[param], np.abs(analysis_df["residual"]))[0, 1]
        print(f"  {param}: {corr:.4f}")

    # Parameter ranges
    print(f"\nParameter Ranges:")
    for param in params:
        print(
            f"  {param}: [{analysis_df[param].min():.2e}, {analysis_df[param].max():.2e}]"
        )
        print(f"    Unique values: {analysis_df[param].nunique()}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot parameter effects from saved analysis results"
    )
    parser.add_argument(
        "analysis_csv", type=Path, help="Path to parameter_effects_detailed.csv"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: same as input CSV)",
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.analysis_csv.parent
    args.output_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    print(f"Loading analysis data from {args.analysis_csv}...")
    analysis_df = pd.read_csv(args.analysis_csv)
    print(f"  Loaded {len(analysis_df)} samples")

    # Print summary
    print_summary_stats(analysis_df)

    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    plot_correlations(analysis_df, args.output_dir)
    plot_predictions_colored_by_params(analysis_df, args.output_dir)
    plot_residuals_vs_params(analysis_df, args.output_dir)
    plot_performance_by_param_bins(analysis_df, args.output_dir)

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
