#!/usr/bin/env python3
"""
Plot residuals and absolute errors for bottleneck size predictions.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns


def plot_residual_analysis(analysis_df, output_dir):
    """
    Comprehensive residual analysis for Nb predictions
    """
    output_dir = Path(output_dir)
    
    # Calculate metrics
    residuals = analysis_df['residual'].values
    abs_errors = np.abs(residuals)
    relative_errors = np.abs(analysis_df['relative_error'].values)
    true_nb = analysis_df['Nb'].values
    pred_nb = analysis_df['predicted_Nb'].values
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Residuals vs True Nb
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(true_nb, residuals, alpha=0.5, s=20, c=abs_errors, 
                         cmap='YlOrRd', edgecolors='none')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.set_xlabel('True Nb', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Residual (True - Predicted)', fontsize=12, fontweight='bold')
    ax1.set_title('Residuals vs True Nb', fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Absolute Error', fontsize=10)
    
    # 2. Absolute Error vs True Nb
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(true_nb, abs_errors, alpha=0.5, s=20, color='#E74C3C', edgecolors='none')
    ax2.set_xlabel('True Nb', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
    ax2.set_title('Absolute Error vs True Nb', fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Add median line
    nb_bins = np.logspace(np.log10(true_nb.min()), np.log10(true_nb.max()), 20)
    bin_centers = np.sqrt(nb_bins[:-1] * nb_bins[1:])
    median_errors = []
    for i in range(len(nb_bins) - 1):
        mask = (true_nb >= nb_bins[i]) & (true_nb < nb_bins[i+1])
        if mask.sum() > 0:
            median_errors.append(np.median(abs_errors[mask]))
        else:
            median_errors.append(np.nan)
    ax2.plot(bin_centers, median_errors, 'b-', linewidth=3, label='Median error', alpha=0.8)
    ax2.legend()
    
    # 3. Relative Error vs True Nb
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(true_nb, relative_errors * 100, alpha=0.5, s=20, 
               color='#3498DB', edgecolors='none')
    ax3.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.7, label='100% error')
    ax3.set_xlabel('True Nb', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Relative Error vs True Nb', fontsize=13, fontweight='bold')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Histogram of Residuals
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(residuals, bins=50, alpha=0.7, color='#2ECC71', edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax4.axvline(x=np.median(residuals), color='blue', linestyle='--', linewidth=2, 
               label=f'Median: {np.median(residuals):.2f}')
    ax4.set_xlabel('Residual (True - Predicted)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax4.set_title('Distribution of Residuals', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend()
    
    # 5. Histogram of Absolute Errors
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(abs_errors, bins=50, alpha=0.7, color='#E74C3C', edgecolor='black')
    ax5.axvline(x=np.median(abs_errors), color='blue', linestyle='--', linewidth=2,
               label=f'Median: {np.median(abs_errors):.2f}')
    ax5.axvline(x=np.mean(abs_errors), color='green', linestyle='--', linewidth=2,
               label=f'Mean (MAE): {np.mean(abs_errors):.2f}')
    ax5.set_xlabel('Absolute Error', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax5.set_title('Distribution of Absolute Errors', fontsize=13, fontweight='bold')
    ax5.set_xscale('log')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.legend()
    
    # 6. Q-Q plot (normality check for residuals)
    ax6 = fig.add_subplot(gs[1, 2])
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax6)
    ax6.set_title('Q-Q Plot (Residual Normality)', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. Error by Nb bins (boxplot)
    ax7 = fig.add_subplot(gs[2, 0])
    nb_ranges = [(1, 10), (10, 100), (100, 1000)]
    labels = ['1-10', '10-100', '100-1000']
    colors = ['#3498DB', '#E74C3C', '#F39C12']
    
    data_to_plot = []
    for nb_min, nb_max in nb_ranges:
        mask = (true_nb >= nb_min) & (true_nb < nb_max)
        if mask.sum() > 0:
            data_to_plot.append(abs_errors[mask])
        else:
            data_to_plot.append([])
    
    bp = ax7.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax7.set_xlabel('Nb Range', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
    ax7.set_title('Error Distribution by Nb Range', fontsize=13, fontweight='bold')
    ax7.set_yscale('log')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Performance metrics by Nb range
    ax8 = fig.add_subplot(gs[2, 1])
    
    mae_by_range = []
    r2_by_range = []
    for nb_min, nb_max in nb_ranges:
        mask = (true_nb >= nb_min) & (true_nb < nb_max)
        if mask.sum() > 0:
            mae_by_range.append(mean_absolute_error(true_nb[mask], pred_nb[mask]))
            r2_by_range.append(r2_score(true_nb[mask], pred_nb[mask]))
        else:
            mae_by_range.append(0)
            r2_by_range.append(0)
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax8_twin = ax8.twinx()
    bars1 = ax8.bar(x - width/2, mae_by_range, width, label='MAE', 
                    color='#E74C3C', alpha=0.7, edgecolor='black')
    bars2 = ax8_twin.bar(x + width/2, r2_by_range, width, label='R²',
                         color='#3498DB', alpha=0.7, edgecolor='black')
    
    ax8.set_xlabel('Nb Range', fontsize=12, fontweight='bold')
    ax8.set_ylabel('MAE', fontsize=12, fontweight='bold', color='#E74C3C')
    ax8_twin.set_ylabel('R² Score', fontsize=12, fontweight='bold', color='#3498DB')
    ax8.set_title('Performance Metrics by Nb Range', fontsize=13, fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_xticklabels(labels)
    ax8.tick_params(axis='y', labelcolor='#E74C3C')
    ax8_twin.tick_params(axis='y', labelcolor='#3498DB')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars1, mae_by_range):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    for bar, val in zip(bars2, r2_by_range):
        height = bar.get_height()
        ax8_twin.text(bar.get_x() + bar.get_width()/2., height,
                     f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Combined legend
    lines1, labels1 = ax8.get_legend_handles_labels()
    lines2, labels2 = ax8_twin.get_legend_handles_labels()
    ax8.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 9. Cumulative distribution of absolute errors
    ax9 = fig.add_subplot(gs[2, 2])
    sorted_errors = np.sort(abs_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    
    ax9.plot(sorted_errors, cumulative, linewidth=2, color='#2ECC71')
    ax9.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50th percentile')
    ax9.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90th percentile')
    ax9.axhline(y=95, color='purple', linestyle='--', alpha=0.5, label='95th percentile')
    
    # Add percentile values
    p50 = np.percentile(abs_errors, 50)
    p90 = np.percentile(abs_errors, 90)
    p95 = np.percentile(abs_errors, 95)
    
    ax9.axvline(x=p50, color='red', linestyle=':', alpha=0.5)
    ax9.axvline(x=p90, color='orange', linestyle=':', alpha=0.5)
    ax9.axvline(x=p95, color='purple', linestyle=':', alpha=0.5)
    
    ax9.set_xlabel('Absolute Error', fontsize=12, fontweight='bold')
    ax9.set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
    ax9.set_title('Cumulative Distribution of Errors', fontsize=13, fontweight='bold')
    ax9.set_xscale('log')
    ax9.grid(True, alpha=0.3)
    ax9.legend(loc='lower right')
    
    # Add text with percentile values
    ax9.text(0.02, 0.98, f'50th: {p50:.1f}\n90th: {p90:.1f}\n95th: {p95:.1f}',
            transform=ax9.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8,
                     edgecolor='black', linewidth=1.5))
    
    plt.savefig(output_dir / 'residual_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved comprehensive residual analysis to {output_dir / 'residual_analysis_comprehensive.png'}")
    plt.close()


def plot_error_by_nb_detailed(analysis_df, output_dir):
    """
    Detailed plot of errors stratified by Nb value
    """
    output_dir = Path(output_dir)
    
    true_nb = analysis_df['Nb'].values
    abs_errors = np.abs(analysis_df['residual'].values)
    
    # Create fine-grained Nb bins
    nb_bins = np.logspace(np.log10(true_nb.min()), np.log10(true_nb.max()), 30)
    bin_centers = np.sqrt(nb_bins[:-1] * nb_bins[1:])
    
    mean_errors = []
    median_errors = []
    q25_errors = []
    q75_errors = []
    counts = []
    
    for i in range(len(nb_bins) - 1):
        mask = (true_nb >= nb_bins[i]) & (true_nb < nb_bins[i+1])
        if mask.sum() > 0:
            errors_in_bin = abs_errors[mask]
            mean_errors.append(np.mean(errors_in_bin))
            median_errors.append(np.median(errors_in_bin))
            q25_errors.append(np.percentile(errors_in_bin, 25))
            q75_errors.append(np.percentile(errors_in_bin, 75))
            counts.append(mask.sum())
        else:
            mean_errors.append(np.nan)
            median_errors.append(np.nan)
            q25_errors.append(np.nan)
            q75_errors.append(np.nan)
            counts.append(0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Error trends
    ax1.fill_between(bin_centers, q25_errors, q75_errors, alpha=0.3, color='#3498DB',
                     label='25th-75th percentile')
    ax1.plot(bin_centers, median_errors, 'o-', linewidth=2, markersize=6,
            color='#E74C3C', label='Median error')
    ax1.plot(bin_centers, mean_errors, 's-', linewidth=2, markersize=6,
            color='#2ECC71', label='Mean error (MAE)')
    
    ax1.set_xlabel('True Nb', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Absolute Error', fontsize=13, fontweight='bold')
    ax1.set_title('Error Statistics across Nb Values', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Sample counts per bin
    ax2.bar(bin_centers, counts, width=np.diff(nb_bins), alpha=0.7, 
           color='#95A5A6', edgecolor='black')
    ax2.set_xlabel('True Nb', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Sample Count', fontsize=13, fontweight='bold')
    ax2.set_title('Distribution of Samples across Nb Values', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_by_nb_detailed.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved detailed error by Nb plot to {output_dir / 'error_by_nb_detailed.png'}")
    plt.close()


def print_error_summary(analysis_df):
    """Print summary statistics for errors"""
    
    residuals = analysis_df['residual'].values
    abs_errors = np.abs(residuals)
    relative_errors = np.abs(analysis_df['relative_error'].values)
    true_nb = analysis_df['Nb'].values
    pred_nb = analysis_df['predicted_Nb'].values
    
    print("\n" + "="*70)
    print("ERROR ANALYSIS SUMMARY")
    print("="*70)
    
    print("\nOverall Statistics:")
    print(f"  Mean Residual: {np.mean(residuals):.2f}")
    print(f"  Median Residual: {np.median(residuals):.2f}")
    print(f"  Std Residual: {np.std(residuals):.2f}")
    
    print("\nAbsolute Error:")
    print(f"  MAE (Mean): {np.mean(abs_errors):.2f}")
    print(f"  Median: {np.median(abs_errors):.2f}")
    print(f"  25th percentile: {np.percentile(abs_errors, 25):.2f}")
    print(f"  75th percentile: {np.percentile(abs_errors, 75):.2f}")
    print(f"  90th percentile: {np.percentile(abs_errors, 90):.2f}")
    print(f"  95th percentile: {np.percentile(abs_errors, 95):.2f}")
    print(f"  Max: {np.max(abs_errors):.2f}")
    
    print("\nRelative Error:")
    print(f"  MAPE: {np.mean(relative_errors) * 100:.1f}%")
    print(f"  Median: {np.median(relative_errors) * 100:.1f}%")
    
    print("\nModel Performance:")
    print(f"  R²: {r2_score(true_nb, pred_nb):.4f}")
    
    print("\nBy Nb Range:")
    nb_ranges = [(1, 10), (10, 100), (100, 1000)]
    for nb_min, nb_max in nb_ranges:
        mask = (true_nb >= nb_min) & (true_nb < nb_max)
        if mask.sum() > 0:
            mae = mean_absolute_error(true_nb[mask], pred_nb[mask])
            r2 = r2_score(true_nb[mask], pred_nb[mask])
            mape = np.mean(np.abs(relative_errors[mask])) * 100
            median_err = np.median(abs_errors[mask])
            print(f"\n  Nb ∈ [{nb_min}, {nb_max}):")
            print(f"    n = {mask.sum()}")
            print(f"    MAE = {mae:.2f}")
            print(f"    Median error = {median_err:.2f}")
            print(f"    MAPE = {mape:.1f}%")
            print(f"    R² = {r2:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot residual and error analysis for Nb predictions"
    )
    parser.add_argument(
        'analysis_csv',
        type=Path,
        help='Path to parameter_effects_detailed.csv'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=None,
        help='Output directory for plots (default: same as input CSV)'
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
    
    # Print summary statistics
    print_error_summary(analysis_df)
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    plot_residual_analysis(analysis_df, args.output_dir)
    plot_error_by_nb_detailed(analysis_df, args.output_dir)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()