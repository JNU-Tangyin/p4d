#!/usr/bin/env python3
"""
NMAE Distribution by Value Range Plot
Normalized error (NMAE) visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set matplotlib parameters for better English-only output
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_regression_data():
    """Load regression prediction data from results/investment_decision/"""
    predictions_file = Path("results/investment_decision/physo_predictions.csv")

    if predictions_file.exists():
        print(f"Loading regression data from: {predictions_file}")
        df = pd.read_csv(predictions_file)
        print(f"Loaded regression predictions: {len(df)} samples")
        return df

    print("ERROR: No regression prediction data found!")
    print("Please run: python main.py --task_type regression --epochs 15")
    return None

def create_nmae_distribution_plot(df):
    """
    Create NMAE distribution by value range (single plot)
    Shows NMAE (%) with sample size n for each range
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    y_true, y_pred = df['y_true'], df['y_pred']
    residuals = y_true - y_pred
    abs_errors = np.abs(residuals)

    # Calculate normalization factor (range of y_true)
    y_range = y_true.max() - y_true.min()

    # Calculate NMAE for each sample
    nmae_per_sample = (abs_errors / y_range) * 100

    # Define value bins
    bins = [0, 500, 1000, 1500, 2000, 4000]
    bin_labels = ['0-500', '500-1000', '1000-1500', '1500-2000', '2000+']

    # Create bin column
    df_copy = df.copy()
    df_copy['value_bin'] = pd.cut(df_copy['y_true'], bins=bins, labels=bin_labels, right=False)
    df_copy['nmae'] = nmae_per_sample

    # Calculate statistics for each bin
    stats_by_bin = df_copy.groupby('value_bin').agg({
        'nmae': 'mean',
        'y_true': 'count'
    }).rename(columns={'y_true': 'n'})

    # Also calculate median and std for reference
    stats_by_bin['nmae_median'] = df_copy.groupby('value_bin')['nmae'].median()
    stats_by_bin['nmae_std'] = df_copy.groupby('value_bin')['nmae'].std()

    # Set up bar positions
    x = np.arange(len(bin_labels))
    width = 0.5

    # Plot bars - NMAE only
    bars = ax.bar(x, stats_by_bin['nmae'], width,
                  label='NMAE (%)', color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1)

    # Add value labels and sample size on bars
    for i, bar in enumerate(bars):
        nmae_val = stats_by_bin['nmae'].iloc[i]
        n_val = stats_by_bin['n'].iloc[i]

        # NMAE value label on bar
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{nmae_val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Sample size annotation above bar
        ax.text(i, bar.get_height() + bar.get_height()*0.1, f'n={int(n_val)}',
               ha='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax.set_xlabel('Value Range (10K CNY)', fontsize=12)
    ax.set_ylabel('Normalized Mean Absolute Error (%)', fontsize=12)
    ax.set_title('NMAE Distribution by Value Range', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_file = 'PhySO_Regression_NMAE_Distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"NMAE distribution plot saved as: {output_file}")
    plt.close()

    return output_file

def print_nmae_statistics(df):
    """Print detailed NMAE statistics"""
    print("\n" + "="*70)
    print("NMAE STATISTICS BY VALUE RANGE")
    print("="*70)

    y_true, y_pred = df['y_true'], df['y_pred']
    residuals = y_true - y_pred
    abs_errors = np.abs(residuals)

    # Calculate normalization factor
    y_range = y_true.max() - y_true.min()
    print(f"Normalization factor (y range): {y_range:.2f}")

    # Calculate NMAE
    nmae = (abs_errors / y_range) * 100

    # Overall NMAE
    print(f"Overall NMAE: {np.mean(nmae):.2f}%")
    print(f"Median NMAE: {np.median(nmae):.2f}%")
    print(f"Std NMAE: {np.std(nmae):.2f}%")

    # NMAE by value range
    bins = [0, 500, 1000, 1500, 2000, 4000]
    bin_labels = ['0-500', '500-1000', '1000-1500', '1500-2000', '2000+']

    df_copy = df.copy()
    df_copy['value_bin'] = pd.cut(df_copy['y_true'], bins=bins, labels=bin_labels, right=False)
    df_copy['nmae'] = nmae

    print("\nNMAE by Value Range:")
    for label in bin_labels:
        subset = df_copy[df_copy['value_bin'] == label]
        if len(subset) > 0:
            print(f"  {label}: NMAE={np.mean(subset['nmae']):.2f}%, n={len(subset)}")

    print("="*70)

def main():
    """Main function"""
    print("="*70)
    print("NMAE DISTRIBUTION VISUALIZATION")
    print("="*70)

    # Load data
    df = load_regression_data()
    if df is None:
        return

    # Print NMAE statistics
    print_nmae_statistics(df)

    # Create NMAE distribution plot
    print("\nCreating NMAE distribution plot...")
    nmae_file = create_nmae_distribution_plot(df)

    print("\n" + "="*70)
    print("NMAE DISTRIBUTION PLOT COMPLETED")
    print("="*70)
    print(f"Generated file: {nmae_file}")
    print("="*70)

if __name__ == "__main__":
    main()
