#!/usr/bin/env python3
"""
MAPE Distribution by Value Range Plot
Relative error (MAPE) visualization instead of absolute error
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

def create_mape_distribution_plot(df):
    """
    Create MAPE distribution by value range (single plot)
    Shows MAPE (%) with sample size n for each range
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    y_true, y_pred = df['y_true'], df['y_pred']
    residuals = y_true - y_pred

    # Calculate MAPE for each sample
    mape_per_sample = np.abs(residuals / y_true) * 100

    # Define value bins
    bins = [0, 500, 1000, 1500, 2000, float('inf')]
    bin_labels = ['0-500', '500-1000', '1000-1500', '1500-2000', '2000+']

    # Create bin column
    df_copy = df.copy()
    df_copy['value_bin'] = pd.cut(df_copy['y_true'], bins=bins, labels=bin_labels, right=False)
    df_copy['mape'] = mape_per_sample

    # Calculate statistics for each bin
    stats_by_bin = df_copy.groupby('value_bin').agg({
        'mape': 'mean',
        'y_true': 'count'
    }).rename(columns={'y_true': 'n'})

    # Also calculate median and std for reference
    stats_by_bin['mape_median'] = df_copy.groupby('value_bin')['mape'].median()
    stats_by_bin['mape_std'] = df_copy.groupby('value_bin')['mape'].std()

    # Set up bar positions
    x = np.arange(len(bin_labels))
    width = 0.5

    # Plot bars - MAPE only
    bars = ax.bar(x, stats_by_bin['mape'], width,
                  label='MAPE (%)', color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1)

    # Add value labels and sample size on bars
    for i, bar in enumerate(bars):
        mape_val = stats_by_bin['mape'].iloc[i]
        n_val = stats_by_bin['n'].iloc[i]

        # MAPE value label on bar
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{mape_val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Sample size annotation above bar
        ax.text(i, bar.get_height() + bar.get_height()*0.1, f'n={int(n_val)}',
               ha='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax.set_xlabel('Value Range (10K CNY)', fontsize=12)
    ax.set_ylabel('Mean Absolute Percentage Error (%)', fontsize=12)
    ax.set_title('MAPE Distribution by Value Range', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_file = 'PhySO_Regression_MAPE_Distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"MAPE distribution plot saved as: {output_file}")
    plt.close()

    return output_file

def print_mape_statistics(df):
    """Print detailed MAPE statistics"""
    print("\n" + "="*70)
    print("MAPE STATISTICS BY VALUE RANGE")
    print("="*70)

    y_true, y_pred = df['y_true'], df['y_pred']
    residuals = y_true - y_pred
    mape = np.abs(residuals / y_true) * 100

    # Overall MAPE
    print(f"Overall MAPE: {np.mean(mape):.2f}%")
    print(f"Median MAPE: {np.median(mape):.2f}%")
    print(f"Std MAPE: {np.std(mape):.2f}%")

    # MAPE by value range
    bins = [0, 500, 1000, 1500, 2000, float('inf')]
    bin_labels = ['0-500', '500-1000', '1000-1500', '1500-2000', '2000+']

    df_copy = df.copy()
    df_copy['value_bin'] = pd.cut(df_copy['y_true'], bins=bins, labels=bin_labels, right=False)
    df_copy['mape'] = mape

    print("\nMAPE by Value Range:")
    for label in bin_labels:
        subset = df_copy[df_copy['value_bin'] == label]
        if len(subset) > 0:
            print(f"  {label}: MAPE={np.mean(subset['mape']):.2f}%, n={len(subset)}")

    print("="*70)

def main():
    """Main function"""
    print("="*70)
    print("MAPE DISTRIBUTION VISUALIZATION")
    print("="*70)

    # Load data
    df = load_regression_data()
    if df is None:
        return

    # Print MAPE statistics
    print_mape_statistics(df)

    # Create MAPE distribution plot
    print("\nCreating MAPE distribution plot...")
    mape_file = create_mape_distribution_plot(df)

    print("\n" + "="*70)
    print("MAPE DISTRIBUTION PLOT COMPLETED")
    print("="*70)
    print(f"Generated file: {mape_file}")
    print("="*70)

if __name__ == "__main__":
    main()
