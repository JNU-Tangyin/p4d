#!/usr/bin/env python3
"""
Improved Regression Charts
根据回归作图.txt建议创建改进的回归图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Set matplotlib parameters for better English-only output
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_regression_data():
    """Load regression prediction data"""
    # Find the latest experiment data
    experiment_dir = Path("archive/old_experiments/experiment_data")
    if not experiment_dir.exists():
        print("No experiment_data directory found in archive!")
        return None

    # Find latest data directory
    data_dirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
    if not data_dirs:
        print("No experiment data directories found!")
        return None

    latest_dir = max(data_dirs, key=lambda x: x.stat().st_mtime)
    print(f"Loading regression data from: {latest_dir}")

    # Load predictions data
    predictions_file = latest_dir / "regression" / "physo_predictions.csv"
    if not predictions_file.exists():
        print(f"Predictions file not found: {predictions_file}")
        return None

    df = pd.read_csv(predictions_file)
    print(f"Loaded regression predictions: {len(df)} samples")
    return df

def create_log_scale_parity_plot(df):
    """Create log-scale parity plot as suggested"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: Regular scale
    ax1.scatter(df['y_true'], df['y_pred'], alpha=0.7, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
    min_val, max_val = min(df['y_true'].min(), df['y_pred'].min()), max(df['y_true'].max(), df['y_pred'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Add linear fit
    z = np.polyfit(df['y_true'], df['y_pred'], 1)
    p = np.poly1d(z)
    x_fit = np.linspace(min_val, max_val, 100)
    ax1.plot(x_fit, p(x_fit), 'g-', linewidth=2, alpha=0.7, label=f'Linear Fit (slope={z[0]:.2f})')

    ax1.set_xlabel('Actual Values', fontsize=12)
    ax1.set_ylabel('Predicted Values', fontsize=12)
    ax1.set_title('Actual vs Predicted (Linear Scale)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Calculate R²
    r2 = 1 - np.sum((df['y_true'] - df['y_pred'])**2) / np.sum((df['y_true'] - df['y_true'].mean())**2)
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Right: Log scale
    ax2.scatter(df['y_true'], df['y_pred'], alpha=0.7, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.plot(x_fit, p(x_fit), 'g-', linewidth=2, alpha=0.7, label=f'Linear Fit (slope={z[0]:.2f})')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Actual Values (log scale)', fontsize=12)
    ax2.set_ylabel('Predicted Values (log scale)', fontsize=12)
    ax2.set_title('Actual vs Predicted (Log Scale)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('PhySO Regression: Parity Plot Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_file = 'PhySO_Regression_Parity_Comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Parity plot saved as: {output_file}")
    plt.show()

    return output_file

def create_residual_analysis_plot(df):
    """Create residual analysis plot as suggested"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Residuals vs Fitted Values
    residuals = df['y_true'] - df['y_pred']
    ax1.scatter(df['y_pred'], residuals, alpha=0.7, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)

    # Add trend line for residuals
    z = np.polyfit(df['y_pred'], residuals, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['y_pred'].min(), df['y_pred'].max(), 100)
    ax1.plot(x_trend, p(x_trend), 'g-', linewidth=2, alpha=0.7, label=f'Trend (slope={z[0]:.3f})')

    ax1.set_xlabel('Fitted Values', fontsize=12)
    ax1.set_ylabel('Residuals', fontsize=12)
    ax1.set_title('Residuals vs Fitted Values', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add annotation about systematic underestimation
    ax1.text(0.05, 0.95, 'Systematic underestimation\nat high values',
             transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
             fontsize=10, verticalalignment='top')

    # 2. Residuals vs Actual Values
    ax2.scatter(df['y_true'], residuals, alpha=0.7, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Actual Values', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Residuals vs Actual Values', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Histogram of Residuals
    ax3.hist(residuals, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Residuals', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distribution of Residuals', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Add statistics
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    ax3.text(0.05, 0.95, f'Mean: {mean_res:.1f}\nStd: {std_res:.1f}',
             transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 4. Percentage Error vs Actual Values
    ax4.scatter(df['y_true'], df['percentage_error'], alpha=0.7, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax4.set_xlabel('Actual Values', fontsize=12)
    ax4.set_ylabel('Percentage Error (%)', fontsize=12)
    ax4.set_title('Percentage Error vs Actual Values', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('PhySO Regression: Comprehensive Residual Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_file = 'PhySO_Regression_Residual_Analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Residual analysis plot saved as: {output_file}")
    plt.show()

    return output_file

def create_performance_summary_plot(df):
    """Create performance summary with key metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Calculate metrics
    y_true, y_pred = df['y_true'], df['y_pred']
    residuals = y_true - y_pred

    # R² Score
    r2 = 1 - np.sum(residuals**2) / np.sum((y_true - y_true.mean())**2)
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    mape = np.mean(np.abs(residuals / y_true)) * 100

    # 1. Metrics Summary (Bar chart)
    metrics = ['R²', 'RMSE', 'MAE', 'MAPE (%)']
    values = [r2, rmse, mae, mape]
    colors = ['#2E8B57', '#4682B4', '#CD5C5C', '#FFB347']

    bars = ax1.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Performance Metrics Summary', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. Prediction Range Analysis
    actual_range = [y_true.min(), y_true.max()]
    pred_range = [y_pred.min(), y_pred.max()]

    ax2.bar(['Actual Min', 'Actual Max', 'Predicted Min', 'Predicted Max'],
            [actual_range[0], actual_range[1], pred_range[0], pred_range[1]],
            color=['#FF6B6B', '#FF6B6B', '#4ECDC4', '#4ECDC4'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Value Range Comparison', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Error Distribution by Value Range
    # Create value bins
    bins = [0, 500, 1000, 1500, 2000, float('inf')]
    bin_labels = ['0-500', '500-1000', '1000-1500', '1500-2000', '2000+']

    df['value_bin'] = pd.cut(df['y_true'], bins=bins, labels=bin_labels, right=False)
    error_by_bin = df.groupby('value_bin')['abs_error'].mean()

    ax3.bar(error_by_bin.index, error_by_bin.values, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Value Range', fontsize=12)
    ax3.set_ylabel('Mean Absolute Error', fontsize=12)
    ax3.set_title('Error Distribution by Value Range', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Actual vs Predicted Scatter with Focus Area
    ax4.scatter(df['y_true'], df['y_pred'], alpha=0.7, s=50, color='steelblue', edgecolors='black', linewidth=0.5)

    # Perfect prediction line
    min_val, max_val = 0, 2500
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Add focus area rectangle (0-600 range as suggested)
    focus_rect = mpatches.Rectangle((0, 0), 600, 600, linewidth=2,
                                   edgecolor='orange', facecolor='orange', alpha=0.2)
    ax4.add_patch(focus_rect)

    ax4.set_xlim(0, max_val)
    ax4.set_ylim(0, max_val)
    ax4.set_xlabel('Actual Values', fontsize=12)
    ax4.set_ylabel('Predicted Values', fontsize=12)
    ax4.set_title('Actual vs Predicted with Focus Area Highlight', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Add annotation
    ax4.text(0.05, 0.95, 'Orange rectangle:\nMain data range\n(0-600)',
             transform=ax4.transAxes, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
             fontsize=10, verticalalignment='top')

    plt.suptitle('PhySO Regression: Comprehensive Performance Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_file = 'PhySO_Regression_Performance_Summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Performance summary plot saved as: {output_file}")
    plt.show()

    return output_file

def print_detailed_statistics(df):
    """Print detailed regression statistics"""
    print("\n" + "="*70)
    print("DETAILED REGRESSION STATISTICS")
    print("="*70)

    y_true, y_pred = df['y_true'], df['y_pred']
    residuals = y_true - y_pred

    # Basic metrics
    r2 = 1 - np.sum(residuals**2) / np.sum((y_true - y_true.mean())**2)
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    mape = np.mean(np.abs(residuals / y_true)) * 100

    print(f"Dataset Size: {len(df)} samples")
    print(f"Actual Value Range: [{y_true.min():.1f}, {y_true.max():.1f}]")
    print(f"Predicted Value Range: [{y_pred.min():.1f}, {y_pred.max():.1f}]")
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # Error analysis
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    max_positive_error = np.max(residuals)
    max_negative_error = np.min(residuals)

    print(f"\nResidual Analysis:")
    print(f"Mean Residual: {mean_residual:.2f}")
    print(f"Std Residual: {std_residual:.2f}")
    print(f"Max Positive Error (underestimation): {max_positive_error:.2f}")
    print(f"Max Negative Error (overestimation): {max_negative_error:.2f}")

    # Value range analysis
    low_values = df[df['y_true'] < 600]
    high_values = df[df['y_true'] >= 600]

    if len(low_values) > 0:
        low_mae = np.mean(np.abs(low_values['y_true'] - low_values['y_pred']))
        print(f"\nLow Value Range (<600): {len(low_values)} samples, MAE: {low_mae:.2f}")

    if len(high_values) > 0:
        high_mae = np.mean(np.abs(high_values['y_true'] - high_values['y_pred']))
        print(f"High Value Range (>=600): {len(high_values)} samples, MAE: {high_mae:.2f}")

    # Systematic bias analysis
    high_value_bias = np.mean(residuals[df['y_true'] >= 1000]) if len(df[df['y_true'] >= 1000]) > 0 else 0
    print(f"\nHigh Value Systematic Bias (>=1000): {high_value_bias:.2f}")

    print("="*70)

def main():
    """Main function"""
    print("="*70)
    print("IMPROVED PHYSO REGRESSION VISUALIZATION")
    print("Based on regression plotting recommendations")
    print("="*70)

    # Load data
    df = load_regression_data()
    if df is None:
        print("Failed to load regression data!")
        return

    # Print detailed statistics
    print_detailed_statistics(df)

    # Create improved visualizations
    print("\nCreating improved regression charts...")

    # 1. Log-scale parity plot
    parity_file = create_log_scale_parity_plot(df)

    # 2. Residual analysis plot
    residual_file = create_residual_analysis_plot(df)

    # 3. Performance summary plot
    performance_file = create_performance_summary_plot(df)

    print("\n" + "="*70)
    print("IMPROVED REGRESSION CHARTS COMPLETED")
    print("="*70)
    print("Generated files:")
    print(f"1. {parity_file}")
    print(f"2. {residual_file}")
    print(f"3. {performance_file}")
    print("\nKey improvements implemented:")
    print("- Log-scale parity plot for better visualization")
    print("- Comprehensive residual analysis")
    print("- Performance metrics with detailed statistics")
    print("- Highlighted systematic underestimation patterns")
    print("- Focus on interpretability and business insights")
    print("="*70)

if __name__ == "__main__":
    main()