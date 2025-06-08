import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import cv2
import torch.nn as nn
from tabulate import tabulate

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from criteria import *


def _to_numpy(data, batch_idx=0):
    """Convert tensor/array to 2D numpy"""
    if hasattr(data, 'cpu'):  # PyTorch tensor
        data = data.cpu().detach().numpy()
    elif hasattr(data, 'numpy'):  # TensorFlow
        data = data.numpy()
    else:
        data = np.array(data)

    # Handle different dimensions
    if data.ndim == 4:  # (B, C, H, W)
        return data[batch_idx, 0]
    elif data.ndim == 3:
        if data.shape[0] == 1:  # (1, H, W)
            return data[0]
        elif data.shape[-1] == 1:  # (H, W, 1)
            return data.squeeze()
        else:  # (H, W, C) - assume it's (H, W, 1)
            return data.squeeze()
    else:  # (H, W)
        return data


def _prepare_data(gt_depth, ori_estimation, improved_estimation=None, batch_idx=0):
    """
    Internal function to prepare data for analysis
    Supports both 2-input and 3-input modes
    """
    # Convert to numpy
    gt_np = _to_numpy(gt_depth, batch_idx)
    ori_np = _to_numpy(ori_estimation, batch_idx)

    # Handle optional third parameter
    if improved_estimation is not None:
        imp_np = _to_numpy(improved_estimation, batch_idx)
    else:
        imp_np = None

    # Ensure all same shape (resize to GT)
    target_shape = gt_np.shape

    if ori_np.shape != target_shape:
        ori_np = cv2.resize(ori_np.astype(np.float32), (target_shape[1], target_shape[0]))

    if imp_np is not None and imp_np.shape != target_shape:
        imp_np = cv2.resize(imp_np.astype(np.float32), (target_shape[1], target_shape[0]))

    # Calculate difference if we have improved estimation
    if imp_np is not None:
        diff_np = imp_np - ori_np
        return gt_np, ori_np, imp_np, diff_np
    else:
        return gt_np, ori_np


def calculate_depth_metrics(pred, target, valid_mask=None):
    """
    Calculate comprehensive depth estimation metrics

    Args:
        pred: Predicted depth (numpy array)
        target: Ground truth depth (numpy array)
        valid_mask: Valid pixel mask (optional)

    Returns:
        dict: Dictionary with all metrics
    """

    if valid_mask is None:
        valid_mask = target > 0

    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if isinstance(valid_mask, torch.Tensor):
        valid_mask = valid_mask.cpu().numpy()

    # Extract valid pixels
    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]

    if len(pred_valid) == 0:
        return None

    # Basic error metrics
    abs_diff = np.abs(pred_valid - target_valid)
    sq_diff = (pred_valid - target_valid) ** 2
    rel_diff = abs_diff / target_valid

    # Calculate metrics
    metrics = {
        'MAE': abs_diff.mean(),
        'RMSE': np.sqrt(sq_diff.mean()),
        'MSE': sq_diff.mean(),
        'MRE': rel_diff.mean() * 100,  # Percentage
        'Bias': (pred_valid - target_valid).mean(),
        'Std': (pred_valid - target_valid).std(),

        # Threshold metrics (δ < 1.25, 1.25², 1.25³)
        'δ1': np.mean(np.maximum(pred_valid / target_valid, target_valid / pred_valid) < 1.25) * 100,
        'δ2': np.mean(np.maximum(pred_valid / target_valid, target_valid / pred_valid) < 1.25 ** 2) * 100,
        'δ3': np.mean(np.maximum(pred_valid / target_valid, target_valid / pred_valid) < 1.25 ** 3) * 100,

        # Additional metrics
        'Median_AE': np.median(abs_diff),
        'Correlation': np.corrcoef(pred_valid, target_valid)[0, 1],
        'Valid_pixels': len(pred_valid),
        'Coverage': len(pred_valid) / pred.size * 100
    }

    return metrics


# =============================================================================
# VISUALIZATION FUNCTION
# =============================================================================

def plot_depth_comparison(gt_depth, ori_estimation, improved_estimation,
                          method_name="Improved", batch_idx=0, figsize=(14, 20),
                          use_gt_scale=True, scale_percentile=99):
    """
    Plot basic depth comparison: GT, Original, Improved, Difference + Histogram

    Args:
        gt_depth: Ground truth depth
        ori_estimation: Original estimation
        improved_estimation: Improved estimation
        method_name: Name of the improvement method
        batch_idx: Batch index for 4D tensors
        figsize: Figure size
        use_gt_scale: If True, use GT depth range for all depth maps
        scale_percentile: Percentile to use for scale limits
    """

    # Prepare data (3-input mode)
    gt_np, ori_np, imp_np, diff_np = _prepare_data(gt_depth, ori_estimation, improved_estimation, batch_idx)

    # Calculate scale limits based on GT or individual maps
    if use_gt_scale:
        # Use ground truth scale for all depth visualizations
        valid_mask = gt_np > 0
        if valid_mask.sum() > 0:
            gt_valid = gt_np[valid_mask]
            vmin_depth = np.percentile(gt_valid, 100 - scale_percentile)
            vmax_depth = np.percentile(gt_valid, scale_percentile)
        else:
            # Fallback if no valid GT pixels
            vmin_depth = gt_np.min()
            vmax_depth = gt_np.max()

        print(f"Using GT-based color scale: {vmin_depth:.3f} - {vmax_depth:.3f}m (percentile: {scale_percentile}%)")
    else:
        # Individual scales will be calculated for each map
        vmin_depth = None
        vmax_depth = None
        print("Using individual color scales for each depth map")

    # Create figure with 5 subplots (4 maps + 1 histogram)
    fig = plt.figure(figsize=figsize)

    # Ground Truth
    plt.subplot(5, 1, 1)
    if use_gt_scale:
        im1 = plt.imshow(gt_np, cmap='plasma', vmin=vmin_depth, vmax=vmax_depth)
        title_suffix = f"\nScale: {vmin_depth:.3f} - {vmax_depth:.3f}m"
    else:
        im1 = plt.imshow(gt_np, cmap='plasma')
        title_suffix = f"\nRange: {gt_np.min():.3f} - {gt_np.max():.3f}m"

    plt.title(f'Ground Truth Depth{title_suffix}',
              fontsize=14, fontweight='bold')
    plt.colorbar(im1, label='Depth (m)', shrink=0.8)
    plt.axis('off')

    # Original estimation
    plt.subplot(5, 1, 2)
    if use_gt_scale:
        im2 = plt.imshow(ori_np, cmap='plasma', vmin=vmin_depth, vmax=vmax_depth)
        title_suffix = f"\nActual: {ori_np.min():.3f} - {ori_np.max():.3f}m"
    else:
        im2 = plt.imshow(ori_np, cmap='plasma')
        title_suffix = f"\nRange: {ori_np.min():.3f} - {ori_np.max():.3f}m"

    plt.title(f'Original Estimation{title_suffix}',
              fontsize=14, fontweight='bold')
    plt.colorbar(im2, label='Depth (m)', shrink=0.8)
    plt.axis('off')

    # Improved estimation
    plt.subplot(5, 1, 3)
    if use_gt_scale:
        im3 = plt.imshow(imp_np, cmap='plasma', vmin=vmin_depth, vmax=vmax_depth)
        title_suffix = f"\nActual: {imp_np.min():.3f} - {imp_np.max():.3f}m"
    else:
        im3 = plt.imshow(imp_np, cmap='plasma')
        title_suffix = f"\nRange: {imp_np.min():.3f} - {imp_np.max():.3f}m"

    plt.title(f'{method_name} Estimation{title_suffix}',
              fontsize=14, fontweight='bold')
    plt.colorbar(im3, label='Depth (m)', shrink=0.8)
    plt.axis('off')

    # Difference map (always uses its own scale for better visibility)
    plt.subplot(5, 1, 4)
    vmax_diff = abs(diff_np).max()
    im4 = plt.imshow(diff_np, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
    plt.title(f'Difference ({method_name} - Original)\nMean: {diff_np.mean():.4f}m, Std: {diff_np.std():.4f}m',
              fontsize=14, fontweight='bold')
    plt.colorbar(im4, label='Depth Difference (m)', shrink=0.8)
    plt.axis('off')

    # Histogram of differences
    plt.subplot(5, 1, 5)
    plt.hist(diff_np.flatten(), bins=100, alpha=0.7, edgecolor='black', color='skyblue')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Change')
    plt.axvline(x=diff_np.mean(), color='green', linestyle='--', linewidth=2,
                label=f'Mean: {diff_np.mean():.4f}m')
    plt.xlabel('Depth Difference (m)', fontsize=12)
    plt.ylabel('Number of Pixels', fontsize=12)
    plt.title(f'Distribution of Depth Differences ({method_name} - Original)',
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# STATISTICS FUNCTION
# =============================================================================

def depth_comparison_stats(gt_depth, ori_estimation, improved_estimation,
                           method_name="Improved", batch_idx=0, display_table=True):
    """
    Calculate and display comprehensive depth comparison statistics

    Args:
        gt_depth: Ground truth depth
        ori_estimation: Original estimation
        improved_estimation: Improved estimation
        method_name: Name of the improvement method
        batch_idx: Batch index for 4D tensors
        display_table: Whether to display formatted table

    Returns:
        dict: Dictionary with all statistics and metrics
    """

    # Prepare data
    gt_np, ori_np, imp_np, diff_np = _prepare_data(gt_depth, ori_estimation, improved_estimation, batch_idx)

    # Calculate metrics for original vs GT
    ori_metrics = calculate_depth_metrics(ori_np, gt_np)

    # Calculate metrics for improved vs GT
    imp_metrics = calculate_depth_metrics(imp_np, gt_np)

    # Calculate difference statistics (improved - original)
    diff_stats = {
        'Mean_diff': diff_np.mean(),
        'Std_diff': diff_np.std(),
        'Max_pos_diff': diff_np.max(),
        'Max_neg_diff': diff_np.min(),
        'Pixels_improved_pct': (diff_np < 0).sum() / diff_np.size * 100,
        'Pixels_worsened_pct': (diff_np > 0).sum() / diff_np.size * 100,
        'Pixels_unchanged_pct': (diff_np == 0).sum() / diff_np.size * 100
    }

    # Calculate improvements
    improvements = {}
    if ori_metrics and imp_metrics:
        for metric in ['MAE', 'RMSE', 'MSE', 'MRE']:
            if metric in ori_metrics and metric in imp_metrics:
                absolute_imp = ori_metrics[metric] - imp_metrics[metric]
                relative_imp = (absolute_imp / ori_metrics[metric]) * 100 if ori_metrics[metric] != 0 else 0
                improvements[f'{metric}_improvement'] = absolute_imp
                improvements[f'{metric}_improvement_pct'] = relative_imp

    # Combine all results
    results = {
        'method_name': method_name,
        'original_metrics': ori_metrics,
        'improved_metrics': imp_metrics,
        'difference_stats': diff_stats,
        'improvements': improvements
    }

    if display_table:
        _display_comparison_table(results)

    return results


def _display_comparison_table(results):
    """
    Display formatted comparison table using tabulate
    """
    method_name = results['method_name']
    ori_metrics = results['original_metrics']
    imp_metrics = results['improved_metrics']
    improvements = results['improvements']
    diff_stats = results['difference_stats']

    print(f"\n{'=' * 80}")
    print(f"DEPTH ESTIMATION COMPARISON: Original vs {method_name}")
    print(f"{'=' * 80}")

    if ori_metrics and imp_metrics:
        # Prepare data for the main metrics table
        table_data = []
        headers = ['Metric', 'Original', method_name, 'Change', 'Improvement (%)']

        # Define metrics to display with their formatting
        metrics_to_show = [
            ('MAE (m)', 'MAE', '.4f'),
            ('RMSE (m)', 'RMSE', '.4f'),
            ('MSE (m²)', 'MSE', '.6f'),
            ('MRE (%)', 'MRE', '.2f'),
            ('Bias (m)', 'Bias', '.4f'),
            ('Std (m)', 'Std', '.4f'),
            ('Median AE (m)', 'Median_AE', '.4f'),
            ('δ < 1.25 (%)', 'δ1', '.2f'),
            ('δ < 1.25² (%)', 'δ2', '.2f'),
            ('δ < 1.25³ (%)', 'δ3', '.2f'),
            ('Correlation', 'Correlation', '.4f'),
            ('Valid Pixels', 'Valid_pixels', ',d'),
            ('Coverage (%)', 'Coverage', '.2f')
        ]

        for display_name, metric_key, fmt in metrics_to_show:
            if metric_key in ori_metrics and metric_key in imp_metrics:
                ori_val = ori_metrics[metric_key]
                imp_val = imp_metrics[metric_key]

                # Calculate improvement and determine direction
                if metric_key in ['MAE', 'RMSE', 'MSE', 'MRE', 'Bias', 'Std', 'Median_AE']:
                    # Lower is better
                    improvement = ori_val - imp_val
                    improvement_pct = (improvement / ori_val * 100) if ori_val != 0 else 0
                    improvement_symbol = "↓" if improvement > 0 else "↑" if improvement < 0 else "→"
                elif metric_key in ['δ1', 'δ2', 'δ3', 'Correlation']:
                    # Higher is better
                    improvement = imp_val - ori_val
                    improvement_pct = (improvement / ori_val * 100) if ori_val != 0 else 0
                    improvement_symbol = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
                else:
                    improvement = imp_val - ori_val
                    improvement_pct = 0
                    improvement_symbol = "→"

                # Format the row data
                row = [
                    display_name,
                    f"{ori_val:{fmt}}",
                    f"{imp_val:{fmt}}",
                    f"{improvement_symbol} {improvement:+{fmt.replace(',d', '.3f')}}",
                    f"{improvement_pct:+.1f}%"
                ]
                table_data.append(row)

        # Display the main metrics table using tabulate
        print(tabulate(table_data, headers=headers, tablefmt='grid', stralign='center'))

        # Display difference analysis table
        print(f"\n{'-' * 80}")
        print("DIFFERENCE ANALYSIS (Improved - Original):")
        print(f"{'-' * 80}")

        diff_data = [
            ['Mean difference', f"{diff_stats['Mean_diff']:+.4f}m"],
            ['Standard deviation', f"{diff_stats['Std_diff']:.4f}m"],
            ['Max improvement', f"{-diff_stats['Max_neg_diff']:.4f}m"],
            ['Max degradation', f"{diff_stats['Max_pos_diff']:.4f}m"],
            ['Pixels improved', f"{diff_stats['Pixels_improved_pct']:.2f}%"],
            ['Pixels worsened', f"{diff_stats['Pixels_worsened_pct']:.2f}%"],
            ['Pixels unchanged', f"{diff_stats['Pixels_unchanged_pct']:.2f}%"]
        ]

        diff_headers = ['Statistic', 'Value']
        print(tabulate(diff_data, headers=diff_headers, tablefmt='grid'))

        # Display summary
        print(f"\n{'-' * 80}")
        print("SUMMARY:")
        print(f"{'-' * 80}")

        mae_imp = improvements.get('MAE_improvement_pct', 0)
        rmse_imp = improvements.get('RMSE_improvement_pct', 0)

        # Create summary table
        summary_data = []
        if mae_imp > 0 and rmse_imp > 0:
            summary_data.append([f"✅ {method_name} shows IMPROVEMENT"])
        elif mae_imp < 0 and rmse_imp < 0:
            summary_data.append([f"❌ {method_name} shows DEGRADATION"])
        else:
            summary_data.append([f"⚖️  {method_name} shows MIXED results"])

        summary_data.extend([
            [f"MAE improvement: {mae_imp:+.2f}%"],
            [f"RMSE improvement: {rmse_imp:+.2f}%"]
        ])

        print(tabulate(summary_data, tablefmt='simple'))

    print(f"{'=' * 80}")


# =============================================================================
# ADDITIONAL FUNCTION FOR EXPORTING TABLES TO LATEX
# =============================================================================

def export_comparison_to_latex(results, filename="depth_comparison.tex"):
    """
    Export comparison results to LaTeX table format for thesis

    Args:
        results: Results dictionary from depth_comparison_stats
        filename: Output filename for LaTeX table
    """
    method_name = results['method_name']
    ori_metrics = results['original_metrics']
    imp_metrics = results['improved_metrics']

    if not ori_metrics or not imp_metrics:
        print("No metrics available for LaTeX export")
        return

    # Prepare data for LaTeX table
    latex_data = []

    metrics_to_show = [
        ('MAE (m)', 'MAE', '.4f'),
        ('RMSE (m)', 'RMSE', '.4f'),
        ('MRE (\\%)', 'MRE', '.2f'),
        ('$\\delta < 1.25$ (\\%)', 'δ1', '.2f'),
        ('$\\delta < 1.25^2$ (\\%)', 'δ2', '.2f'),
        ('$\\delta < 1.25^3$ (\\%)', 'δ3', '.2f'),
        ('Correlation', 'Correlation', '.4f'),
    ]

    for display_name, metric_key, fmt in metrics_to_show:
        if metric_key in ori_metrics and metric_key in imp_metrics:
            ori_val = ori_metrics[metric_key]
            imp_val = imp_metrics[metric_key]

            # Calculate improvement percentage
            if metric_key in ['MAE', 'RMSE', 'MRE']:
                improvement_pct = ((ori_val - imp_val) / ori_val * 100) if ori_val != 0 else 0
            else:  # δ metrics and Correlation
                improvement_pct = ((imp_val - ori_val) / ori_val * 100) if ori_val != 0 else 0

            latex_data.append([
                display_name,
                f"{ori_val:{fmt}}",
                f"{imp_val:{fmt}}",
                f"{improvement_pct:+.1f}\\%"
            ])

    # Generate LaTeX table
    latex_table = tabulate(
        latex_data,
        headers=['Metric', 'Original', method_name, 'Improvement'],
        tablefmt='latex_booktabs',
        floatfmt='.4f'
    )

    # Save to file
    with open(filename, 'w') as f:
        f.write("% LaTeX table for depth estimation comparison\n")
        f.write("% Generated automatically from depth comparison analysis\n\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Depth Estimation Comparison Results}\n")
        f.write("\\label{tab:depth_comparison}\n")
        f.write(latex_table)
        f.write("\n\\end{table}\n")

    print(f"LaTeX table exported to {filename}")


"""
utils.py - Comprehensive utilities for depth estimation analysis

Functions for analyzing depth estimation improvements with LIDAR data fusion.
Each function takes: GT_depth, original_estimation, improved_estimation

Usage:
    from utils import plot_depth_comparison, plot_improvement_analysis, plot_detailed_analysis
    from utils import visualize_depth_comparison, apply_affine_correction, analyze_depth_improvement

    # Basic comparison
    plot_depth_comparison(gt_depth, ori_estimation, pnp_estimation, method_name="PnP")

    # Simple visualization (for 2 inputs)
    visualize_depth_comparison(gt_depth, depth_estimation, image=rgb_image)

    # Improvement analysis
    plot_improvement_analysis(gt_depth, ori_estimation, pnp_estimation, method_name="PnP")

    # Detailed regional analysis
    plot_detailed_analysis(gt_depth, ori_estimation, pnp_estimation, method_name="PnP")

    # Complete analysis pipeline
    analyze_depth_improvement(gt_depth, ori_estimation, improved_estimation, method_name="PnP")

    # Affine correction with LIDAR
    corrected_depth, params, metrics = apply_affine_correction(gt_depth, ori_estimation)
"""


def plot_improvement_analysis(gt_depth, ori_estimation, improved_estimation,
                              method_name="Improved", batch_idx=0, improvement_threshold=0.01,
                              figsize=(16, 14)):
    """
    Plot improvement analysis: Quality change classification and error maps

    Args:
        gt_depth: Ground truth depth
        ori_estimation: Original estimation
        improved_estimation: Improved estimation
        method_name: Name of the improvement method
        batch_idx: Batch index for 4D tensors
        improvement_threshold: Threshold for significant change (meters)
        figsize: Figure size
    """

    # Prepare data (3-input mode)
    gt_np, ori_np, imp_np, diff_np = _prepare_data(gt_depth, ori_estimation, improved_estimation, batch_idx)

    # Calculate errors
    ori_error = abs(ori_np - gt_np)
    imp_error = abs(imp_np - gt_np)
    improvement_magnitude = ori_error - imp_error

    # Create improvement masks
    improved_mask = improvement_magnitude > improvement_threshold
    worsened_mask = improvement_magnitude < -improvement_threshold
    unchanged_mask = ~(improved_mask | worsened_mask)

    # Create improvement map
    improvement_map = np.zeros_like(ori_np, dtype=int)
    improvement_map[improved_mask] = 2  # Improved (green)
    improvement_map[worsened_mask] = 1  # Worsened (red)
    improvement_map[unchanged_mask] = 0  # Unchanged (gray)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Pixel Quality Change Classification
    colors = ['#808080', '#FF6B6B', '#4ECDC4']  # Gray, Red, Cyan
    cmap_discrete = ListedColormap(colors)
    im1 = axes[0, 0].imshow(improvement_map, cmap=cmap_discrete, vmin=0, vmax=2)
    axes[0, 0].set_title(f'Pixel Quality Change Classification\n'
                         f'Improved: {improved_mask.sum():,} ({improved_mask.sum() / improvement_map.size * 100:.1f}%)\n'
                         f'Worsened: {worsened_mask.sum():,} ({worsened_mask.sum() / improvement_map.size * 100:.1f}%)\n'
                         f'Unchanged: {unchanged_mask.sum():,} ({unchanged_mask.sum() / improvement_map.size * 100:.1f}%)',
                         fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Custom colorbar
    cbar1 = plt.colorbar(im1, ax=axes[0, 0], shrink=0.4, ticks=[0, 1, 2])
    cbar1.ax.set_yticklabels(['Unchanged', 'Worsened', 'Improved'])

    # 2. Magnitude of Error Change
    im2 = axes[0, 1].imshow(improvement_magnitude, cmap='RdYlGn',
                            vmin=-abs(improvement_magnitude).max(),
                            vmax=abs(improvement_magnitude).max())
    axes[0, 1].set_title(f'Magnitude of Error Change\n'
                         f'Mean improvement: {improvement_magnitude.mean():.4f}m\n'
                         f'Max improvement: {improvement_magnitude.max():.4f}m',
                         fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.4, label='Error Reduction (m)')

    # 3. Original Model Error
    im3 = axes[1, 0].imshow(ori_error, cmap='Reds')
    axes[1, 0].set_title(f'Original Model Error\nMAE: {ori_error.mean():.4f}m',
                         fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.4, label='Absolute Error (m)')

    # 4. Improved Model Error
    im4 = axes[1, 1].imshow(imp_error, cmap='Reds')
    mae_improvement_pct = (ori_error.mean() - imp_error.mean()) / ori_error.mean() * 100
    axes[1, 1].set_title(f'{method_name} Model Error\nMAE: {imp_error.mean():.4f}m ({mae_improvement_pct:+.1f}%)',
                         fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.4, label='Absolute Error (m)')

    plt.tight_layout()
    plt.show()

    # Print detailed statistics
    print(f"\n=== DETAILED IMPROVEMENT ANALYSIS ({method_name}) ===")
    print(f"Threshold for significant change: {improvement_threshold:.3f}m")
    print(f"\nPixel Classification:")
    print(f"  Improved pixels: {improved_mask.sum():,} ({improved_mask.sum() / improvement_map.size * 100:.2f}%)")
    print(f"  Worsened pixels: {worsened_mask.sum():,} ({worsened_mask.sum() / improvement_map.size * 100:.2f}%)")
    print(f"  Unchanged pixels: {unchanged_mask.sum():,} ({unchanged_mask.sum() / improvement_map.size * 100:.2f}%)")

    if improved_mask.sum() > 0:
        print(f"\nImprovement Statistics:")
        print(f"  Mean improvement (improved pixels): {improvement_magnitude[improved_mask].mean():.6f}m")
        print(f"  Max improvement: {improvement_magnitude[improved_mask].max():.6f}m")

    if worsened_mask.sum() > 0:
        print(f"  Mean worsening (worsened pixels): {-improvement_magnitude[worsened_mask].mean():.6f}m")
        print(f"  Max worsening: {-improvement_magnitude[worsened_mask].min():.6f}m")

    print("=" * 60)


def plot_detailed_analysis(gt_depth, ori_estimation, improved_estimation,
                           method_name="Improved", batch_idx=0, improvement_threshold=0.01,
                           figsize=(24, 6)):
    """
    Plot detailed regional analysis with three comprehensive charts:
    1. Improvement by distance ranges (bar chart)
    2. Error distribution comparison (histogram)
    3. Depth distribution comparison (NEW! GT vs Original vs Improved)

    Args:
        gt_depth: Ground truth depth
        ori_estimation: Original estimation
        improved_estimation: Improved estimation
        method_name: Name of the improvement method
        batch_idx: Batch index for 4D tensors
        improvement_threshold: Threshold for significant change (meters)
        figsize: Figure size (increased to accommodate 3 charts)
    """

    # Prepare data (3-input mode)
    gt_np, ori_np, imp_np, diff_np = _prepare_data(gt_depth, ori_estimation, improved_estimation, batch_idx)

    # Calculate errors and improvement
    ori_error = abs(ori_np - gt_np)
    imp_error = abs(imp_np - gt_np)
    improvement_magnitude = ori_error - imp_error

    # Create improvement masks
    improved_mask = improvement_magnitude > improvement_threshold
    worsened_mask = improvement_magnitude < -improvement_threshold

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Improvement by Distance Ranges (more granular)
    distance_ranges = [
        (0, 5, '0-5m'),
        (5, 10, '5-10m'),
        (10, 15, '10-15m'),
        (15, 25, '15-25m'),
        (25, 40, '25-40m'),
        (40, 100, '40m+')
    ]

    improvement_by_range = []
    worsening_by_range = []
    range_labels = []

    for min_dist, max_dist, label in distance_ranges:
        if max_dist == 100:  # Last range is 40m+
            range_mask = gt_np >= min_dist
        else:
            range_mask = (gt_np >= min_dist) & (gt_np < max_dist)

        if range_mask.sum() > 0:
            improved_in_range = (improved_mask & range_mask).sum() / range_mask.sum() * 100
            worsened_in_range = (worsened_mask & range_mask).sum() / range_mask.sum() * 100
            pixel_count = range_mask.sum()
        else:
            improved_in_range = 0
            worsened_in_range = 0
            pixel_count = 0

        improvement_by_range.append(improved_in_range)
        worsening_by_range.append(worsened_in_range)
        range_labels.append(f'{label}\n({pixel_count:,} px)')

    x = np.arange(len(distance_ranges))
    width = 0.35

    bars1 = axes[0].bar(x - width / 2, improvement_by_range, width, label='Improved', color='green', alpha=0.7)
    bars2 = axes[0].bar(x + width / 2, worsening_by_range, width, label='Worsened', color='red', alpha=0.7)

    # Add percentage labels on bars
    for i, (imp, wors) in enumerate(zip(improvement_by_range, worsening_by_range)):
        if imp > 1:  # Only show if > 1%
            axes[0].text(x[i] - width / 2, imp + 0.5, f'{imp:.1f}%', ha='center', va='bottom', fontsize=9)
        if wors > 1:  # Only show if > 1%
            axes[0].text(x[i] + width / 2, wors + 0.5, f'{wors:.1f}%', ha='center', va='bottom', fontsize=9)

    axes[0].set_ylabel('Percentage of Pixels (%)', fontsize=12)
    axes[0].set_xlabel('Distance Range', fontsize=12)
    axes[0].set_title(f'Improvement by Distance Range\n({method_name} vs Original)', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(range_labels, fontsize=10)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # 2. Error Distribution Comparison (Before vs After)
    valid_mask = gt_np > 0
    ori_error_valid = ori_error[valid_mask]
    imp_error_valid = imp_error[valid_mask]

    # Create bins for histogram
    max_error = max(np.percentile(ori_error_valid, 95), np.percentile(imp_error_valid, 95))
    bins = np.linspace(0, max_error, 50)

    axes[1].hist(ori_error_valid, bins=bins, alpha=0.6, color='red',
                 label=f'Original (MAE: {ori_error_valid.mean():.3f}m)', density=True)
    axes[1].hist(imp_error_valid, bins=bins, alpha=0.6, color='blue',
                 label=f'{method_name} (MAE: {imp_error_valid.mean():.3f}m)', density=True)

    # Add median lines
    axes[1].axvline(np.median(ori_error_valid), color='red', linestyle='--', alpha=0.8,
                    label=f'Original median: {np.median(ori_error_valid):.3f}m')
    axes[1].axvline(np.median(imp_error_valid), color='blue', linestyle='--', alpha=0.8,
                    label=f'{method_name} median: {np.median(imp_error_valid):.3f}m')

    axes[1].set_xlabel('Absolute Error (m)', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title(f'Error Distribution Comparison\n(Valid pixels: {valid_mask.sum():,})', fontsize=14,
                      fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # 3. Depth Distribution Comparison (GT vs Original vs Improved) - IMPROVED VISIBILITY
    valid_mask = gt_np > 0
    gt_valid = gt_np[valid_mask]
    ori_valid = ori_np[valid_mask]
    imp_valid = imp_np[valid_mask]

    # Find common depth range for all three distributions
    min_depth = min(gt_valid.min(), ori_valid.min(), imp_valid.min())
    max_depth = max(np.percentile(gt_valid, 95),
                    np.percentile(ori_valid, 95),
                    np.percentile(imp_valid, 95))

    # Create bins for depth histogram
    depth_bins = np.linspace(min_depth, max_depth, 50)

    # Calculate histograms manually for better control
    gt_hist, _ = np.histogram(gt_valid, bins=depth_bins, density=True)
    ori_hist, _ = np.histogram(ori_valid, bins=depth_bins, density=True)
    imp_hist, _ = np.histogram(imp_valid, bins=depth_bins, density=True)

    # Bin centers for plotting
    bin_centers = (depth_bins[:-1] + depth_bins[1:]) / 2

    # Use high-contrast colors and step plots
    colors = ['#FF6B35', '#004E89', '#00C851']  # Orange, Navy Blue, Green
    line_styles = ['-', '--', '-.']

    # Plot as step curves with different styles
    axes[2].plot(bin_centers, gt_hist, color=colors[0], linewidth=2.5, linestyle=line_styles[0],
                 label=f'Ground Truth (μ: {gt_valid.mean():.2f}m)')
    axes[2].plot(bin_centers, ori_hist, color=colors[1], linewidth=2.5, linestyle=line_styles[1],
                 label=f'Original (μ: {ori_valid.mean():.2f}m)')
    axes[2].plot(bin_centers, imp_hist, color=colors[2], linewidth=2.5, linestyle=line_styles[2],
                 label=f'{method_name} (μ: {imp_valid.mean():.2f}m)')

    # Optional: Add very light fill for better visibility
    axes[2].fill_between(bin_centers, gt_hist, alpha=0.15, color=colors[0])
    axes[2].fill_between(bin_centers, ori_hist, alpha=0.15, color=colors[1])
    axes[2].fill_between(bin_centers, imp_hist, alpha=0.15, color=colors[2])

    # Add median lines with matching colors and styles
    axes[2].axvline(np.median(gt_valid), color=colors[0], linestyle=':', linewidth=2, alpha=0.8,
                    label=f'GT median: {np.median(gt_valid):.2f}m')
    axes[2].axvline(np.median(ori_valid), color=colors[1], linestyle=':', linewidth=2, alpha=0.8,
                    label=f'Orig median: {np.median(ori_valid):.2f}m')
    axes[2].axvline(np.median(imp_valid), color=colors[2], linestyle=':', linewidth=2, alpha=0.8,
                    label=f'{method_name} median: {np.median(imp_valid):.2f}m')

    axes[2].set_xlabel('Depth (m)', fontsize=12)
    axes[2].set_ylabel('Density', fontsize=12)
    axes[2].set_title(f'Depth Distribution Comparison\n(Valid pixels: {valid_mask.sum():,})', fontsize=14,
                      fontweight='bold')
    axes[2].legend(fontsize=9, loc='upper right', framealpha=0.9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print detailed statistics
    print(f"\n=== DETAILED ANALYSIS ({method_name}) ===")
    print(f"Threshold for significant change: {improvement_threshold:.3f}m")

    # Overall improvement statistics
    mae_before = ori_error_valid.mean()
    mae_after = imp_error_valid.mean()
    mae_improvement = mae_before - mae_after

    rmse_before = np.sqrt((ori_error_valid ** 2).mean())
    rmse_after = np.sqrt((imp_error_valid ** 2).mean())
    rmse_improvement = rmse_before - rmse_after

    print(f"\nOverall Performance:")
    print(f"  MAE improvement: {mae_improvement:.4f}m ({mae_before:.4f}m → {mae_after:.4f}m)")
    print(f"  RMSE improvement: {rmse_improvement:.4f}m ({rmse_before:.4f}m → {rmse_after:.4f}m)")
    print(f"  Pixels improved: {improved_mask.sum():,} ({improved_mask.sum() / gt_np.size * 100:.1f}%)")
    print(f"  Pixels worsened: {worsened_mask.sum():,} ({worsened_mask.sum() / gt_np.size * 100:.1f}%)")

    # Depth distribution statistics
    print(f"\nDepth Distribution Statistics:")
    print(
        f"  Ground Truth    - Mean: {gt_valid.mean():.3f}m, Median: {np.median(gt_valid):.3f}m, Std: {gt_valid.std():.3f}m")
    print(
        f"  Original        - Mean: {ori_valid.mean():.3f}m, Median: {np.median(ori_valid):.3f}m, Std: {ori_valid.std():.3f}m")
    print(
        f"  {method_name:14s} - Mean: {imp_valid.mean():.3f}m, Median: {np.median(imp_valid):.3f}m, Std: {imp_valid.std():.3f}m")

    # Depth range analysis
    print(f"\nDepth Range Analysis:")
    print(f"  Ground Truth    - Range: {gt_valid.min():.2f}m - {gt_valid.max():.2f}m")
    print(f"  Original        - Range: {ori_valid.min():.2f}m - {ori_valid.max():.2f}m")
    print(f"  {method_name:14s} - Range: {imp_valid.min():.2f}m - {imp_valid.max():.2f}m")
    print("=" * 60)





def visualize_depth_comparison(gt_depth, depth_estimation, corrected_estimation=None, image=None, batch_idx=0,
                               figsize=(16, 12), use_gt_scale=True, scale_percentile=99, user_max = 5, user_min = -5):
    """
    Wizualizacja porównania GT depth vs estimation

    Args:
        gt_depth: Ground truth depth
        depth_estimation: Estimated depth
        image: Optional RGB image
        batch_idx: Batch index for 4D tensors
        figsize: Figure size
        use_gt_scale: If True, use GT depth range for both depth maps; if False, use individual scales
        scale_percentile: Percentile to use for scale limits (default: 99 to avoid outliers)
    """

    # Prepare data (2-input mode)
    gt_np, est_np = _prepare_data(gt_depth, depth_estimation, batch_idx=batch_idx)
    # Valid mask
    valid_mask = gt_np > 0

    # Calculate difference
    diff_np = est_np - gt_np

    # Calculate scale limits based on GT or individual maps
    if use_gt_scale:
        # Use ground truth scale for both depth visualizations
        if valid_mask.sum() > 0:
            gt_valid = gt_np[valid_mask]
            vmin_depth = np.percentile(gt_valid, 100 - scale_percentile)
            vmax_depth = np.percentile(gt_valid, scale_percentile)
        else:
            # Fallback if no valid GT pixels
            vmin_depth = gt_np.min()
            vmax_depth = gt_np.max()

        print(f"Using GT-based color scale: {vmin_depth:.3f} - {vmax_depth:.3f}m (percentile: {scale_percentile}%)")
    else:
        # Individual scales will be calculated for each map
        vmin_depth = None
        vmax_depth = None
        print("Using individual color scales for each depth map")

    # Create subplots
    if image is not None:
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # RGB Image
        img_np = image[0].cpu().detach().numpy()
        if img_np.ndim == 3 and img_np.shape[0] == 3:  # CHW to HWC
            img_np = np.transpose(img_np, (1, 2, 0))
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('RGB Image', fontweight='bold')
        axes[0, 0].axis('off')

        # GT Depth
        gt_display = np.copy(gt_np)
        gt_display[~valid_mask] = np.nan

        if use_gt_scale:
            im1 = axes[0, 1].imshow(gt_display, cmap='plasma', vmin=vmin_depth, vmax=vmax_depth)
            title_suffix = f'Scale: {vmin_depth:.1f}-{vmax_depth:.1f}m'
        else:
            im1 = axes[0, 1].imshow(gt_display, cmap='plasma')
            title_suffix = f'{gt_np[valid_mask].min():.1f}-{gt_np[valid_mask].max():.1f}m'

        axes[0, 1].set_title(f'Ground Truth\n{title_suffix}', fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], shrink=0.4)

        # Estimation
        if use_gt_scale:
            im2 = axes[1, 0].imshow(est_np, cmap='plasma', vmin=vmin_depth, vmax=vmax_depth)
            title_suffix = f'Actual: {est_np.min():.1f}-{est_np.max():.1f}m'
        else:
            im2 = axes[1, 0].imshow(est_np, cmap='plasma')
            title_suffix = f'{est_np.min():.1f}-{est_np.max():.1f}m'

        axes[1, 0].set_title(f'Estimation\n{title_suffix}', fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], shrink=0.4)

        # Difference (always uses its own scale for better visibility)
        diff_masked = np.copy(diff_np)
        diff_masked[~valid_mask] = np.nan
        vmax = np.nanpercentile(np.abs(diff_masked), 95)
        im3 = axes[1, 1].imshow(diff_masked, cmap='RdBu_r', vmin=user_min, vmax=user_max)
        axes[1, 1].set_title(f'Difference (Est-GT)\nMean: {diff_np[valid_mask].mean():.3f}m', fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1], shrink=0.4)

    else:
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # GT Depth
        gt_display = np.copy(gt_np)
        gt_display[~valid_mask] = np.nan

        if use_gt_scale:
            im1 = axes[0].imshow(gt_display, cmap='plasma', vmin=vmin_depth, vmax=vmax_depth)
            title_suffix = f'Scale: {vmin_depth:.1f}-{vmax_depth:.1f}m'
        else:
            im1 = axes[0].imshow(gt_display, cmap='plasma')
            title_suffix = f'{gt_np[valid_mask].min():.1f}-{gt_np[valid_mask].max():.1f}m'

        axes[0].set_title(f'Ground Truth\n{title_suffix}', fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)

        # Estimation
        if use_gt_scale:
            im2 = axes[1].imshow(est_np, cmap='plasma', vmin=vmin_depth, vmax=vmax_depth)
            title_suffix = f'Actual: {est_np.min():.1f}-{est_np.max():.1f}m'
        else:
            im2 = axes[1].imshow(est_np, cmap='plasma')
            title_suffix = f'{est_np.min():.1f}-{est_np.max():.1f}m'

        axes[1].set_title(f'Estimation\n{title_suffix}', fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)

        # Difference (always uses its own scale for better visibility)
        diff_masked = np.copy(diff_np)
        diff_masked[~valid_mask] = np.nan
        vmax = np.nanpercentile(np.abs(diff_masked), 95)
        im3 = axes[2].imshow(diff_masked, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[2].set_title(f'Difference (Est-GT)\nMean: {diff_np[valid_mask].mean():.3f}m', fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    plt.show()
    print("MAXXXXX", vmax)

    # ><><><><><><><><><><><>><><><><><><><><><><><><>
    # Improvement map (error difference)
    if corrected_estimation is not None:
        corr_np = _to_numpy(corrected_estimation, 0)
        if corr_np.shape != gt_np.shape:
            corr_np = cv2.resize(corr_np.astype(np.float32), (gt_np.shape[1], gt_np.shape[0]))

        orig_error = np.abs(est_np - gt_np)
        corr_error = np.abs(corr_np - gt_np)
        improvement_map = orig_error - corr_error  # Positive = improvement
        improvement_masked = np.copy(improvement_map)
        improvement_masked[~valid_mask] = np.nan

        max_imp = np.nanpercentile(np.abs(improvement_masked), 95)
        plt.imshow(improvement_masked, cmap='RdYlGn', vmin=-max_imp, vmax=max_imp)
        plt.title(f'Error Improvement Map\nMean improvement: {improvement_map[valid_mask].mean():.4f}m',
                             fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.colorbar(shrink=0.4)
        plt.show()
        # ><><><><><><><><><><><>><><><><><><><><><><><><>


    # Metrics
    gt_valid = gt_np[valid_mask]
    est_valid = est_np[valid_mask]

    mae = np.mean(np.abs(est_valid - gt_valid))
    rmse = np.sqrt(np.mean((est_valid - gt_valid) ** 2))
    bias = np.mean(est_valid - gt_valid)

    print(f"\n=== DEPTH COMPARISON METRICS ===")
    if use_gt_scale:
        print(f"Color scale mode: GT-based ({vmin_depth:.3f} - {vmax_depth:.3f}m)")
        # Check if estimation values are outside the GT scale
        est_clipped = np.sum((est_np < vmin_depth) | (est_np > vmax_depth))
        if est_clipped > 0:
            print(f"Estimation pixels outside GT scale: {est_clipped:,} ({est_clipped / est_np.size * 100:.1f}%)")
    else:
        print(f"Color scale mode: Individual scales")

    print(f"MAE: {mae:.4f}m")
    print(f"RMSE: {rmse:.4f}m")
    print(f"Bias: {bias:.4f}m")
    print(f"Valid pixels: {valid_mask.sum():,}/{gt_np.size:,} ({valid_mask.sum() / gt_np.size * 100:.1f}%)")

    if use_gt_scale:
        print(f"\nScale Information:")
        print(f"  GT scale used: {vmin_depth:.3f} - {vmax_depth:.3f}m")
        print(f"  GT actual range: {gt_np[valid_mask].min():.3f} - {gt_np[valid_mask].max():.3f}m")
        print(f"  Estimation actual range: {est_np.min():.3f} - {est_np.max():.3f}m")

    print("=" * 35)





