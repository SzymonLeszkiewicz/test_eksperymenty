import matplotlib.pyplot as plt
import cv2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import torch
from datetime import datetime
import numpy as np

def save_data(gt_depth, original_est, improved_est, transformed_depth=None, base_dir="depth_data"):
    """
    Save depth data to timestamped folder

    Args:
        gt_depth: Ground truth depth tensor
        original_est: Original estimation tensor
        improved_est: Improved estimation tensor
        transformed_depth: Optional transformed depth tensor
        base_dir: Base directory for saving

    Returns:
        str: Path to created folder
    """
    # Create timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, f"depth_batch_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Save each tensor
    torch.save(gt_depth, os.path.join(save_dir, "gt_depth.pt"))
    torch.save(original_est, os.path.join(save_dir, "original_est.pt"))
    torch.save(improved_est, os.path.join(save_dir, "improved_est.pt"))

    if transformed_depth is not None:
        torch.save(transformed_depth, os.path.join(save_dir, "transformed_depth.pt"))

    # Save metadata
    metadata = {
        'gt_shape': list(gt_depth.shape),
        'original_shape': list(original_est.shape),
        'improved_shape': list(improved_est.shape),
        'timestamp': timestamp,
        'has_transformed': transformed_depth is not None
    }

    if transformed_depth is not None:
        metadata['transformed_shape'] = list(transformed_depth.shape)

    torch.save(metadata, os.path.join(save_dir, "metadata.pt"))

    return save_dir


def load_data(folder_path):
    """
    Load depth data from folder

    Args:
        folder_path: Path to folder with saved data

    Returns:
        tuple: (gt_depth, original_est, improved_est, transformed_depth, metadata)
               transformed_depth is None if not saved
    """
    gt_depth = torch.load(os.path.join(folder_path, "gt_depth.pt"))
    original_est = torch.load(os.path.join(folder_path, "original_est.pt"))
    improved_est = torch.load(os.path.join(folder_path, "improved_est.pt"))
    metadata = torch.load(os.path.join(folder_path, "metadata.pt"))

    transformed_depth = None
    if metadata.get('has_transformed', False):
        transformed_depth = torch.load(os.path.join(folder_path, "transformed_depth.pt"))

    return gt_depth, original_est, improved_est, transformed_depth, metadata


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


def affine_depth_correction(gt_depth, ori_estimation, valid_mask=None,
                            sample_ratio=0.1, min_samples=1000, batch_idx=0):
    """
    Perform affine depth correction using LIDAR sparse points

    Args:
        gt_depth: Ground truth depth (LIDAR)
        ori_estimation: Original depth estimation
        valid_mask: Mask for valid LIDAR points (optional, auto-generated if None)
        sample_ratio: Ratio of points to use for fitting (for speed)
        min_samples: Minimum number of samples required
        batch_idx: Batch index for 4D tensors

    Returns:
        corrected_depth: Affine corrected depth map
        params: Transformation parameters (a, b)
        metrics: Before/after metrics
    """

    # Prepare data (2-input mode)
    gt_np, ori_np = _prepare_data(gt_depth, ori_estimation, batch_idx=batch_idx)

    # Create valid mask if not provided
    if valid_mask is None:
        valid_mask = gt_np > 0  # Assume LIDAR has 0 for invalid pixels
        print("Auto-generated valid mask from GT > 0")
    else:
        # Convert valid_mask to numpy if needed
        valid_mask = _to_numpy(valid_mask, batch_idx)

        # Resize valid_mask to match GT if needed
        if valid_mask.shape != gt_np.shape:
            valid_mask = cv2.resize(valid_mask.astype(np.uint8),
                                    (gt_np.shape[1], gt_np.shape[0])).astype(bool)

    # Extract valid pixels
    est_valid = ori_np[valid_mask]
    lidar_valid = gt_np[valid_mask]

    print(f"=== AFFINE DEPTH CORRECTION ===")
    print(f"Valid LIDAR points: {len(est_valid):,}")

    # Sampling for speed (if too many points)
    n_samples = len(est_valid)
    if n_samples > min_samples:
        sample_size = max(min_samples, int(n_samples * sample_ratio))
        indices = np.random.choice(n_samples, sample_size, replace=False)
        est_sample = est_valid[indices]
        lidar_sample = lidar_valid[indices]
        print(f"Using {sample_size:,} sampled points for fitting")
    else:
        est_sample = est_valid
        lidar_sample = lidar_valid
        print(f"Using all {n_samples:,} points for fitting")

    # Remove outliers (optional)
    depth_threshold_min = 0.5  # min 50cm
    depth_threshold_max = 100.0  # max 100m

    mask_outliers = ((est_sample > depth_threshold_min) &
                     (est_sample < depth_threshold_max) &
                     (lidar_sample > depth_threshold_min) &
                     (lidar_sample < depth_threshold_max))

    est_clean = est_sample[mask_outliers]
    lidar_clean = lidar_sample[mask_outliers]

    print(f"After outlier removal: {len(est_clean):,} points")

    # KEY PART: Fit linear transformation
    # corrected_depth = a * original_depth + b

    # Reshape for sklearn
    X = est_clean.reshape(-1, 1)  # estimation as input
    y = lidar_clean  # LIDAR as target

    # Linear regression
    reg = LinearRegression()
    reg.fit(X, y)

    # Transformation parameters
    a = reg.coef_[0]  # slope
    b = reg.intercept_  # intercept

    print(f"\nTransformation parameters:")
    print(f"a (slope): {a:.6f}")
    print(f"b (intercept): {b:.6f}")
    print(f"R²: {r2_score(y, reg.predict(X)):.4f}")

    # Apply transformation to entire depth map
    corrected_depth = a * ori_np + b

    # Calculate metrics before and after correction
    metrics = _calculate_correction_metrics(ori_np, corrected_depth, gt_np, valid_mask)

    return corrected_depth, (a, b), metrics


def _calculate_correction_metrics(original_est, corrected_est, lidar_gt, valid_mask):
    """
    Calculate metrics before and after correction
    """
    # Valid pixels
    orig_valid = original_est[valid_mask]
    corr_valid = corrected_est[valid_mask]
    lidar_valid = lidar_gt[valid_mask]

    # Metrics before correction
    mae_before = np.mean(np.abs(orig_valid - lidar_valid))
    rmse_before = np.sqrt(np.mean((orig_valid - lidar_valid) ** 2))
    bias_before = np.mean(orig_valid - lidar_valid)

    # Metrics after correction
    mae_after = np.mean(np.abs(corr_valid - lidar_valid))
    rmse_after = np.sqrt(np.mean((corr_valid - lidar_valid) ** 2))
    bias_after = np.mean(corr_valid - lidar_valid)

    # Improvements
    mae_improvement = (mae_before - mae_after) / mae_before * 100
    rmse_improvement = (rmse_before - rmse_after) / rmse_before * 100
    bias_improvement = abs(bias_before) - abs(bias_after)

    metrics = {
        'mae_before': mae_before, 'mae_after': mae_after, 'mae_improvement': mae_improvement,
        'rmse_before': rmse_before, 'rmse_after': rmse_after, 'rmse_improvement': rmse_improvement,
        'bias_before': bias_before, 'bias_after': bias_after, 'bias_improvement': bias_improvement
    }

    print(f"\n=== CORRECTION RESULTS ===")
    print(f"MAE:  {mae_before:.4f}m → {mae_after:.4f}m (improvement: {mae_improvement:+.2f}%)")
    print(f"RMSE: {rmse_before:.4f}m → {rmse_after:.4f}m (improvement: {rmse_improvement:+.2f}%)")
    print(f"Bias: {bias_before:+.4f}m → {bias_after:+.4f}m (improvement: {bias_improvement:+.4f}m)")
    print(f"========================")

    return metrics


def plot_affine_correction(gt_depth, ori_estimation, corrected_estimation, valid_mask,
                           params, figsize=(16, 18)):
    """
    Visualize affine correction results - 3x2 layout

    Args:
        gt_depth: Ground truth depth (LIDAR)
        ori_estimation: Original depth estimation
        corrected_estimation: Affine corrected estimation
        valid_mask: Mask for valid LIDAR points
        params: Transformation parameters (a, b)
        figsize: Figure size
    """

    # Prepare data (2-input mode for original)
    gt_np, ori_np = _prepare_data(gt_depth, ori_estimation, batch_idx=0)

    # Handle corrected estimation
    corr_np = _to_numpy(corrected_estimation, 0)
    if corr_np.shape != gt_np.shape:
        corr_np = cv2.resize(corr_np.astype(np.float32), (gt_np.shape[1], gt_np.shape[0]))

    # Handle valid_mask
    valid_mask = _to_numpy(valid_mask, 0)
    if valid_mask.shape != gt_np.shape:
        valid_mask = cv2.resize(valid_mask.astype(np.uint8),
                                (gt_np.shape[1], gt_np.shape[0])).astype(bool)

    a, b = params

    fig, axes = plt.subplots(3, 2, figsize=figsize)

    # Adjust spacing between plots
    plt.subplots_adjust(hspace=0.3, wspace=0.3, left=0.05, right=0.95, top=0.95, bottom=0.05)

    # === FIRST ROW - ORIGINAL vs CORRECTED ===

    # Original estimation
    im1 = axes[0, 0].imshow(ori_np, cmap='plasma')
    axes[0, 0].set_title(f'Original Estimation\nRange: {ori_np.min():.1f}-{ori_np.max():.1f}m',
                         fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8, label='Depth (m)')

    # Corrected estimation
    im2 = axes[0, 1].imshow(corr_np, cmap='plasma')
    axes[0, 1].set_title(f'Corrected Estimation\nTransform: y = {a:.3f}x + {b:.3f}',
                         fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8, label='Depth (m)')

    # === SECOND ROW - GROUND TRUTH AND IMPROVEMENT ===

    # Ground truth
    gt_display = np.copy(gt_np)
    gt_display[~valid_mask] = np.nan
    im3 = axes[1, 0].imshow(gt_display, cmap='plasma')
    axes[1, 0].set_title(f'LIDAR Ground Truth\nValid pixels: {valid_mask.sum():,}',
                         fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8, label='Depth (m)')

    # Improvement map (error difference)
    orig_error = np.abs(ori_np - gt_np)
    corr_error = np.abs(corr_np - gt_np)
    improvement_map = orig_error - corr_error  # Positive = improvement
    improvement_masked = np.copy(improvement_map)
    improvement_masked[~valid_mask] = np.nan

    max_imp = np.nanpercentile(np.abs(improvement_masked), 95)
    im4 = axes[1, 1].imshow(improvement_masked, cmap='RdYlGn', vmin=-max_imp, vmax=max_imp)
    axes[1, 1].set_title(f'Error Improvement Map\nMean improvement: {improvement_map[valid_mask].mean():.4f}m',
                         fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    cbar4 = plt.colorbar(im4, ax=axes[1, 1], shrink=0.8, label='Error Reduction (m)')
    cbar4.ax.text(0.5, -0.1, 'Green: Improved, Red: Worsened',
                  transform=cbar4.ax.transAxes, ha='center', fontsize=10)

    # === THIRD ROW - ERROR MAPS ===

    # Error before correction (Estimation - LIDAR)
    error_before = ori_np - gt_np
    error_before_masked = np.copy(error_before)
    error_before_masked[~valid_mask] = np.nan

    # Find range for symmetric scale
    max_error = max(np.nanpercentile(np.abs(error_before_masked), 95),
                    np.nanpercentile(np.abs((corr_np - gt_np)[valid_mask]), 95))

    im5 = axes[2, 0].imshow(error_before_masked, cmap='RdBu_r', vmin=-max_error, vmax=max_error)
    axes[2, 0].set_title(
        f'Error Before Correction\n(Original - LIDAR)\nMAE: {np.abs(error_before[valid_mask]).mean():.3f}m',
        fontsize=14, fontweight='bold')
    axes[2, 0].axis('off')
    cbar5 = plt.colorbar(im5, ax=axes[2, 0], shrink=0.8, label='Error (m)')
    cbar5.ax.text(0.5, -0.1, 'Red: Overestimation, Blue: Underestimation',
                  transform=cbar5.ax.transAxes, ha='center', fontsize=10)

    # Error after correction (Corrected - LIDAR)
    error_after = corr_np - gt_np
    error_after_masked = np.copy(error_after)
    error_after_masked[~valid_mask] = np.nan

    im6 = axes[2, 1].imshow(error_after_masked, cmap='RdBu_r', vmin=-max_error, vmax=max_error)

    # Calculate improvement
    mae_before = np.abs(error_before[valid_mask]).mean()
    mae_after = np.abs(error_after[valid_mask]).mean()
    improvement_pct = (mae_before - mae_after) / mae_before * 100

    axes[2, 1].set_title(
        f'Error After Correction\n(Corrected - LIDAR)\nMAE: {mae_after:.3f}m ({improvement_pct:+.1f}%)',
        fontsize=14, fontweight='bold')
    axes[2, 1].axis('off')
    cbar6 = plt.colorbar(im6, ax=axes[2, 1], shrink=0.8, label='Error (m)')
    cbar6.ax.text(0.5, -0.1, 'Red: Overestimation, Blue: Underestimation',
                  transform=cbar6.ax.transAxes, ha='center', fontsize=10)

    plt.show()


def apply_affine_correction(gt_depth, ori_estimation, valid_mask=None,
                            sample_ratio=0.1, min_samples=1000, batch_idx=0,
                            show_plot=True):
    """
    Complete affine correction pipeline - correction + visualization

    Args:
        gt_depth: Ground truth depth (LIDAR)
        ori_estimation: Original depth estimation
        valid_mask: Mask for valid LIDAR points (optional)
        sample_ratio: Ratio of points for fitting
        min_samples: Minimum samples required
        batch_idx: Batch index for 4D tensors
        show_plot: Whether to show visualization

    Returns:
        corrected_depth: Affine corrected depth map
        params: Transformation parameters (a, b)
        metrics: Before/after metrics
    """

    print("=== APPLYING AFFINE DEPTH CORRECTION ===")

    # Perform correction
    corrected_depth, params, metrics = affine_depth_correction(
        gt_depth, ori_estimation, valid_mask, sample_ratio, min_samples, batch_idx
    )

    # Show visualization if requested
    if show_plot:
        print("\nCreating visualization...")
        if valid_mask is None:
            # Recreate valid mask for plotting
            gt_np, _ = _prepare_data(gt_depth, ori_estimation, batch_idx=batch_idx)
            valid_mask = gt_np > 0

        plot_affine_correction(gt_depth, ori_estimation, corrected_depth, valid_mask, params)

    # Print final transformation
    a, b = params
    print(f"\n=== FINAL TRANSFORMATION ===")
    print(f"corrected_depth = {a:.6f} * original_depth + {b:.6f}")
    print(f"============================")

    return corrected_depth, params, metrics
