import matplotlib.pyplot as plt

def alpha_actualization(current_alpha, previous_loss, current_loss, best_loss, no_improvement_count,
                       alpha_increase_factor=1.1, alpha_decrease_factor=0.8,
                       min_alpha=0.001, max_alpha=0.1, patience=2):
    """
    Aktualizacja learning rate na podstawie zmiany loss

    Args:
        current_alpha: obecny learning rate
        previous_loss: poprzednia wartość loss
        current_loss: obecna wartość loss
        best_loss: najlepsza dotychczasowa wartość loss
        no_improvement_count: licznik braku poprawy
        alpha_increase_factor: czynnik zwiększania alpha
        alpha_decrease_factor: czynnik zmniejszania alpha
        min_alpha: minimalna wartość alpha
        max_alpha: maksymalna wartość alpha
        patience: ile iteracji czekać przed zmniejszeniem alpha

    Returns:
        tuple: (new_alpha, new_best_loss, new_no_improvement_count, message)
    """

    if previous_loss is None:
        return current_alpha, best_loss, no_improvement_count, None

    loss_improvement = previous_loss - current_loss
    new_alpha = current_alpha
    new_best_loss = best_loss
    new_no_improvement_count = no_improvement_count
    message = None

    if loss_improvement > 0:
        if current_loss < best_loss:
            new_best_loss = current_loss
            new_no_improvement_count = 0
            new_alpha = min(current_alpha * alpha_increase_factor, max_alpha)
            if new_alpha != current_alpha:
                message = f"  Loss improved ({previous_loss:.6f} -> {current_loss:.6f}), increasing alpha: {current_alpha:.6f} -> {new_alpha:.6f}"
        else:
            new_no_improvement_count += 1
    else:
        new_no_improvement_count += 1
        new_alpha = max(current_alpha * alpha_decrease_factor, min_alpha)
        message = f"  Loss increased ({previous_loss:.6f} -> {current_loss:.6f}), decreasing alpha: {current_alpha:.6f} -> {new_alpha:.6f}"

    if new_no_improvement_count >= patience:
        patience_alpha = max(current_alpha * alpha_decrease_factor, min_alpha)
        if patience_alpha != current_alpha:
            message = f"  No improvement for {new_no_improvement_count} iterations, reducing alpha: {current_alpha:.6f} -> {patience_alpha:.6f}"
            new_alpha = patience_alpha
        new_no_improvement_count = 0

    return new_alpha, new_best_loss, new_no_improvement_count, message

def visualization(loss_history, alpha_history, alpha_adaptation, pnp_alpha, min_alpha, max_alpha):
    """
    Wizualizacja konwergencji loss i zmian alpha

    Args:
        loss_history: historia wartości loss
        alpha_history: historia wartości alpha
        alpha_adaptation: czy używano adaptacyjnego alpha
        pnp_alpha: początkowa wartość alpha
        min_alpha: minimalna wartość alpha
        max_alpha: maksymalna wartość alpha
    """

    if len(loss_history) <= 1:
        print("Not enough data points to plot curves")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    iterations = range(1, len(loss_history) + 1)

    # Wykres Loss
    ax1.plot(iterations, loss_history, 'b-o', linewidth=2, markersize=8, label='PnP Loss')
    ax1.set_xlabel('PnP Iteration')
    ax1.set_ylabel('L1 Loss')
    title = 'PnP Inference - Loss Convergence'
    if alpha_adaptation:
        title += ' (Adaptive Alpha)'
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    for i, loss_val in enumerate(loss_history):
        ax1.annotate(f'{loss_val:.4f}',
                     (i + 1, loss_val),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')

    if len(loss_history) > 1:
        ax1.axhline(y=loss_history[0], color='r', linestyle='--', alpha=0.5,
                    label=f'Initial: {loss_history[0]:.4f}')
        ax1.axhline(y=loss_history[-1], color='g', linestyle='--', alpha=0.5,
                    label=f'Final: {loss_history[-1]:.4f}')
        ax1.legend()

    # Wykres Alpha
    if alpha_adaptation and len(alpha_history) > 1:
        ax2.plot(iterations, alpha_history, 'r-s', linewidth=2, markersize=6, label='Learning Rate (Alpha)')
        ax2.set_xlabel('PnP Iteration')
        ax2.set_ylabel('Alpha Value')
        ax2.set_title('Adaptive Learning Rate Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        for i, alpha_val in enumerate(alpha_history):
            ax2.annotate(f'{alpha_val:.4f}',
                         (i + 1, alpha_val),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center')

        ax2.axhline(y=min_alpha, color='orange', linestyle=':', alpha=0.7, label=f'Min: {min_alpha}')
        ax2.axhline(y=max_alpha, color='purple', linestyle=':', alpha=0.7, label=f'Max: {max_alpha}')
        ax2.legend()
    else:
        ax2.axhline(y=pnp_alpha, color='r', linewidth=2, label=f'Fixed Alpha: {pnp_alpha:.4f}')
        ax2.set_xlabel('PnP Iteration')
        ax2.set_ylabel('Alpha Value')
        ax2.set_title('Fixed Learning Rate')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim([pnp_alpha * 0.8, pnp_alpha * 1.2])

    plt.tight_layout()
    plt.show()




