"""
CLARA Visualization Module

Functions for visualizing results, confusion matrices, and attention weights.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from typing import List, Dict, Optional
import os


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
    cmap: str = 'Blues'
):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
        cmap: Colormap
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Normalized Frequency'}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
):
    """
    Plot training curves (loss, accuracy, F1)
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'o-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 's-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'o-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 's-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy', fontsize=11)
    axes[1].set_title('Training Accuracy', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[2].plot(epochs, history['train_f1'], 'o-', label='Train', linewidth=2)
    axes[2].plot(epochs, history['val_f1'], 's-', label='Validation', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('F1 Score', fontsize=11)
    axes[2].set_title('Training F1 Score', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training curves saved to: {save_path}")
    
    plt.show()


def plot_per_class_metrics(
    metrics: Dict[str, np.ndarray],
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    Plot per-class precision, recall, and F1 scores
    
    Args:
        metrics: Dictionary with 'precision', 'recall', 'f1' arrays
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
    """
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(x - width, metrics['precision'], width, label='Precision', alpha=0.8)
    ax.bar(x, metrics['recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + width, metrics['f1'], width, label='F1 Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Per-class metrics saved to: {save_path}")
    
    plt.show()


def plot_performance_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'f1'],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    Plot performance comparison between models
    
    Args:
        results: Dictionary mapping model names to metrics
        metrics: List of metrics to plot
        save_path: Path to save figure
        figsize: Figure size
    """
    models = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, model_name in enumerate(models):
        values = [results[model_name][metric] for metric in metrics]
        ax.bar(x + i * width, values, width, label=model_name, alpha=0.8)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Performance comparison saved to: {save_path}")
    
    plt.show()


def plot_ablation_study(
    ablation_results: Dict[str, float],
    baseline_name: str = 'Full CLARA',
    metric_name: str = 'Weighted F1 (%)',
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    Plot ablation study results
    
    Args:
        ablation_results: Dictionary mapping variant names to scores
        baseline_name: Name of the baseline (full model)
        metric_name: Name of the metric
        save_path: Path to save figure
        figsize: Figure size
    """
    variants = list(ablation_results.keys())
    scores = list(ablation_results.values())
    baseline_score = ablation_results[baseline_name]
    
    # Calculate differences from baseline
    differences = [score - baseline_score for score in scores]
    colors = ['green' if d >= 0 else 'red' for d in differences]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.barh(variants, scores, color=colors, alpha=0.7)
    ax.axvline(baseline_score, color='black', linestyle='--', linewidth=2, label='Baseline')
    
    ax.set_xlabel(metric_name, fontsize=12)
    ax.set_title('Ablation Study Results', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for i, (score, diff) in enumerate(zip(scores, differences)):
        label = f"{score:.2f}"
        if diff != 0:
            label += f" ({diff:+.2f})"
        ax.text(score + 0.5, i, label, va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Ablation study saved to: {save_path}")
    
    plt.show()


def visualize_predictions(
    images: List,
    texts: List[str],
    true_labels: List[str],
    pred_labels: List[str],
    confidences: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 10),
    max_samples: int = 6
):
    """
    Visualize sample predictions
    
    Args:
        images: List of PIL images
        texts: List of text captions
        true_labels: List of true labels
        pred_labels: List of predicted labels
        confidences: List of confidence scores
        save_path: Path to save figure
        figsize: Figure size
        max_samples: Maximum number of samples to show
    """
    n_samples = min(len(images), max_samples)
    cols = 3
    rows = (n_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for i in range(n_samples):
        ax = axes[i]
        ax.imshow(images[i])
        ax.axis('off')
        
        # Title with prediction info
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        title = f"True: {true_labels[i]}\nPred: {pred_labels[i]} ({confidences[i]:.2%})"
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        
        # Add text caption below image
        ax.text(
            0.5, -0.1, f'"{texts[i]}"',
            transform=ax.transAxes,
            ha='center',
            fontsize=8,
            style='italic',
            wrap=True
        )
    
    # Hide extra subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Predictions visualization saved to: {save_path}")
    
    plt.show()


def save_all_figures(
    history: Dict,
    results: Dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_dir: str
):
    """
    Save all visualization figures
    
    Args:
        history: Training history
        results: Evaluation results
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Training curves
    plot_training_curves(
        history,
        save_path=os.path.join(output_dir, 'training_curves.png')
    )
    
    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    # Per-class metrics
    if 'precision_per_class' in results and 'recall_per_class' in results and 'f1_per_class' in results:
        plot_per_class_metrics(
            {
                'precision': np.array(results['precision_per_class']),
                'recall': np.array(results['recall_per_class']),
                'f1': np.array(results['f1_per_class'])
            },
            class_names,
            save_path=os.path.join(output_dir, 'per_class_metrics.png')
        )
    
    print(f"\n✅ All figures saved to: {output_dir}")
