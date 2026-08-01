"""
CLARA Utility Functions

Common utility functions for CLARA project.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get available device (CUDA or CPU)
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU (GPU not available)")
    
    return device


def count_parameters(model: nn.Module, trainable_only: bool = False) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
    
    Returns:
        Dictionary with parameter counts
    """
    if trainable_only:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable = total
    else:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    frozen = total - trainable
    trainable_pct = 100 * trainable / total if total > 0 else 0
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
        'trainable_percentage': trainable_pct
    }


def print_model_summary(model: nn.Module):
    """
    Print model architecture summary
    
    Args:
        model: PyTorch model
    """
    params = count_parameters(model)
    
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Total Parameters:      {params['total']:,}")
    print(f"Trainable Parameters:  {params['trainable']:,}")
    print(f"Frozen Parameters:     {params['frozen']:,}")
    print(f"Trainable Percentage:  {params['trainable_percentage']:.2f}%")
    print("=" * 60)
    
    # Print layer-wise parameters
    print("\nLayer-wise Parameters:")
    print("-" * 60)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name:50s} {param.numel():>10,} (trainable)")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    save_path: str,
    **kwargs
):
    """
    Save training checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        best_metric: Best metric value
        save_path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        **kwargs
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"✅ Checkpoint saved to: {save_path}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Load training checkpoint
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        optimizer: Optional optimizer to load state
        device: Device to load checkpoint on
    
    Returns:
        Checkpoint dictionary
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✅ Checkpoint loaded from: {checkpoint_path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Best Metric: {checkpoint.get('best_metric', 'N/A')}")
    
    return checkpoint


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets
    
    Args:
        labels: Array of labels
        num_classes: Number of classes
    
    Returns:
        Class weights tensor
    """
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    
    # Inverse frequency weighting
    class_weights = total_samples / (num_classes * class_counts)
    
    return torch.FloatTensor(class_weights)


def create_output_dirs(base_dir: str) -> Dict[str, str]:
    """
    Create output directory structure
    
    Args:
        base_dir: Base output directory
    
    Returns:
        Dictionary with output directory paths
    """
    dirs = {
        'base': base_dir,
        'checkpoints': os.path.join(base_dir, 'checkpoints'),
        'logs': os.path.join(base_dir, 'logs'),
        'results': os.path.join(base_dir, 'results'),
        'figures': os.path.join(base_dir, 'figures')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"✅ Output directories created at: {base_dir}")
    
    return dirs


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get GPU memory usage information
    
    Returns:
        Dictionary with memory info in GB
    """
    if not torch.cuda.is_available():
        return {}
    
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'total_gb': total,
        'free_gb': total - reserved
    }


def print_gpu_memory():
    """Print current GPU memory usage"""
    if not torch.cuda.is_available():
        print("No GPU available")
        return
    
    memory = get_gpu_memory_info()
    print("\nGPU Memory Usage:")
    print(f"  Allocated: {memory['allocated_gb']:.2f} GB")
    print(f"  Reserved:  {memory['reserved_gb']:.2f} GB")
    print(f"  Total:     {memory['total_gb']:.2f} GB")
    print(f"  Free:      {memory['free_gb']:.2f} GB")


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current metric score
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def validate_config(config: Dict) -> bool:
    """
    Validate configuration dictionary
    
    Args:
        config: Configuration dictionary
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = ['vision_encoder', 'text_encoder', 'num_classes']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    if config['num_classes'] < 2:
        raise ValueError("num_classes must be >= 2")
    
    if config.get('lora_rank', 8) < 1:
        raise ValueError("lora_rank must be >= 1")
    
    return True
