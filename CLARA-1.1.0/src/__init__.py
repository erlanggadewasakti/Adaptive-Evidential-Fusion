"""
CLARA: Co-attention Learning for Robust multimodal sentiment Analysis

A parameter-efficient vision-language framework for multimodal sentiment analysis.

Authors:
    - Phuong Lam (lamphuong.ict89@gmail.com)
    - Tuoi Thi Phan (pttuoi@ntt.edu.vn)
    - Thien Khai Tran (thientk@huit.edu.vn)

Paper: CLARA: Enhancing Multimodal Sentiment Analysis via Efficient Vision-Language Fusion
Journal: The Visual Computer (Springer)
Status: Under Review
"""

__version__ = "1.0.0"
__author__ = "Phuong Lam, Tuoi Thi Phan, Thien Khai Tran"
__email__ = "thientk@huit.edu.vn"

from .model import CLARAModel, CLARAConfig
from .dataset import CLARADataset, create_dataloaders
from .trainer import CLARATrainer
from .utils import set_seed, get_device, count_parameters

__all__ = [
    "CLARAModel",
    "CLARAConfig",
    "CLARADataset",
    "create_dataloaders",
    "CLARATrainer",
    "set_seed",
    "get_device",
    "count_parameters",
]
