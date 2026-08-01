"""
CLARA Dataset Module

Handles loading and preprocessing of multimodal datasets:
- MVSA-Single
- MVSA-Multiple
- HFM (Hateful Memes)
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPProcessor, DebertaV2Tokenizer
from typing import Dict, List, Optional, Tuple


class CLARADataset(Dataset):
    """Dataset class for CLARA multimodal sentiment analysis"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        vision_model: str = "openai/clip-vit-base-patch16",
        text_model: str = "microsoft/deberta-v3-base",
        max_length: int = 77,
        image_size: int = 224
    ):
        """
        Args:
            data_dir: Path to dataset directory
            split: One of ['train', 'val', 'test']
            vision_model: Vision encoder model name
            text_model: Text encoder model name
            max_length: Maximum text sequence length
            image_size: Image size for resizing
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_length = max_length
        self.image_size = image_size
        
        # Load annotations
        annotations_path = os.path.join(data_dir, split, "annotations.csv")
        self.annotations = pd.read_csv(annotations_path)
        
        # Initialize processors
        self.clip_processor = CLIPProcessor.from_pretrained(vision_model)
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(text_model)
        
        # Label mapping
        self.label_map = self._create_label_map()
    
    def _create_label_map(self) -> Dict[str, int]:
        """Create label to index mapping"""
        unique_labels = self.annotations['label'].unique()
        
        # Standard mappings
        if set(unique_labels) == {'positive', 'neutral', 'negative'}:
            return {'positive': 0, 'neutral': 1, 'negative': 2}
        elif set(unique_labels) == {'hateful', 'non-hateful'}:
            return {'non-hateful': 0, 'hateful': 1}
        else:
            # Auto-generate mapping
            return {label: idx for idx, label in enumerate(sorted(unique_labels))}
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
                - pixel_values: [3, 224, 224]
                - input_ids: [max_length]
                - attention_mask: [max_length]
                - label: int
        """
        row = self.annotations.iloc[idx]
        
        # Load image
        image_path = os.path.join(self.data_dir, self.split, row['image_path'])
        image = Image.open(image_path).convert('RGB')
        
        # Process image
        image_inputs = self.clip_processor(
            images=image,
            return_tensors="pt"
        )
        pixel_values = image_inputs['pixel_values'].squeeze(0)
        
        # Process text
        text = str(row['text'])
        text_inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = text_inputs['input_ids'].squeeze(0)
        attention_mask = text_inputs['attention_mask'].squeeze(0)
        
        # Get label
        label = self.label_map[row['label']]
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    vision_model: str = "openai/clip-vit-base-patch16",
    text_model: str = "microsoft/deberta-v3-base",
    max_length: int = 77,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        vision_model: Vision encoder model name
        text_model: Text encoder model name
        max_length: Maximum text sequence length
        image_size: Image size
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = CLARADataset(
        data_dir=data_dir,
        split="train",
        vision_model=vision_model,
        text_model=text_model,
        max_length=max_length,
        image_size=image_size
    )
    
    val_dataset = CLARADataset(
        data_dir=data_dir,
        split="val",
        vision_model=vision_model,
        text_model=text_model,
        max_length=max_length,
        image_size=image_size
    )
    
    test_dataset = CLARADataset(
        data_dir=data_dir,
        split="test",
        vision_model=vision_model,
        text_model=text_model,
        max_length=max_length,
        image_size=image_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class OversamplingDataset(Dataset):
    """Dataset with oversampling for imbalanced classes"""
    
    def __init__(self, base_dataset: CLARADataset, oversample_factor: Dict[int, float]):
        """
        Args:
            base_dataset: Base CLARADataset
            oversample_factor: Dict mapping class index to oversampling factor
                Example: {2: 9.44} means oversample class 2 by 9.44x
        """
        self.base_dataset = base_dataset
        self.oversample_factor = oversample_factor
        
        # Create oversampled indices
        self.indices = self._create_oversampled_indices()
    
    def _create_oversampled_indices(self) -> List[int]:
        """Create list of indices with oversampling"""
        indices = []
        
        for idx in range(len(self.base_dataset)):
            label = self.base_dataset.annotations.iloc[idx]['label']
            label_idx = self.base_dataset.label_map[label]
            
            # Get oversampling factor for this class
            factor = self.oversample_factor.get(label_idx, 1.0)
            
            # Add index multiple times based on factor
            num_copies = int(factor)
            indices.extend([idx] * num_copies)
        
        return indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        actual_idx = self.indices[idx]
        return self.base_dataset[actual_idx]


def create_oversampled_dataloader(
    data_dir: str,
    split: str,
    batch_size: int,
    oversample_factor: Dict[int, float],
    **kwargs
) -> DataLoader:
    """
    Create dataloader with oversampling for imbalanced datasets
    
    Args:
        data_dir: Path to dataset directory
        split: Data split ('train', 'val', 'test')
        batch_size: Batch size
        oversample_factor: Dict mapping class index to oversampling factor
        **kwargs: Additional arguments for CLARADataset
    
    Returns:
        DataLoader with oversampling
    
    Example:
        # Oversample negative class (index 2) by 9.44x for MVSA-Multiple
        loader = create_oversampled_dataloader(
            data_dir="data/MVSA-Multiple",
            split="train",
            batch_size=32,
            oversample_factor={2: 9.44}
        )
    """
    base_dataset = CLARADataset(data_dir=data_dir, split=split, **kwargs)
    oversampled_dataset = OversamplingDataset(base_dataset, oversample_factor)
    
    return DataLoader(
        oversampled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=kwargs.get('num_workers', 4),
        pin_memory=True
    )
