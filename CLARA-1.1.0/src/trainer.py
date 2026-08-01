"""
CLARA Trainer Module

Handles training, validation, and evaluation of CLARA models.
"""

import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, Optional, List
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class CLARATrainer:
    """Trainer for CLARA model"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        device: Optional[str] = None,
        label_smoothing: float = 0.1,
        early_stopping_patience: int = 10
    ):
        """
        Args:
            model: CLARA model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            output_dir: Directory to save checkpoints and logs
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
            warmup_ratio: Ratio of warmup steps
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to train on ('cuda' or 'cpu')
            label_smoothing: Label smoothing factor
            early_stopping_patience: Patience for early stopping
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = output_dir
        self.max_grad_norm = max_grad_norm
        self.early_stopping_patience = early_stopping_patience
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Tracking
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Move batch to device
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(pixel_values, input_ids, attention_mask)
            loss = self.criterion(outputs['logits'], labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs['logits'], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(pixel_values, input_ids, attention_mask)
                loss = self.criterion(outputs['logits'], labels)
                
                # Track metrics
                total_loss += loss.item()
                preds = torch.argmax(outputs['logits'], dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1
        }
    
    def train(
        self,
        num_epochs: int,
        scheduler: Optional[any] = None
    ) -> Dict[str, List[float]]:
        """
        Train model for multiple epochs
        
        Args:
            num_epochs: Number of epochs to train
            scheduler: Optional learning rate scheduler
        
        Returns:
            Training history
        """
        print(f"Training on device: {self.device}")
        print(f"Total epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("-" * 50)
        
        # Create scheduler if not provided
        if scheduler is None:
            scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            scheduler.step()
            
            # Log metrics
            print(f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.4f} | "
                  f"Train F1: {train_metrics['f1']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f}")
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            
            # Save best model
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_epoch = epoch + 1
                self.epochs_without_improvement = 0
                
                # Save checkpoint
                checkpoint_path = os.path.join(self.output_dir, "best_model.pt")
                self.model.save_pretrained(
                    checkpoint_path,
                    epoch=epoch + 1,
                    best_val_f1=self.best_val_f1,
                    optimizer_state_dict=self.optimizer.state_dict()
                )
                print(f"✅ Saved best model with F1: {self.best_val_f1:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n⚠️ Early stopping triggered after {epoch + 1} epochs")
                print(f"Best F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
                break
        
        # Save training history
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("\n" + "=" * 50)
        print(f"Training completed!")
        print(f"Best F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
        print(f"Checkpoints saved to: {self.output_dir}")
        print("=" * 50)
        
        return self.history
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, any]:
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test dataloader
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                # Move batch to device
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(pixel_values, input_ids, attention_mask)
                probs = torch.softmax(outputs['logits'], dim=-1)
                preds = torch.argmax(probs, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        
        # Per-class metrics
        f1_per_class = f1_score(all_labels, all_preds, average=None)
        precision_per_class = precision_score(all_labels, all_preds, average=None)
        recall_per_class = recall_score(all_labels, all_preds, average=None)
        
        results = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'precision': precision,
            'recall': recall,
            'f1_per_class': f1_per_class.tolist(),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
        
        # Print results
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted F1: {f1_weighted:.4f}")
        print(f"Macro F1: {f1_macro:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print("\nPer-class F1 scores:")
        for i, f1 in enumerate(f1_per_class):
            print(f"  Class {i}: {f1:.4f}")
        print("=" * 50)
        
        return results
