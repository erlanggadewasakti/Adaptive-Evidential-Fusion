"""
CLARA Model Architecture

Implements the CLARA model with:
- CLIP Vision Encoder + LoRA
- DeBERTa Text Encoder + LoRA
- Bidirectional Co-Attention Fusion
- Verification Module
- Feedback Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPProcessor, DebertaV2Model, DebertaV2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class CLARAConfig:
    """Configuration for CLARA model"""
    
    # Encoder settings
    vision_encoder: str = "openai/clip-vit-base-patch16"
    text_encoder: str = "microsoft/deberta-v3-base"
    
    # LoRA settings
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    # Co-attention settings
    hidden_dim: int = 512
    num_attention_heads: int = 8
    num_attention_layers: int = 2
    attention_dropout: float = 0.1
    
    # Classification settings
    num_classes: int = 3  # Positive, Neutral, Negative
    
    # Other settings
    dropout: float = 0.1
    freeze_encoders: bool = True


class CoAttentionLayer(nn.Module):
    """Bidirectional Co-Attention Layer"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Text queries Vision
        self.text_to_vision_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Vision queries Text
        self.vision_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.vision_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward
        self.text_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.vision_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.text_ffn_norm = nn.LayerNorm(hidden_dim)
        self.vision_ffn_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_features: [batch_size, seq_len, hidden_dim]
            vision_features: [batch_size, num_patches, hidden_dim]
        
        Returns:
            text_out: [batch_size, seq_len, hidden_dim]
            vision_out: [batch_size, num_patches, hidden_dim]
        """
        # Text queries Vision
        text_attn_out, _ = self.text_to_vision_attn(
            query=text_features,
            key=vision_features,
            value=vision_features
        )
        text_features = self.text_norm(text_features + text_attn_out)
        
        # Vision queries Text
        vision_attn_out, _ = self.vision_to_text_attn(
            query=vision_features,
            key=text_features,
            value=text_features
        )
        vision_features = self.vision_norm(vision_features + vision_attn_out)
        
        # Feed-forward
        text_out = self.text_ffn_norm(text_features + self.text_ffn(text_features))
        vision_out = self.vision_ffn_norm(vision_features + self.vision_ffn(vision_features))
        
        return text_out, vision_out


class VerificationModule(nn.Module):
    """Computes unimodal predictions and consensus signal"""
    
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        
        self.vision_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.text_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            vision_features: [batch_size, hidden_dim]
            text_features: [batch_size, hidden_dim]
        
        Returns:
            vision_logits: [batch_size, num_classes]
            text_logits: [batch_size, num_classes]
            consensus: [batch_size, num_classes] - agreement signal
        """
        vision_logits = self.vision_classifier(vision_features)
        text_logits = self.text_classifier(text_features)
        
        # Consensus signal: absolute difference of probabilities
        vision_probs = F.softmax(vision_logits, dim=-1)
        text_probs = F.softmax(text_logits, dim=-1)
        consensus = torch.abs(vision_probs - text_probs)
        
        return vision_logits, text_logits, consensus


class FeedbackModule(nn.Module):
    """Refines predictions using consensus signal"""
    
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        
        self.refinement = nn.Sequential(
            nn.Linear(hidden_dim + num_classes, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(
        self,
        fused_features: torch.Tensor,
        consensus: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            fused_features: [batch_size, hidden_dim]
            consensus: [batch_size, num_classes]
        
        Returns:
            refined_logits: [batch_size, num_classes]
        """
        # Concatenate fused features with consensus signal
        combined = torch.cat([fused_features, consensus], dim=-1)
        refined_logits = self.refinement(combined)
        
        return refined_logits


class CLARAModel(nn.Module):
    """CLARA: Co-attention Learning for Robust multimodal sentiment Analysis"""
    
    def __init__(self, config: CLARAConfig):
        super().__init__()
        self.config = config
        
        # Initialize encoders
        self.vision_encoder = CLIPVisionModel.from_pretrained(config.vision_encoder)
        self.text_encoder = DebertaV2Model.from_pretrained(config.text_encoder)
        
        # Freeze encoders if specified
        if config.freeze_encoders:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # Apply LoRA to vision encoder
        vision_lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        self.vision_encoder = get_peft_model(self.vision_encoder, vision_lora_config)
        
        # Apply LoRA to text encoder (layers 0-10, full layer 11)
        text_lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=["query_proj", "value_proj"],
            layers_to_transform=list(range(11)),  # 0-10
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        self.text_encoder = get_peft_model(self.text_encoder, text_lora_config)
        
        # Projection layers
        vision_hidden_size = self.vision_encoder.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size
        
        self.vision_projection = nn.Linear(vision_hidden_size, config.hidden_dim)
        self.text_projection = nn.Linear(text_hidden_size, config.hidden_dim)
        
        # Co-attention layers
        self.co_attention_layers = nn.ModuleList([
            CoAttentionLayer(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_attention_heads,
                dropout=config.attention_dropout
            )
            for _ in range(config.num_attention_layers)
        ])
        
        # Verification module
        self.verification = VerificationModule(
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout
        )
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
        
        # Feedback module
        self.feedback = FeedbackModule(
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout
        )
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: [batch_size, 3, 224, 224]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            Dictionary with:
                - logits: [batch_size, num_classes]
                - vision_logits: [batch_size, num_classes]
                - text_logits: [batch_size, num_classes]
                - consensus: [batch_size, num_classes]
        """
        # Encode vision
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        vision_features = vision_outputs.last_hidden_state  # [B, num_patches, hidden]
        vision_features = self.vision_projection(vision_features)
        
        # Encode text
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state  # [B, seq_len, hidden]
        text_features = self.text_projection(text_features)
        
        # Co-attention fusion
        for co_attn_layer in self.co_attention_layers:
            text_features, vision_features = co_attn_layer(text_features, vision_features)
        
        # Pool features
        vision_pooled = vision_features.mean(dim=1)  # [B, hidden_dim]
        text_pooled = text_features.mean(dim=1)  # [B, hidden_dim]
        fused_features = (vision_pooled + text_pooled) / 2  # [B, hidden_dim]
        
        # Verification: unimodal predictions + consensus
        vision_logits, text_logits, consensus = self.verification(
            vision_pooled, text_pooled
        )
        
        # Fused prediction
        fused_logits = self.prediction_head(fused_features)
        
        # Feedback: refine with consensus
        final_logits = self.feedback(fused_features, consensus)
        final_logits = final_logits + fused_logits  # Residual connection
        
        return {
            "logits": final_logits,
            "vision_logits": vision_logits,
            "text_logits": text_logits,
            "consensus": consensus
        }
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, config: Optional[CLARAConfig] = None):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        if config is None:
            config = CLARAConfig(**checkpoint.get("config", {}))
        
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return model
    
    def save_pretrained(self, save_path: str, **kwargs):
        """Save model checkpoint"""
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.config.__dict__,
            **kwargs
        }, save_path)
    
    def predict(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Inference method"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(pixel_values, input_ids, attention_mask)
            probs = F.softmax(outputs["logits"], dim=-1)
            preds = torch.argmax(probs, dim=-1)
            confidence, _ = torch.max(probs, dim=-1)
            
            return {
                "predictions": preds,
                "probabilities": probs,
                "confidence": confidence
            }
