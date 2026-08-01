"""
Configuration Loader Utility

Helper functions to load and validate CLARA configurations.
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class CLARAConfigLoader:
    """Configuration loader and validator"""
    
    config_path: str
    config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Load configuration on initialization"""
        self.load()
    
    def load(self):
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Validate configuration
        self.validate()
        
        print(f"✅ Configuration loaded from: {self.config_path}")
    
    def validate(self):
        """Validate configuration structure"""
        required_sections = ['model', 'training', 'data', 'output']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate model config
        model_config = self.config['model']
        assert 'num_classes' in model_config, "num_classes is required"
        assert model_config['num_classes'] >= 2, "num_classes must be >= 2"
        
        # Validate training config
        training_config = self.config['training']
        assert 'learning_rate' in training_config, "learning_rate is required"
        assert 'batch_size' in training_config, "batch_size is required"
        
        print("✅ Configuration validated")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key path
        
        Args:
            key_path: Dot-separated path (e.g., 'model.lora_rank')
            default: Default value if key not found
        
        Returns:
            Configuration value
        
        Example:
            >>> loader = CLARAConfigLoader('configs/default_config.yaml')
            >>> lora_rank = loader.get('model.lora_rank')  # Returns 8
            >>> batch_size = loader.get('training.batch_size')  # Returns 32
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def update(self, key_path: str, value: Any):
        """
        Update configuration value by dot-separated key path
        
        Args:
            key_path: Dot-separated path (e.g., 'training.learning_rate')
            value: New value
        
        Example:
            >>> loader = CLARAConfigLoader('configs/default_config.yaml')
            >>> loader.update('training.learning_rate', 1e-5)
            >>> loader.update('model.num_classes', 2)
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        print(f"✅ Updated {key_path} = {value}")
    
    def save(self, output_path: Optional[str] = None):
        """
        Save configuration to YAML file
        
        Args:
            output_path: Path to save (defaults to original path)
        """
        save_path = output_path or self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Configuration saved to: {save_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self.config.copy()
    
    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)
        
        # Model
        print("\n📦 Model:")
        print(f"  Vision Encoder: {self.get('model.vision_encoder')}")
        print(f"  Text Encoder: {self.get('model.text_encoder')}")
        print(f"  Number of Classes: {self.get('model.num_classes')}")
        print(f"  LoRA Rank: {self.get('model.lora_rank')}")
        print(f"  LoRA Alpha: {self.get('model.lora_alpha')}")
        print(f"  Hidden Dim: {self.get('model.hidden_dim')}")
        
        # Training
        print("\n🎯 Training:")
        print(f"  Learning Rate: {self.get('training.learning_rate')}")
        print(f"  Batch Size: {self.get('training.batch_size')}")
        print(f"  Num Epochs: {self.get('training.num_epochs')}")
        print(f"  Early Stopping: {self.get('training.early_stopping_patience')}")
        print(f"  Optimizer: {self.get('training.optimizer')}")
        print(f"  Scheduler: {self.get('training.scheduler')}")
        
        # Data
        print("\n💾 Data:")
        print(f"  Data Directory: {self.get('data.data_dir')}")
        print(f"  Max Text Length: {self.get('data.max_text_length')}")
        print(f"  Image Size: {self.get('data.image_size')}")
        print(f"  Num Workers: {self.get('data.num_workers')}")
        
        # Output
        print("\n📂 Output:")
        print(f"  Output Directory: {self.get('output.output_dir')}")
        print(f"  Save Best Only: {self.get('output.save_best_only')}")
        
        print("=" * 60 + "\n")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Simple function to load configuration
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    loader = CLARAConfigLoader(config_path)
    return loader.to_dict()


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two configurations (override takes precedence)
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
    
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


# Example usage
if __name__ == "__main__":
    # Load default config
    loader = CLARAConfigLoader("configs/default_config.yaml")
    
    # Print summary
    loader.print_summary()
    
    # Get specific values
    print(f"\nLearning Rate: {loader.get('training.learning_rate')}")
    print(f"Batch Size: {loader.get('training.batch_size')}")
    print(f"LoRA Rank: {loader.get('model.lora_rank')}")
    
    # Update values
    loader.update('training.learning_rate', 1e-5)
    loader.update('training.batch_size', 16)
    
    # Save modified config
    loader.save("configs/custom_config.yaml")
