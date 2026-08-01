# CLARA Configuration Files

This folder contains configuration files for training CLARA on different datasets.

---

## 📁 Available Configurations

### 1. **default_config.yaml**
**Dataset:** MVSA-Single  
**Task:** 3-class sentiment analysis (Positive, Neutral, Negative)  
**Performance:** 83.16% accuracy, 83.04% weighted F1  

**Key Settings:**
- LoRA rank: 8, alpha: 16
- Co-attention: 2 layers, 8 heads
- Batch size: 32
- Learning rate: 2e-5
- Early stopping patience: 10

**Use for:**
- Standard multimodal sentiment analysis
- Balanced datasets
- Default CLARA setup

---

### 2. **mvsa_multiple_config.yaml**
**Dataset:** MVSA-Multiple  
**Task:** 3-class sentiment analysis with extreme imbalance  
**Performance:** 73.51% accuracy, 73.45% weighted F1  

**Key Settings:**
- **Oversampling:** Negative class × 9.44
- **Class weights:** Enabled
- Monitor metric: F1-macro (better for imbalanced data)

**Use for:**
- Highly imbalanced datasets (3.2% minority class)
- When one class is severely underrepresented
- Requires careful handling of class distribution

**Class Distribution:**
- Positive: 2,564 (72.1%)
- Neutral: 879 (24.7%)
- Negative: 112 (3.2%) ← MINORITY

---

### 3. **hfm_config.yaml**
**Dataset:** HFM (Hateful Memes)  
**Task:** Binary hate speech detection  
**Performance:** 88.13% accuracy, 87.82% macro F1  

**Key Settings:**
- **Binary classification:** 2 classes
- **Balanced dataset:** No oversampling needed
- Additional metric: ROC-AUC
- Light image augmentation

**Use for:**
- Hate speech detection in memes
- Binary classification tasks
- Cross-modal reasoning requirements

**Class Distribution:**
- Non-hateful: 12,435 (50.5%)
- Hateful: 12,200 (49.5%) ← BALANCED

---

## 🚀 Usage

### Load Configuration in Python

```python
import yaml

# Load config
with open('configs/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Access settings
learning_rate = config['training']['learning_rate']
batch_size = config['training']['batch_size']
num_classes = config['model']['num_classes']
```

### Use with Training Script

```bash
# Train with default config
python scripts/train.py --config configs/default_config.yaml

# Train with MVSA-Multiple config
python scripts/train.py --config configs/mvsa_multiple_config.yaml

# Train with HFM config
python scripts/train.py --config configs/hfm_config.yaml
```

### Modify Configuration

```python
# Load and modify
with open('configs/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Change settings
config['training']['learning_rate'] = 1e-5
config['training']['batch_size'] = 16
config['model']['num_classes'] = 2

# Create model with custom config
from src.model import CLARAConfig, CLARAModel

model_config = CLARAConfig(
    num_classes=config['model']['num_classes'],
    lora_rank=config['model']['lora_rank'],
    lora_alpha=config['model']['lora_alpha']
)
model = CLARAModel(model_config)
```

---

## ⚙️ Configuration Structure

### Model Configuration
```yaml
model:
  vision_encoder: "openai/clip-vit-base-patch16"
  text_encoder: "microsoft/deberta-v3-base"
  hidden_dim: 512
  num_classes: 3
  lora_rank: 8
  lora_alpha: 16
  num_attention_heads: 8
  num_attention_layers: 2
```

### Training Configuration
```yaml
training:
  optimizer: "adamw"
  learning_rate: 2.0e-5
  batch_size: 32
  num_epochs: 50
  early_stopping_patience: 10
  label_smoothing: 0.1
```

### Data Configuration
```yaml
data:
  data_dir: "data/MVSA-Single"
  max_text_length: 77
  image_size: 224
  num_workers: 4
```

---

## 🎯 Best Practices

### 1. **For Balanced Datasets**
Use `default_config.yaml` with:
- No oversampling
- Standard cross-entropy loss
- F1-weighted as monitor metric

### 2. **For Imbalanced Datasets**
Use `mvsa_multiple_config.yaml` with:
- Oversampling minority classes
- Class weights enabled
- F1-macro as monitor metric
- Longer training (patience ≥ 10)

### 3. **For Binary Classification**
Use `hfm_config.yaml` with:
- 2 classes
- ROC-AUC metric
- Balanced approach

### 4. **For Custom Datasets**
1. Copy `default_config.yaml`
2. Modify:
   - `num_classes`
   - `data_dir`
   - `class_names`
   - Oversampling if needed
3. Test on small subset first

---

## 📊 Hyperparameter Recommendations

### Learning Rate
- **Standard:** 2e-5 (default)
- **Large dataset:** 1e-5
- **Small dataset:** 3e-5
- **Fine-tuning:** 5e-6

### Batch Size
- **Default:** 32
- **Large GPU (>16GB):** 64
- **Small GPU (<8GB):** 16
- Use gradient accumulation if needed

### LoRA Rank
- **Default:** 8 (good balance)
- **More capacity:** 16
- **Less parameters:** 4
- Alpha typically 2× rank

### Co-Attention Layers
- **Default:** 2 layers
- **Complex tasks:** 3 layers
- **Speed priority:** 1 layer

---

## 🔧 Custom Configuration Template

```yaml
# my_custom_config.yaml

model:
  vision_encoder: "openai/clip-vit-base-patch16"
  text_encoder: "microsoft/deberta-v3-base"
  hidden_dim: 512
  num_classes: YOUR_NUM_CLASSES
  lora_rank: 8
  lora_alpha: 16
  num_attention_heads: 8
  num_attention_layers: 2

training:
  learning_rate: 2.0e-5
  batch_size: 32
  num_epochs: 50
  early_stopping_patience: 10

data:
  data_dir: "data/YOUR_DATASET"
  max_text_length: 77
  image_size: 224

output:
  output_dir: "outputs/YOUR_EXPERIMENT"

evaluation:
  class_names:
    - "Class1"
    - "Class2"
    # Add more classes...
```

---

## 📖 Configuration Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_rank` | 8 | LoRA adapter rank (4-16) |
| `lora_alpha` | 16 | LoRA scaling factor |
| `hidden_dim` | 512 | Projection dimension |
| `num_attention_heads` | 8 | Co-attention heads |
| `num_attention_layers` | 2 | Co-attention layers |
| `learning_rate` | 2e-5 | AdamW learning rate |
| `batch_size` | 32 | Training batch size |
| `num_epochs` | 50 | Maximum epochs |
| `early_stopping_patience` | 10 | Epochs without improvement |
| `label_smoothing` | 0.1 | Label smoothing factor |

---

## 💡 Tips

1. **Start with default config** and modify incrementally
2. **Monitor both weighted and macro F1** to understand class performance
3. **Use class weights or oversampling** for imbalanced data
4. **Enable mixed precision** for faster training
5. **Set seed** for reproducibility
6. **Save best model only** to save disk space
7. **Use early stopping** to prevent overfitting

---

## 📞 Need Help?

If you need to create a custom configuration:

1. Copy the most similar existing config
2. Modify dataset-specific settings
3. Test with `--debug` flag first
4. Monitor validation metrics closely
5. Adjust hyperparameters based on performance

For more details, see:
- [Training Guide](../docs/TRAINING.md)
- [Paper](../README.md)
- [GitHub Issues](https://github.com/phuonglamgithub/CLARA/issues)
