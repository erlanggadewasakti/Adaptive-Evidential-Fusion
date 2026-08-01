# 📓 CLARA Jupyter Notebooks

Original Jupyter notebooks for training and evaluating CLARA on three benchmark datasets.

---

## 📊 Available Notebooks

### 1. CLARA_MVSA_Single.ipynb
**Dataset:** MVSA-Single (4,869 → 2,592 samples)  
**Task:** 3-class sentiment (Positive, Neutral, Negative)  
**Performance:** 83.16% accuracy, 83.04% weighted F1  

**Key Features:**
- Single image per post
- Balanced classes: Pos (54%), Neu (18%), Neg (28%)
- Training: 1,813 samples
- LoRA fine-tuning (7.45% trainable params)

---

### 2. CLARA_MVSA_Multiple.ipynb
**Dataset:** MVSA-Multiple (19,600+ images → 3,555 posts)  
**Task:** Multi-image sentiment with extreme imbalance  
**Performance:** 73.51% accuracy, 73.45% weighted F1  

**Key Features:**
- Multiple images per post (avg 3.8 images)
- **Extreme imbalance:** Pos (72%), Neu (25%), Neg (3%)
- **9.44× oversampling** for negative class
- Triple-annotated by 3 labelers

---

### 3. CLARA_HFM.ipynb
**Dataset:** HFM - Hateful Memes (24,635 samples)  
**Task:** Binary sentiment classification  
**Performance:** 88.13% accuracy, 87.82% macro F1  

**Key Features:**
- Twitter-style posts with images
- Balanced: Pos (56.8%), Neu (43.2%)
- Train: 19,816 | Val: 2,410 | Test: 2,409
- Text with hashtags, `<url>`, `<user>` tokens

---

## 📂 Expected Data Structure

Notebooks expect data in these formats:

### MVSA Datasets
```
data/MVSA-Single/
├── images/              # 4,869 JPG images
└── labelResultAll.txt   # Format: id | label | text

data/MVSA-Multiple/
├── images/              # 19,600+ JPG images
└── labelResultAll.txt   # Format: id | label | text
```

### HFM Dataset
```
data/HFM/
├── image/               # All images (backup)
├── train/               # 19,816 PNG images
├── val/                 # 2,410 PNG images
├── test/                # 2,409 PNG images
└── text/
    ├── train.txt        # Format: ['id', 'text', label]
    ├── val.txt
    └── test.txt
```

**Download:** 🔗 https://drive.google.com/drive/folders/15Dcqsm1OvJbU9_Ok1ylbK_widCekjK0O

---

## 🚀 Usage

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Install Jupyter
pip install jupyter notebook

# Start Jupyter
jupyter notebook

# Open desired notebook in notebooks/ folder
```

### Running on Google Colab

1. Upload notebook to Google Drive
2. Open with Google Colab
3. Change runtime to GPU:
   - Runtime → Change runtime type → T4 GPU (free)
4. Install requirements:
   ```python
   !pip install transformers==4.30.0 peft==0.4.0 torch torchvision
   ```
5. Mount Google Drive for data access:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
6. Run cells sequentially

---

## ⚙️ Configuration

Each notebook has configuration at the top:

```python
# Model Configuration
CONFIG = {
    # Model
    'vision_encoder': 'openai/clip-vit-base-patch16',
    'text_encoder': 'microsoft/deberta-v3-base',
    'hidden_dim': 512,
    'num_classes': 3,  # 3 for MVSA, 2 for HFM
    
    # LoRA
    'lora_rank': 8,
    'lora_alpha': 16,
    'lora_dropout': 0.1,
    
    # Training
    'num_epochs': 50,
    'batch_size': 32,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'early_stopping_patience': 10,
    
    # Data (MVSA-Multiple only)
    'use_oversampling': True,
    'oversample_factor': {2: 9.44}  # Negative class
}
```

---

## 💻 Requirements

### Hardware
- **Minimum:** 8GB GPU memory (T4, GTX 1080)
- **Recommended:** 12GB+ GPU (V100, A100)
- **CPU Fallback:** Possible but very slow

### Software
```
Python >= 3.8
torch >= 2.0.0
transformers >= 4.30.0
peft >= 0.4.0
pillow >= 9.0.0
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
tqdm >= 4.62.0
```

---

## ⏱️ Training Time

Approximate training times on different GPUs:

| Dataset | V100 (16GB) | T4 (16GB) | GTX 1080 Ti (11GB) |
|---------|-------------|-----------|---------------------|
| **MVSA-Single** | 30-45 min | 60-90 min | 90-120 min |
| **MVSA-Multiple** | 40-60 min | 90-120 min | 120-150 min |
| **HFM** | 2-3 hours | 4-5 hours | 5-6 hours |

**Reduce batch_size to 16 or 8 if GPU OOM occurs.**

---

## 📁 Output Structure

Notebooks generate outputs in these directories:

```
checkpoints/
├── clara_mvsa_single_best.pt      # Best model weights
├── clara_mvsa_multiple_best.pt
└── clara_hfm_best.pt

results/
├── mvsa_single/
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   ├── per_class_metrics.png
│   └── metrics.json
├── mvsa_multiple/
└── hfm/
```

---

## 🔍 Reproducibility

All notebooks use fixed random seeds:

```python
import torch
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Expected variance:** ±0.5% due to hardware/driver differences

---

## 🎯 Key Notebook Sections

Each notebook contains:

1. **Setup & Imports** - Load libraries and set seeds
2. **Data Loading** - Load from correct format (see above)
3. **Data Preprocessing** - CLIP + DeBERTa preprocessing
4. **Model Definition** - CLARA architecture with LoRA
5. **Training Loop** - With early stopping and validation
6. **Evaluation** - Accuracy, F1, confusion matrix
7. **Visualization** - Training curves, attention weights
8. **Save Results** - Checkpoints and metrics

---

## 🆘 Troubleshooting

### GPU Out of Memory
```python
CONFIG['batch_size'] = 16  # Or 8
```

### Data Loading Errors (MVSA)
```python
# Check pipe separator
with open('labelResultAll.txt', 'r') as f:
    line = f.readline()
    parts = line.strip().split(' | ')  # Must have spaces
    print(len(parts))  # Should be 3
```

### Data Loading Errors (HFM)
```python
# Use ast.literal_eval, not json.loads
import ast
with open('train.txt', 'r') as f:
    line = f.readline()
    data = ast.literal_eval(line.strip())  # Not json.loads!
    print(data)  # ['id', 'text', label]
```

### Slow Training
- Reduce batch_size
- Use mixed precision: `torch.cuda.amp.autocast()`
- Reduce num_workers in DataLoader

---

## 📊 Expected Results

### MVSA-Single
```
Accuracy:     83.16%
Weighted F1:  83.04%
Macro F1:     79.88%

Per-class F1:
  Positive:   88.84%
  Neutral:    70.77%
  Negative:   79.83%
```

### MVSA-Multiple
```
Accuracy:     73.51%
Weighted F1:  73.45%
Macro F1:     67.92%

Per-class F1:
  Positive:   81.24%
  Neutral:    75.09%
  Negative:   47.43% (challenging!)
```

### HFM
```
Accuracy:     88.13%
Weighted F1:  88.13%
Macro F1:     87.82%

Per-class F1:
  Neutral (0): 88.33%
  Positive (1): 87.92%
```

---

## 📝 Notes

- **Original Notebooks:** The complete notebooks are available in this repository
- **Checkpoints:** Pre-trained model weights available on Google Drive
- **Custom Modifications:** Feel free to modify configs and experiment
- **GPU Memory:** Adjust batch_size based on available memory

---

## 📧 Support

**Issues with notebooks:**
- GitHub Issues: https://github.com/phuonglamgithub/CLARA/issues
- Email: lamphuong.ict89@gmail.com; thientk@huit.edu.vn

**Data format questions:**
- See `data/README.md` for complete format documentation

---

## 📚 Citation

```bibtex
@article{lam2025clara,
  title={CLARA: Enhancing Multimodal Sentiment Analysis via Efficient Vision-Language Fusion},
  author={Lam, Phuong and Phan, Tuoi Thi and Tran, Thien Khai},
  journal={The Visual Computer},
  year={2025},
  publisher={Springer}
}
```
