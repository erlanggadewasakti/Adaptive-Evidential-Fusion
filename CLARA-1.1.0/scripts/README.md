# 🔧 CLARA Scripts

Training, evaluation, and utility scripts for CLARA.

---

## 📝 SCRIPTS TO BE ADDED

This directory will contain the following scripts:

### 1. train.py
Train CLARA on any dataset with custom configurations.

**Planned Usage:**
```bash
python scripts/train.py \
    --config configs/default_config.yaml \
    --data_dir data/MVSA-Single \
    --output_dir outputs/mvsa_single
```

---

### 2. evaluate.py
Evaluate trained model on test set.

**Planned Usage:**
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/clara_mvsa_single.pt \
    --data_dir data/MVSA-Single \
    --output_dir results/mvsa_single
```

---

### 3. inference.py
Run inference on custom images and texts.

**Planned Usage:**
```bash
python scripts/inference.py \
    --checkpoint checkpoints/clara_mvsa_single.pt \
    --image path/to/image.jpg \
    --text "Your caption here"
```

---

## 💻 CURRENT USAGE

**For now, use the provided Jupyter notebooks:**
- `notebooks/CLARA_MVSA_Single.ipynb`
- `notebooks/CLARA_MVSA_Multiple.ipynb`
- `notebooks/CLARA_HFM.ipynb`

**Or use the source code directly:**

```python
from src.model import CLARAModel, CLARAConfig
from src.trainer import CLARATrainer
from src.dataset import create_dataloaders

# Training example
config = CLARAConfig(num_classes=3)
model = CLARAModel(config)

train_loader, val_loader, test_loader = create_dataloaders(
    data_dir='data/MVSA-Single',
    batch_size=32
)

trainer = CLARATrainer(model, train_loader, val_loader)
trainer.train(num_epochs=50)
```

---

## 🚀 COMING SOON

Standalone scripts will be added in future releases to provide:
- ✅ Command-line training
- ✅ Batch evaluation
- ✅ Easy inference
- ✅ Model export utilities
- ✅ Data preprocessing tools

---

## 📞 SUPPORT

- **Email:** lamphuong.ict89@gmail.com; thientk@huit.edu.vn
- **GitHub Issues:** https://github.com/phuonglamgithub/CLARA/issues
