"""
==========================================================================
Tau (τ) Hyperparameter Optimization for ADEF Model
==========================================================================
Author : Erlangga Dewa Sakti
Purpose: Find the optimal conflict-mass threshold (τ) for the Adaptive
         Evidential Fusion (ADEF) model via two-pass grid search, then
         retrain the full model with the discovered optimal τ.

Usage  : python optimize_tau.py
         (run from the project root directory where adef.ipynb lives)

Note   : This script is designed to be run on a machine with GPU (CUDA).
         It replicates the full model architecture from adef.ipynb so that
         it is self-contained.
==========================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.models as models
from transformers import RobertaTokenizer, RobertaModel

from PIL import Image
import pandas as pd
import numpy as np
import os
import sys
import warnings
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# REPRODUCIBILITY & SEEDING
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
print("[INFO] Imports loaded & global seed set to 42.")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")


# ============================================================
# CONFIGURATION CLASS
# ============================================================
class CFG:
    # --- Seed & System Setup ---
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 0  # 0 for safe multi-processing on Windows

    # --- Dataset & File Paths ---
    DATASET_NAME = "MVSA-Single"
    ROOT_DIR = "D:/MVSA_SINGLE" if os.path.exists("D:/MVSA_SINGLE") else "./data/MVSA_Single"
    DATA_DIR = os.path.join(ROOT_DIR, "data") if os.path.exists(os.path.join(ROOT_DIR, "data")) else "./data/MVSA_Single/images"
    LABEL_PATH = os.path.join(ROOT_DIR, "labelResultAllFinal.txt") if os.path.exists(os.path.join(ROOT_DIR, "labelResultAllFinal.txt")) else "./data/MVSA_Single/data_label.csv"
    OUTPUT_DIR = "./output"

    # --- Unimodal Feature Backbones ---
    TEXT_BACKBONE = "roberta-base"
    IMAGE_BACKBONE = "densenet121"
    FEATURE_DIM = 768
    NUM_CLASSES = 3

    # --- Image Preprocessing Settings ---
    IMAGE_SIZE = 224
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    # --- Text Preprocessing Settings ---
    MAX_LEN = 128

    # --- ADEF Threshold Hyperparameter ---
    TAU_THRESHOLD = 0.03  # Current baseline value (will be optimized)

    # --- Training Hyperparameters ---
    BATCH_SIZE = 32
    EPOCHS = 20
    LR_BACKBONE = 2e-5
    LR_HEAD = 1e-4
    WEIGHT_DECAY = 0.01
    KL_ANNEALING_EPOCHS = 10

    # --- Visualization & Logging ---
    SAVE_PLOTS = True
    PLOT_FORMAT = "png"

os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
print(f"[INFO] CFG loaded. Operating on Device: {CFG.DEVICE}")
print(f"Dataset root: {CFG.ROOT_DIR}")
print(f"Output directory: {CFG.OUTPUT_DIR}")


# ============================================================
# DATA PREPARATION & PYTORCH DATASET
# ============================================================
def load_and_preprocess_dataframe():
    if CFG.LABEL_PATH.endswith(".txt"):
        df = pd.read_csv(CFG.LABEL_PATH, header=0, sep=",")
        df.columns = ["id", "text_label", "image_label", "final_label"]

        def is_valid(row):
            if row["text_label"] == "positive" and row["image_label"] == "negative":
                return False
            if row["text_label"] == "negative" and row["image_label"] == "positive":
                return False
            return True

        df = df[df.apply(is_valid, axis=1)].reset_index(drop=True)
    else:
        df = pd.read_csv(CFG.LABEL_PATH)

    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    if "label" not in df.columns and "final_label" in df.columns:
        df["label"] = df["final_label"].map(label_map)

    def load_text(sample_id):
        path = os.path.join(CFG.DATA_DIR, f"{sample_id}.txt")
        if not os.path.exists(path):
            path_alt = os.path.join(CFG.ROOT_DIR, "data", f"{sample_id}.txt")
            if os.path.exists(path_alt):
                path = path_alt
            else:
                return ""
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
        for enc in encodings:
            try:
                with open(path, "r", encoding=enc) as f:
                    text = f.read().strip()
                    if text:
                        return text
            except Exception:
                continue
        return ""

    if "text" not in df.columns or df["text"].isnull().all():
        df["text"] = df["id"].apply(load_text)

    def get_image_path(sample_id):
        img_path = os.path.join(CFG.DATA_DIR, f"{sample_id}.jpg")
        if not os.path.exists(img_path):
            img_path_alt = os.path.join(CFG.ROOT_DIR, "data", f"{sample_id}.jpg")
            if os.path.exists(img_path_alt):
                return img_path_alt
        return img_path

    df["image_path"] = df["id"].apply(get_image_path)
    return df


class MultimodalDataset(Dataset):
    def __init__(self, dataframe, tokenizer, transform, max_len=128):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"]) if pd.notna(row["text"]) and str(row["text"]).strip() != "" else "empty"
        image_path = row["image_path"]
        label = int(row["label"])

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
        except Exception:
            image = torch.zeros(3, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "label": torch.tensor(label, dtype=torch.long)
        }


# ============================================================
# ARCHITECTURE BUILDING BLOCKS
# ============================================================
class CoAttentionModule(nn.Module):
    def __init__(self, feature_dim=768):
        super(CoAttentionModule, self).__init__()
        self.W = nn.Linear(feature_dim, feature_dim, bias=False)
        self.proj_c = nn.Linear(2 * feature_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, h_t, h_v):
        h_v_proj = self.W(h_v)
        scores = torch.matmul(h_t, h_v_proj.transpose(1, 2)) / (h_t.size(-1) ** 0.5)
        A = F.softmax(scores, dim=-1)
        h_t_att = torch.matmul(A, h_v)
        h_v_att = torch.matmul(A.transpose(1, 2), h_t)
        bar_h_t = h_t_att.mean(dim=1)
        bar_h_v = h_v_att.mean(dim=1)
        h_c_cat = torch.cat([bar_h_t, bar_h_v], dim=-1)
        h_c = self.dropout(F.relu(self.proj_c(h_c_cat)))
        h_c = self.layer_norm(h_c)
        return h_c, A


class EvidentialLayer(nn.Module):
    def __init__(self, in_dim=768, num_classes=3):
        super(EvidentialLayer, self).__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, h):
        logits = self.fc(h)
        evidence = F.relu(logits)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)
        b = evidence / S
        u = self.num_classes / S
        return {
            "evidence": evidence,
            "alpha": alpha,
            "b": b,
            "u": u,
            "S": S
        }


class ADEFModule(nn.Module):
    def __init__(self, tau=0.5, num_classes=3):
        super(ADEFModule, self).__init__()
        self.tau = tau
        self.num_classes = num_classes

    def forward(self, b_t, u_t, b_v, u_v, b_c, u_c):
        # 1. Conflict Mass K_tv = sum_{i != j} b_t_i * b_v_j
        b_t_sum = b_t.sum(dim=-1, keepdim=True)
        b_v_sum = b_v.sum(dim=-1, keepdim=True)
        K_tv = b_t_sum * b_v_sum - torch.sum(b_t * b_v, dim=-1, keepdim=True)
        K_tv = torch.clamp(K_tv, min=0.0, max=0.9999)

        # Route A: Standard Dempster-Shafer
        denom_tv = torch.clamp(1.0 - K_tv, min=1e-8)
        b_tv = (b_t * b_v + b_t * u_v + b_v * u_t) / denom_tv
        u_tv = (u_t * u_v) / denom_tv

        b_tv_sum = b_tv.sum(dim=-1, keepdim=True)
        b_c_sum = b_c.sum(dim=-1, keepdim=True)
        K_tvc = b_tv_sum * b_c_sum - torch.sum(b_tv * b_c, dim=-1, keepdim=True)
        denom_tvc = torch.clamp(1.0 - K_tvc, min=1e-8)

        b_routeA = (b_tv * b_c + b_tv * u_c + b_c * u_tv) / denom_tvc
        u_routeA = (u_tv * u_c) / denom_tvc

        # Route B: Adaptive Conflict-Aware Fusion
        b_routeB = (1.0 - K_tv) * ((b_t + b_v) / 2.0) + K_tv * b_c
        u_routeB = 1.0 - torch.sum(b_routeB, dim=-1, keepdim=True)
        u_routeB = torch.clamp(u_routeB, min=0.0)

        # Dynamic Selection
        route_a_mask = (K_tv <= self.tau)
        b_fusion = torch.where(route_a_mask, b_routeA, b_routeB)
        u_fusion = torch.where(route_a_mask, u_routeA, u_routeB)

        # Final Probability: p_i = b_fusion_i + u_fusion / M
        p = b_fusion + (u_fusion / self.num_classes)

        return p, b_fusion, u_fusion, K_tv


class ADEFModel(nn.Module):
    def __init__(self, cfg):
        super(ADEFModel, self).__init__()
        self.cfg = cfg
        self.text_encoder = RobertaModel.from_pretrained(cfg.TEXT_BACKBONE)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, cfg.FEATURE_DIM)
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.visual_encoder = densenet.features
        self.visual_proj = nn.Linear(1024, cfg.FEATURE_DIM)
        self.co_attention = CoAttentionModule(feature_dim=cfg.FEATURE_DIM)
        self.enn_text = EvidentialLayer(in_dim=cfg.FEATURE_DIM, num_classes=cfg.NUM_CLASSES)
        self.enn_visual = EvidentialLayer(in_dim=cfg.FEATURE_DIM, num_classes=cfg.NUM_CLASSES)
        self.enn_coatt = EvidentialLayer(in_dim=cfg.FEATURE_DIM, num_classes=cfg.NUM_CLASSES)
        self.adef = ADEFModule(tau=cfg.TAU_THRESHOLD, num_classes=cfg.NUM_CLASSES)

    def forward(self, input_ids, attention_mask, image):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        h_t_seq = self.text_proj(text_out.last_hidden_state)
        h_t = h_t_seq.mean(dim=1)

        v_feat = self.visual_encoder(image)
        B, C, H, W = v_feat.size()
        v_spatial = v_feat.view(B, C, H * W).permute(0, 2, 1)
        h_v_spatial = self.visual_proj(v_spatial)
        h_v = h_v_spatial.mean(dim=1)

        h_c, attn_matrix = self.co_attention(h_t_seq, h_v_spatial)

        enn_t = self.enn_text(h_t)
        enn_v = self.enn_visual(h_v)
        enn_c = self.enn_coatt(h_c)

        p, b_fusion, u_fusion, K_tv = self.adef(
            enn_t["b"], enn_t["u"],
            enn_v["b"], enn_v["u"],
            enn_c["b"], enn_c["u"]
        )

        return {
            "probs": p,
            "b_fusion": b_fusion,
            "u_fusion": u_fusion,
            "K_tv": K_tv,
            "enn_text": enn_t,
            "enn_visual": enn_v,
            "enn_coatt": enn_c,
            "attn_matrix": attn_matrix
        }


# ============================================================
# EDL LOSS
# ============================================================
class EDLLoss(nn.Module):
    def __init__(self, num_classes=3):
        super(EDLLoss, self).__init__()
        self.num_classes = num_classes

    def kl_divergence(self, alpha, num_classes):
        beta = torch.ones((1, num_classes), dtype=torch.float32, device=alpha.device)
        S_alpha = torch.sum(alpha, dim=-1, keepdim=True)
        S_beta = torch.sum(beta, dim=-1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True)
        lnB_prior = torch.sum(torch.lgamma(beta), dim=-1, keepdim=True) - torch.lgamma(S_beta)
        dgAlpha = torch.digamma(alpha)
        dgSAlpha = torch.digamma(S_alpha)
        kl = torch.sum((alpha - beta) * (dgAlpha - dgSAlpha), dim=-1, keepdim=True) + lnB + lnB_prior
        return kl

    def forward(self, alpha, y_onehot, epoch, annealing_epochs):
        S = torch.sum(alpha, dim=-1, keepdim=True)
        p_hat = alpha / S
        err = y_onehot - p_hat
        var = (p_hat * (1.0 - p_hat)) / (S + 1.0)
        l_mse = torch.sum(err ** 2 + var, dim=-1, keepdim=True)
        alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
        kl_reg = self.kl_divergence(alpha_tilde, self.num_classes)
        annealing_coef = min(1.0, float(epoch) / float(annealing_epochs))
        total_loss = l_mse + annealing_coef * kl_reg
        return torch.mean(total_loss)


# ============================================================
# TRAINING & VALIDATION FUNCTIONS
# ============================================================
def train_epoch(model, dataloader, criterion, optimizer, scaler, epoch, cfg):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]")
    for batch in pbar:
        input_ids = batch["input_ids"].to(cfg.DEVICE)
        attention_mask = batch["attention_mask"].to(cfg.DEVICE)
        images = batch["image"].to(cfg.DEVICE)
        labels = batch["label"].to(cfg.DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(cfg.DEVICE == "cuda")):
            out = model(input_ids, attention_mask, images)
            y_onehot = F.one_hot(labels, num_classes=cfg.NUM_CLASSES).float()
            alpha_fused = out["b_fusion"] * (cfg.NUM_CLASSES / torch.clamp(out["u_fusion"], min=1e-6)) + 1.0
            loss_fused = criterion(alpha_fused, y_onehot, epoch, cfg.KL_ANNEALING_EPOCHS)
            loss_t = criterion(out["enn_text"]["alpha"], y_onehot, epoch, cfg.KL_ANNEALING_EPOCHS)
            loss_v = criterion(out["enn_visual"]["alpha"], y_onehot, epoch, cfg.KL_ANNEALING_EPOCHS)
            loss_c = criterion(out["enn_coatt"]["alpha"], y_onehot, epoch, cfg.KL_ANNEALING_EPOCHS)
            loss = loss_fused + 0.2 * (loss_t + loss_v + loss_c)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = torch.argmax(out["probs"], dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, epoch_acc, epoch_f1


def evaluate(model, dataloader, criterion, epoch, cfg):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(cfg.DEVICE)
            attention_mask = batch["attention_mask"].to(cfg.DEVICE)
            images = batch["image"].to(cfg.DEVICE)
            labels = batch["label"].to(cfg.DEVICE)

            with torch.cuda.amp.autocast(enabled=(cfg.DEVICE == "cuda")):
                out = model(input_ids, attention_mask, images)
                y_onehot = F.one_hot(labels, num_classes=cfg.NUM_CLASSES).float()
                alpha_fused = out["b_fusion"] * (cfg.NUM_CLASSES / torch.clamp(out["u_fusion"], min=1e-6)) + 1.0
                loss = criterion(alpha_fused, y_onehot, epoch, cfg.KL_ANNEALING_EPOCHS)

            total_loss += loss.item()
            preds = torch.argmax(out["probs"], dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average="macro")
    return val_loss, val_acc, val_f1


# ============================================================
# STANDALONE ADEF FUSION FUNCTION (NumPy — for tau sweep)
# ============================================================
def adef_fusion_numpy(b_t, u_t, b_v, u_v, b_c, u_c, tau, num_classes=3):
    """
    Recompute ADEF fusion for a given tau value using NumPy arrays.
    All inputs are numpy arrays of shape (N, num_classes) for b and (N, 1) for u.

    Returns:
        preds:        (N,)   — predicted class indices
        probs:        (N, M) — final probability vector p
        K_tv:         (N, 1) — conflict mass values
        route_a_mask: (N, 1) — boolean mask for Route A samples
    """
    # 1. Conflict Mass K_tv = sum_{i != j} b_t_i * b_v_j
    b_t_sum = b_t.sum(axis=-1, keepdims=True)    # (N, 1)
    b_v_sum = b_v.sum(axis=-1, keepdims=True)    # (N, 1)
    K_tv = b_t_sum * b_v_sum - np.sum(b_t * b_v, axis=-1, keepdims=True)  # (N, 1)
    K_tv = np.clip(K_tv, 0.0, 0.9999)

    # ----- Route A: Standard Dempster-Shafer Combination -----
    denom_tv = np.clip(1.0 - K_tv, a_min=1e-8, a_max=None)
    b_tv = (b_t * b_v + b_t * u_v + b_v * u_t) / denom_tv   # (N, M)
    u_tv = (u_t * u_v) / denom_tv                             # (N, 1)

    # Second DS fusion: TV ⊕ C
    b_tv_sum = b_tv.sum(axis=-1, keepdims=True)
    b_c_sum = b_c.sum(axis=-1, keepdims=True)
    K_tvc = b_tv_sum * b_c_sum - np.sum(b_tv * b_c, axis=-1, keepdims=True)
    denom_tvc = np.clip(1.0 - K_tvc, a_min=1e-8, a_max=None)

    b_routeA = (b_tv * b_c + b_tv * u_c + b_c * u_tv) / denom_tvc  # (N, M)
    u_routeA = (u_tv * u_c) / denom_tvc                              # (N, 1)

    # ----- Route B: Adaptive Conflict-Aware Fusion -----
    b_routeB = (1.0 - K_tv) * ((b_t + b_v) / 2.0) + K_tv * b_c      # (N, M)
    u_routeB = 1.0 - np.sum(b_routeB, axis=-1, keepdims=True)        # (N, 1)
    u_routeB = np.clip(u_routeB, a_min=0.0, a_max=None)

    # ----- Dynamic Selection based on tau -----
    route_a_mask = (K_tv <= tau)  # (N, 1) boolean

    b_fusion = np.where(route_a_mask, b_routeA, b_routeB)   # (N, M)
    u_fusion = np.where(route_a_mask, u_routeA, u_routeB)   # (N, 1)

    # Final Probability: p_i = b_fusion_i + u_fusion / M
    probs = b_fusion + (u_fusion / num_classes)              # (N, M)

    preds = np.argmax(probs, axis=-1)  # (N,)

    return preds, probs, K_tv, route_a_mask


# ============================================================
# EXTRACT ENN INTERMEDIATE OUTPUTS
# ============================================================
def extract_enn_outputs(model, dataloader, cfg):
    """
    Run the model in eval mode and collect all ENN intermediate outputs
    (belief masses, uncertainty masses) and ground truth labels.
    """
    model.eval()

    all_b_t, all_u_t = [], []
    all_b_v, all_u_v = [], []
    all_b_c, all_u_c = [], []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting ENN outputs"):
            input_ids = batch["input_ids"].to(cfg.DEVICE)
            attention_mask = batch["attention_mask"].to(cfg.DEVICE)
            images = batch["image"].to(cfg.DEVICE)
            labels = batch["label"].to(cfg.DEVICE)

            out = model(input_ids, attention_mask, images)

            all_b_t.append(out["enn_text"]["b"].cpu().numpy())
            all_u_t.append(out["enn_text"]["u"].cpu().numpy())
            all_b_v.append(out["enn_visual"]["b"].cpu().numpy())
            all_u_v.append(out["enn_visual"]["u"].cpu().numpy())
            all_b_c.append(out["enn_coatt"]["b"].cpu().numpy())
            all_u_c.append(out["enn_coatt"]["u"].cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return {
        "b_t": np.concatenate(all_b_t, axis=0),
        "u_t": np.concatenate(all_u_t, axis=0),
        "b_v": np.concatenate(all_b_v, axis=0),
        "u_v": np.concatenate(all_u_v, axis=0),
        "b_c": np.concatenate(all_b_c, axis=0),
        "u_c": np.concatenate(all_u_c, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
    }


# ============================================================
# TAU GRID SEARCH
# ============================================================
def tau_grid_search(enn_data, tau_candidates, num_classes=3):
    """
    For each candidate tau, recompute the ADEF fusion and evaluate metrics.

    Returns:
        results: list of dicts with tau, f1, accuracy, precision, recall,
                 route_a_pct, route_b_pct
    """
    results = []
    b_t = enn_data["b_t"]
    u_t = enn_data["u_t"]
    b_v = enn_data["b_v"]
    u_v = enn_data["u_v"]
    b_c = enn_data["b_c"]
    u_c = enn_data["u_c"]
    labels = enn_data["labels"]

    for tau in tau_candidates:
        preds, probs, K_tv, route_a_mask = adef_fusion_numpy(
            b_t, u_t, b_v, u_v, b_c, u_c, tau, num_classes
        )

        macro_f1 = f1_score(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
        macro_prec = precision_score(labels, preds, average="macro")
        macro_rec = recall_score(labels, preds, average="macro")

        n_total = len(labels)
        # route_a_mask has shape (N, 1) — count along first column
        n_route_a = route_a_mask[:, 0].sum()
        route_a_pct = n_route_a / n_total * 100
        route_b_pct = (n_total - n_route_a) / n_total * 100

        results.append({
            "tau": tau,
            "macro_f1": macro_f1,
            "accuracy": acc,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "route_a_pct": route_a_pct,
            "route_b_pct": route_b_pct,
        })

        print(f"  τ={tau:.4f} | F1={macro_f1:.4f} | Acc={acc:.4f} | "
              f"Route A: {route_a_pct:.1f}% | Route B: {route_b_pct:.1f}%")

    return results


# ============================================================
# VISUALIZATION: TAU OPTIMIZATION CURVE
# ============================================================
def plot_tau_optimization(results, optimal_tau, save_path=None):
    """
    Generate a publication-quality plot showing:
    - Tau vs Macro F1-Score (left axis)
    - Route A/B distribution (right axis)
    - Highlighted optimal point
    """
    taus = [r["tau"] for r in results]
    f1s = [r["macro_f1"] for r in results]
    accs = [r["accuracy"] for r in results]
    route_a_pcts = [r["route_a_pct"] for r in results]

    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)

    # --- Left axis: F1 and Accuracy ---
    color_f1 = "#2196F3"
    color_acc = "#4CAF50"
    ax1.set_xlabel("Tau (τ) Threshold", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Score", fontsize=13, fontweight="bold")

    line_f1, = ax1.plot(taus, f1s, "o-", color=color_f1, linewidth=2.5,
                        markersize=8, label="Macro F1-Score", zorder=3)
    line_acc, = ax1.plot(taus, accs, "s--", color=color_acc, linewidth=2.0,
                         markersize=6, label="Accuracy", zorder=3)

    # Highlight optimal point
    opt_idx = next(i for i, r in enumerate(results) if r["tau"] == optimal_tau)
    opt_f1 = f1s[opt_idx]
    opt_acc = accs[opt_idx]

    ax1.scatter([optimal_tau], [opt_f1], color="red", s=200, zorder=5,
                edgecolors="darkred", linewidths=2)
    ax1.annotate(
        f"Optimal τ={optimal_tau:.4f}\nF1={opt_f1:.4f}",
        xy=(optimal_tau, opt_f1),
        xytext=(optimal_tau + 0.03, opt_f1 + 0.01),
        fontsize=10, fontweight="bold", color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="red", alpha=0.9),
    )

    # Vertical line at optimal tau
    ax1.axvline(x=optimal_tau, color="red", linestyle=":", alpha=0.5, linewidth=1.5)

    ax1.tick_params(axis="y", labelsize=11)
    ax1.tick_params(axis="x", labelsize=11)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # --- Right axis: Route A percentage ---
    ax2 = ax1.twinx()
    color_route = "#FF9800"
    ax2.set_ylabel("Route A (DS Fusion) %", fontsize=12, color=color_route)
    bar_width = (max(taus) - min(taus)) / (len(taus) * 3) if len(taus) > 1 else 0.01
    bars = ax2.bar(taus, route_a_pcts, width=bar_width, alpha=0.25, color=color_route,
                   label="Route A %", zorder=1)
    ax2.tick_params(axis="y", labelcolor=color_route, labelsize=11)
    ax2.set_ylim(0, 105)

    # Combined legend
    lines = [line_f1, line_acc]
    labels_legend = [l.get_label() for l in lines] + ["Route A %"]
    ax1.legend(lines + [bars], labels_legend, loc="lower right", fontsize=10,
               framealpha=0.9, edgecolor="gray")

    plt.title("Tau (τ) Hyperparameter Optimization — ADEF Model",
              fontsize=15, fontweight="bold", pad=15)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Plot saved to: {save_path}")

    plt.show()
    plt.close(fig)


def print_results_table(results, title="Tau Optimization Results"):
    """Print a nicely formatted results table."""
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)
    print(f"  {'Tau':>8s} | {'Macro F1':>10s} | {'Accuracy':>10s} | {'Precision':>10s} | "
          f"{'Recall':>10s} | {'Route A%':>9s} | {'Route B%':>9s}")
    print("-" * 90)
    for r in results:
        print(f"  {r['tau']:8.4f} | {r['macro_f1']:10.4f} | {r['accuracy']:10.4f} | "
              f"{r['macro_precision']:10.4f} | {r['macro_recall']:10.4f} | "
              f"{r['route_a_pct']:8.1f}% | {r['route_b_pct']:8.1f}%")
    print("=" * 90)


# ============================================================
# FULL RETRAINING WITH OPTIMAL TAU
# ============================================================
def retrain_with_optimal_tau(optimal_tau, train_loader, val_loader, test_loader, cfg):
    """
    Retrain the full ADEF model from scratch with the discovered optimal tau.
    """
    print(f"\n{'='*60}")
    print(f"  RETRAINING MODEL WITH OPTIMAL TAU = {optimal_tau:.4f}")
    print(f"{'='*60}")

    # Update config
    cfg.TAU_THRESHOLD = optimal_tau

    # Build fresh model
    set_seed(cfg.SEED)
    model = ADEFModel(cfg).to(cfg.DEVICE)
    print(f"[INFO] New ADEFModel with tau={optimal_tau:.4f} on device {cfg.DEVICE}")

    # Loss & Optimizer
    edl_criterion = EDLLoss(num_classes=cfg.NUM_CLASSES)

    backbone_params = list(model.text_encoder.parameters()) + list(model.visual_encoder.parameters())
    head_params = (
        list(model.text_proj.parameters()) +
        list(model.visual_proj.parameters()) +
        list(model.co_attention.parameters()) +
        list(model.enn_text.parameters()) +
        list(model.enn_visual.parameters()) +
        list(model.enn_coatt.parameters()) +
        list(model.adef.parameters())
    )
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": cfg.LR_BACKBONE},
        {"params": head_params, "lr": cfg.LR_HEAD}
    ], weight_decay=cfg.WEIGHT_DECAY)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.DEVICE == "cuda"))

    # Training Loop
    history = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_f1": []
    }
    best_val_f1 = 0.0
    best_model_path = os.path.join(cfg.OUTPUT_DIR, f"best_adef_model_tau_{optimal_tau:.4f}.pt")

    for epoch in range(cfg.EPOCHS):
        tr_loss, tr_acc, tr_f1 = train_epoch(
            model, train_loader, edl_criterion, optimizer, scaler, epoch, cfg
        )
        vl_loss, vl_acc, vl_f1 = evaluate(
            model, val_loader, edl_criterion, epoch, cfg
        )

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["val_f1"].append(vl_f1)

        print(f"Epoch {epoch+1:02d}/{cfg.EPOCHS:02d} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} F1: {tr_f1:.4f} | "
              f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f} F1: {vl_f1:.4f}")

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Best Model Saved! Val Macro F1: {best_val_f1:.4f}")

    print(f"[INFO] Retraining complete. Best Val F1: {best_val_f1:.4f}")

    # Test Evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=cfg.DEVICE))
    model.eval()

    test_preds, test_labels_list = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing retrained model"):
            input_ids = batch["input_ids"].to(cfg.DEVICE)
            attention_mask = batch["attention_mask"].to(cfg.DEVICE)
            images = batch["image"].to(cfg.DEVICE)
            labels = batch["label"].to(cfg.DEVICE)

            out = model(input_ids, attention_mask, images)
            preds = torch.argmax(out["probs"], dim=-1)
            test_preds.extend(preds.cpu().numpy())
            test_labels_list.extend(labels.cpu().numpy())

    test_preds = np.array(test_preds)
    test_labels_arr = np.array(test_labels_list)

    test_acc = accuracy_score(test_labels_arr, test_preds)
    test_f1 = f1_score(test_labels_arr, test_preds, average="macro")
    test_prec = precision_score(test_labels_arr, test_preds, average="macro")
    test_rec = recall_score(test_labels_arr, test_preds, average="macro")

    print(f"\n{'='*60}")
    print(f"  RETRAINED MODEL TEST RESULTS (tau={optimal_tau:.4f})")
    print(f"{'='*60}")
    print(f"  Test Accuracy:          {test_acc * 100:.2f}%")
    print(f"  Test Macro Precision:   {test_prec * 100:.2f}%")
    print(f"  Test Macro Recall:      {test_rec * 100:.2f}%")
    print(f"  Test Macro F1-Score:    {test_f1 * 100:.2f}%")
    print(f"{'='*60}")

    target_names = ["Negative", "Neutral", "Positive"]
    print(classification_report(test_labels_arr, test_preds,
                                target_names=target_names, digits=4))

    return model, history, {
        "accuracy": test_acc,
        "macro_f1": test_f1,
        "macro_precision": test_prec,
        "macro_recall": test_rec,
    }


# ============================================================
# MAIN: TAU OPTIMIZATION PIPELINE
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  TAU (τ) HYPERPARAMETER OPTIMIZATION FOR ADEF")
    print("=" * 60)

    # --- Step 1: Prepare Data ---
    print("\n[Step 1/7] Loading data...")
    df_full = load_and_preprocess_dataframe()
    print(f"Total samples after filtering: {len(df_full)}")

    tokenizer = RobertaTokenizer.from_pretrained(CFG.TEXT_BACKBONE)
    image_transform = transforms.Compose([
        transforms.Resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CFG.NORM_MEAN, std=CFG.NORM_STD)
    ])

    train_df, temp_df = train_test_split(
        df_full, test_size=0.30, stratify=df_full["label"], random_state=CFG.SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=CFG.SEED
    )

    train_dataset = MultimodalDataset(train_df, tokenizer, image_transform, max_len=CFG.MAX_LEN)
    val_dataset = MultimodalDataset(val_df, tokenizer, image_transform, max_len=CFG.MAX_LEN)
    test_dataset = MultimodalDataset(test_df, tokenizer, image_transform, max_len=CFG.MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False,
                            num_workers=CFG.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False,
                             num_workers=CFG.NUM_WORKERS, pin_memory=True)

    print(f"[INFO] DataLoaders Ready: Train ({len(train_df)}), "
          f"Val ({len(val_df)}), Test ({len(test_df)})")

    # --- Step 2: Load Trained Model ---
    print("\n[Step 2/7] Loading trained model...")
    model = ADEFModel(CFG).to(CFG.DEVICE)
    best_model_path = os.path.join(CFG.OUTPUT_DIR, "best_adef_model.pt")

    if not os.path.exists(best_model_path):
        print(f"[ERROR] Trained model not found at {best_model_path}")
        print("        Please train the model first using adef.ipynb")
        sys.exit(1)

    model.load_state_dict(torch.load(best_model_path, map_location=CFG.DEVICE))
    print(f"[INFO] Model loaded from {best_model_path}")

    # --- Step 3: Extract ENN Outputs from Validation Set ---
    print("\n[Step 3/7] Extracting ENN intermediate outputs from validation set...")
    val_enn_data = extract_enn_outputs(model, val_loader, CFG)
    print(f"[INFO] Extracted ENN outputs for {len(val_enn_data['labels'])} validation samples")

    # --- Step 4: Coarse Grid Search ---
    print("\n[Step 4/7] Running COARSE tau grid search on validation set...")
    coarse_taus = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    coarse_results = tau_grid_search(val_enn_data, coarse_taus, CFG.NUM_CLASSES)
    print_results_table(coarse_results, "COARSE Grid Search Results (Validation Set)")

    # Find coarse optimum
    coarse_best = max(coarse_results, key=lambda r: r["macro_f1"])
    coarse_optimal_tau = coarse_best["tau"]
    print(f"\n[INFO] Coarse optimal tau: {coarse_optimal_tau:.4f} "
          f"(Val F1 = {coarse_best['macro_f1']:.4f})")

    # --- Step 5: Fine Grid Search ---
    print("\n[Step 5/7] Running FINE tau grid search around coarse optimum...")
    fine_step = 0.005
    fine_range = 0.03  # ±0.03 around coarse optimum
    fine_taus = sorted(set([
        round(coarse_optimal_tau + i * fine_step, 4)
        for i in range(int(-fine_range / fine_step), int(fine_range / fine_step) + 1)
        if round(coarse_optimal_tau + i * fine_step, 4) > 0
    ]))
    # Remove the coarse optimal itself if it's already in the list (it will be re-evaluated)
    print(f"  Fine search candidates: {fine_taus}")
    fine_results = tau_grid_search(val_enn_data, fine_taus, CFG.NUM_CLASSES)
    print_results_table(fine_results, "FINE Grid Search Results (Validation Set)")

    # Find overall optimum
    all_results = coarse_results + fine_results
    # Deduplicate by tau (keep best)
    seen_taus = {}
    for r in all_results:
        t = r["tau"]
        if t not in seen_taus or r["macro_f1"] > seen_taus[t]["macro_f1"]:
            seen_taus[t] = r
    all_results_dedup = sorted(seen_taus.values(), key=lambda r: r["tau"])

    overall_best = max(all_results_dedup, key=lambda r: r["macro_f1"])
    optimal_tau = overall_best["tau"]

    print(f"\n{'='*60}")
    print(f"  OPTIMAL TAU FOUND: τ = {optimal_tau:.4f}")
    print(f"  Validation Macro F1: {overall_best['macro_f1']:.4f}")
    print(f"  Validation Accuracy: {overall_best['accuracy']:.4f}")
    print(f"  Route A: {overall_best['route_a_pct']:.1f}% | "
          f"Route B: {overall_best['route_b_pct']:.1f}%")
    print(f"{'='*60}")

    # --- Step 6: Evaluate Optimal Tau on Test Set ---
    print("\n[Step 6/7] Evaluating optimal tau on TEST set...")
    test_enn_data = extract_enn_outputs(model, test_loader, CFG)

    test_preds, test_probs, test_K_tv, test_route_mask = adef_fusion_numpy(
        test_enn_data["b_t"], test_enn_data["u_t"],
        test_enn_data["b_v"], test_enn_data["u_v"],
        test_enn_data["b_c"], test_enn_data["u_c"],
        optimal_tau, CFG.NUM_CLASSES
    )
    test_labels = test_enn_data["labels"]

    test_f1 = f1_score(test_labels, test_preds, average="macro")
    test_acc = accuracy_score(test_labels, test_preds)
    test_prec = precision_score(test_labels, test_preds, average="macro")
    test_rec = recall_score(test_labels, test_preds, average="macro")

    # Also compute baseline (original tau=0.03) on test set for comparison
    baseline_preds, _, _, _ = adef_fusion_numpy(
        test_enn_data["b_t"], test_enn_data["u_t"],
        test_enn_data["b_v"], test_enn_data["u_v"],
        test_enn_data["b_c"], test_enn_data["u_c"],
        0.03, CFG.NUM_CLASSES
    )
    baseline_f1 = f1_score(test_labels, baseline_preds, average="macro")
    baseline_acc = accuracy_score(test_labels, baseline_preds)

    print(f"\n{'='*60}")
    print(f"  TEST SET RESULTS — COMPARISON")
    print(f"{'='*60}")
    print(f"  Baseline (τ=0.03):")
    print(f"    Accuracy:  {baseline_acc * 100:.2f}%")
    print(f"    Macro F1:  {baseline_f1 * 100:.2f}%")
    print(f"  Optimal  (τ={optimal_tau:.4f}):")
    print(f"    Accuracy:  {test_acc * 100:.2f}%")
    print(f"    Macro F1:  {test_f1 * 100:.2f}%")
    print(f"    Precision: {test_prec * 100:.2f}%")
    print(f"    Recall:    {test_rec * 100:.2f}%")
    f1_delta = (test_f1 - baseline_f1) * 100
    acc_delta = (test_acc - baseline_acc) * 100
    print(f"  Improvement: F1 {f1_delta:+.2f}%, Acc {acc_delta:+.2f}%")
    print(f"{'='*60}")

    target_names = ["Negative", "Neutral", "Positive"]
    print("\nFull Test Classification Report (optimal tau):")
    print(classification_report(test_labels, test_preds,
                                target_names=target_names, digits=4))

    # --- Step 7: Generate Visualization ---
    print("\n[Step 7/7] Generating visualization...")
    plot_path = os.path.join(CFG.OUTPUT_DIR, "tau_optimization_curve.png")
    plot_tau_optimization(all_results_dedup, optimal_tau, save_path=plot_path)

    # Print combined table
    print_results_table(all_results_dedup, "ALL TAU CANDIDATES — COMPLETE RESULTS")

    # --- Step 8 (Optional): Retrain with Optimal Tau ---
    print(f"\n{'='*60}")
    print(f"  RETRAINING WITH OPTIMAL TAU = {optimal_tau:.4f}")
    print(f"{'='*60}")

    retrained_model, retrain_history, retrain_test_metrics = retrain_with_optimal_tau(
        optimal_tau, train_loader, val_loader, test_loader, CFG
    )

    # --- Final Summary ---
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY — TAU OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Baseline model (τ=0.03):")
    print(f"    Test Accuracy:  {baseline_acc * 100:.2f}%")
    print(f"    Test Macro F1:  {baseline_f1 * 100:.2f}%")
    print(f"")
    print(f"  Post-hoc optimal (τ={optimal_tau:.4f}, same model weights):")
    print(f"    Test Accuracy:  {test_acc * 100:.2f}%")
    print(f"    Test Macro F1:  {test_f1 * 100:.2f}%")
    print(f"")
    print(f"  Retrained model (τ={optimal_tau:.4f}, fresh training):")
    print(f"    Test Accuracy:  {retrain_test_metrics['accuracy'] * 100:.2f}%")
    print(f"    Test Macro F1:  {retrain_test_metrics['macro_f1'] * 100:.2f}%")
    print(f"    Test Precision: {retrain_test_metrics['macro_precision'] * 100:.2f}%")
    print(f"    Test Recall:    {retrain_test_metrics['macro_recall'] * 100:.2f}%")
    print(f"")
    print(f"  Best model saved at:")
    print(f"    {os.path.join(CFG.OUTPUT_DIR, f'best_adef_model_tau_{optimal_tau:.4f}.pt')}")
    print(f"  Optimization plot saved at:")
    print(f"    {plot_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
