# Analisis Komparatif Mekanisme Co-Attention: Single-Head vs Multi-Head

Dokumen ini menyajikan perbandingan teknis, matematis, dan arsitektural antara dua mekanisme Co-Attention yang diuji dalam penelitian **Adaptive Evidential Fusion (ADEF)**:
1. **Single-Head Co-Attention** pada file: [`adef-clara-singlehead-co-attention.ipynb`](file:///d:/Coding/Project/All%20Thesis/Adaptive%20Evidential%20Fusion/adef-clara-singlehead-co-attention.ipynb)
2. **Multi-Head Co-Attention (CLARA Style)** pada file: [`adef-clara-multihead-co-attention.ipynb`](file:///d:/Coding/Project/All%20Thesis/Adaptive%20Evidential%20Fusion/adef-clara-multihead-co-attention.ipynb)

---

## 1. Ikhtisar Arsitektural

```
Single-Head Co-Attention (ADEF Original)
[Teks h_t] ---\
               ===> [Single Matrix Product A = Softmax(h_t W h_v^T / sqrt(D))] ===> [Concat & Mean Pool] ===> h_c
[Gambar h_v] --/

Multi-Head Co-Attention (CLARA Style)
[Teks h_t]   ===> [Layer 1: 8-Head T->V & V->T + Residual + GELU FFN] ===> [Layer 2: 8-Head T->V & V->T + Residual + GELU FFN] ===> h_c
[Gambar h_v] ===/
```

---

## 2. Perbandingan Per Fitur Kunci & Kutipan Kode PyTorch

### Fitur 1: Proyeksi Sub-Space (Single-Head vs 8 Multi-Heads)

* **Single-Head (`adef-clara-singlehead-co-attention.ipynb`)**:
  Menggunakan satu matriks linier tunggal `self.W` untuk memetakan seluruh dimensi fitur ($D=512$). Model hanya dapat memperhatikan **satu pola asosiasi spasial-tekstual** dalam satu waktu.

  ```python
  # Kutipan Kode: Single-Head Co-Attention
  class CoAttentionModule(nn.Module):
      def __init__(self, feature_dim=512):
          super(CoAttentionModule, self).__init__()
          # Satu matriks linier tunggal untuk interaksi
          self.W = nn.Linear(feature_dim, feature_dim, bias=False)

      def forward(self, h_t, h_v):
          # Hitung skor matriks korelasi tunggal (B x L x N)
          h_v_proj = self.W(h_v)
          scores = torch.matmul(h_t, h_v_proj.transpose(1, 2)) / (h_t.size(-1) ** 0.5)
          A = F.softmax(scores, dim=-1)
  ```

* **Multi-Head (`adef-clara-multihead-co-attention.ipynb`)**:
  Membagi dimensi fitur ($D=512$) menjadi 8 sub-space berdimensi $d_k = 64$. Hal ini memungkinkan model memperhatikan **8 aspek interaksi berbeda secara simultan** (misal: Head 1 fokus pada ekspresi muka vs kata sifat, Head 2 fokus pada latar belakang gambar vs objek teks).

  ```python
  # Kutipan Kode: Multi-Head Co-Attention
  class MultiHeadCoAttentionLayer(nn.Module):
      def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1):
          super(MultiHeadCoAttentionLayer, self).__init__()
          # Text queries Vision via 8 Attention Heads
          self.text_to_vision_attn = nn.MultiheadAttention(
              embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
          )
          # Vision queries Text via 8 Attention Heads
          self.vision_to_text_attn = nn.MultiheadAttention(
              embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
          )
  ```

---

### Fitur 2: Kedalaman & Hirarki Interaksi (1 Layer vs 2-Layer Stacked)

* **Single-Head (`adef-clara-singlehead-co-attention.ipynb`)**:
  Interaksi dilakukan secara **satu kali (*single-pass*)**, langsung dilanjutkan dengan *mean pooling* rata-rata. Interaksi berkedalaman dangkal ini sulit menangkap konteks emosi yang bertingkat/implisit.

  ```python
  # Kutipan Kode: Single-Pass Aggregation & Mean Pooling
  # Teks yang dipengaruhi gambar & Gambar yang dipengaruhi teks
  h_t_att = torch.matmul(A, h_v)
  h_v_att = torch.matmul(A.transpose(1, 2), h_t)

  # Langsung di-mean pooling tanpa penumpukan layer
  bar_h_t = h_t_att.mean(dim=1)
  bar_h_v = h_v_att.mean(dim=1)
  ```

* **Multi-Head (`adef-clara-multihead-co-attention.ipynb`)**:
  Menggunakan **2 Layer Co-Attention Bertumpuk (*Stacked*)**. Layer 1 menangkap jajaran kata-patch lokal (*direct region-phrase alignment*), sedangkan Layer 2 menyaring konteks abstrak semantik (*higher-order reasoning*).

  ```python
  # Kutipan Kode: 2-Layer Stacked Execution pada ADEFCLARAModel
  self.co_attention_layers = nn.ModuleList([
      MultiHeadCoAttentionLayer(hidden_dim=cfg.FEATURE_DIM, num_heads=cfg.NUM_HEADS, dropout=0.1)
      for _ in range(cfg.NUM_COATT_LAYERS) # NUM_COATT_LAYERS = 2
  ])

  # Eksekusi bertumpuk di forward pass
  h_t_att, h_v_att = h_t_seq, h_v_spatial
  for layer in self.co_attention_layers:
      h_t_att, h_v_att = layer(h_t_att, h_v_att)
  ```

---

### Fitur 3: Transformasi Fitur Non-Linier (Simple Linear vs GELU FFN)

* **Single-Head (`adef-clara-singlehead-co-attention.ipynb`)**:
  Transformasi non-linier pasca-attention hanya mengandalkan satu proyeksi linier `self.proj_c` dan modul `ReLU`.

  ```python
  # Kutipan Kode: Simple Linear + ReLU
  h_c_cat = torch.cat([bar_h_t, bar_h_v], dim=-1)
  h_c = self.dropout(F.relu(self.proj_c(h_c_cat)))
  h_c = self.layer_norm(h_c)
  ```

* **Multi-Head (`adef-clara-multihead-co-attention.ipynb`)**:
  Dilengkapi dengan **Feed-Forward Networks (FFN)** beraktivasi GELU pada setiap jalur modalitas ($512 \to 2048 \to 512$), memberikan kapasitas transformasi non-linier yang jauh lebih fleksibel.

  ```python
  # Kutipan Kode: GELU Feed-Forward Network (FFN)
  self.text_ffn = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim * 4), # 512 -> 2048
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim * 4, hidden_dim), # 2048 -> 512
      nn.Dropout(dropout)
  )
  ```

---

### Fitur 4: Aliran Gradien & Stabilitas Training (Residual vs Non-Residual)

* **Single-Head (`adef-clara-singlehead-co-attention.ipynb`)**:
  Tidak memiliki sambungan *residual connection*. Informasi asli dari fitur teks dan gambar rentan mengalami degradasi saat melewati matriks perkalian $A$.

* **Multi-Head (`adef-clara-multihead-co-attention.ipynb`)**:
  Menerapkan **Residual Connections (`+`) dan Layer Normalization** pada setiap blok attention dan FFN, menjamin aliran gradien tetap stabil saat fine-tuning backbone besar (DeBERTa-v3 & CLIP).

  ```python
  # Kutipan Kode: Residual Connection + Layer Normalization
  # Pass 1: Multi-Head Attention dengan Residual + LayerNorm
  text_attn_out, _ = self.text_to_vision_attn(query=text_features, key=vision_features, value=vision_features)
  text_features = self.text_norm(text_features + text_attn_out)

  # Pass 2: FFN dengan Residual + LayerNorm
  text_out = self.text_ffn_norm(text_features + self.text_ffn(text_features))
  ```

---

## 3. Matriks Perbandingan Ringkas

| Dimensi Evaluasi | Single-Head Co-Attention | Multi-Head Co-Attention (CLARA) |
| :--- | :--- | :--- |
| **Lokasi File** | [`adef-clara-singlehead-co-attention.ipynb`](file:///d:/Coding/Project/All%20Thesis/Adaptive%20Evidential%20Fusion/adef-clara-singlehead-co-attention.ipynb) | [`adef-clara-multihead-co-attention.ipynb`](file:///d:/Coding/Project/All%20Thesis/Adaptive%20Evidential%20Fusion/adef-clara-multihead-co-attention.ipynb) |
| **Jumlah Attention Heads** | 1 Head ($D=512$) | **8 Heads** ($d_k=64$ per head) |
| **Kedalaman Layer** | 1 Pass Single Matrix | **2 Layer Stacked** |
| **Transformasi Fitur** | Linear Proj + ReLU | **GELU Feed-Forward Network (FFN)** |
| **Stabilitas Gradien** | Standard | **Residual Connections + LayerNorm** |
| **Peluang Overfitting** | Lebih mudah terjebak di lokal minima | Lebih stabil karena regularisasi residual |
| **Peran dalam Tesis** | **Baseline Ablation Study** | **Proposed Full Hybrid Model** |

---

## 4. Kesimpulan untuk Pembahasan Tesis

Perbandingan ini memberikan dasar ilmiah yang kuat untuk **Bab 4 (Hasil dan Pembahasan)** tesis Anda:
* Pengujian pada `adef-clara-singlehead-co-attention.ipynb` mengisolasi kontribusi peningkatan yang berasal murni dari **Backbone (CLIP + DeBERTa-v3) & LoRA**.
* Pengujian pada `adef-clara-multihead-co-attention.ipynb` mengukur peningkatan tambahan yang dihasilkan oleh **Multi-Head Co-Attention 2-Layer**, membuktikan bahwa ekstraksi fitur interaksi yang lebih dalam dan multi-perspektif secara signifikan meningkatkan akurasi dan F1-score sentiment analysis multimodal.
