**Role & Task:**
Act as a Principal AI Research Engineer specializing in Multimodal Deep Learning, Evidential Deep Learning (EDL), and PyTorch architecture design. Your task is to write a complete, modular, publication-ready PyTorch implementation of the **Adaptive Evidential Multimodal Sentiment Analysis Pipeline with Gated Calibration Buffer**.
**Architecture Blueprint & Mathematical Specifications:**

>

1. **Dual-Stream Feature Extraction Stage:**
   >

- **Text Stream:** Use HuggingFace `RoBERTa` (`roberta-base`) to process tokenized text inputs $T$. Extract token sequence embeddings $\mathbf{H}_t \in \mathbb{R}^{B \times N \times d}$ and the summary embedding $\mathbf{h}_t^{cls} \in \mathbb{R}^{B \times d}$ from the `[CLS]` token.
- **Vision Stream:** Use HuggingFace `CLIP-ViT` (`openai/clip-vit-base-patch32`) to process image patch inputs $I$. Extract spatial patch embeddings $\mathbf{H}_v \in \mathbb{R}^{B \times M \times d}$ and visual pooled output $\mathbf{h}_v^{cls} \in \mathbb{R}^{B \times d}$.
- Add linear projection layers if needed so both modal representations match hidden dimension $d$ (e.g., $d = 768$).
  >

2. **Bidirectional Multi-Head Co-Attention Layer:**
   >

- Implement a cross-attention module that computes key-query alignment between $\mathbf{H}_t$ and $\mathbf{H}_v$:
  > $$\mathbf{A}^{t \to v} = \text{Softmax}\left( \frac{(\mathbf{H}_t \mathbf{W}_Q) (\mathbf{H}_v \mathbf{W}_K)^T}{\sqrt{d_k}} \right) \in \mathbb{R}^{B \times N \times M}$$
- Aggregate values $\mathbf{H}_v \mathbf{W}_V$ using $\mathbf{A}^{t \to v}$ and mean-pool across tokens $N$ to produce a dense fused cross-attended representation vector $\mathbf{z}_{att} \in \mathbb{R}^{B \times d}$.
  >

3. **Adaptive Gated Calibration Buffer (Core Proposed Module):**
   Implement a module `GatedCalibrationBuffer` that takes $\mathbf{z}_{att}$, $\mathbf{h}_t^{cls}$, and $\mathbf{h}_v^{cls}$ as inputs and computes a calibration multiplier $g_{cal} \in [0, 1]^{B \times 1}$:
   >

- **Sub-Branch A (Non-Parametric Informational Saliency $\Phi$):**
- Pass $\mathbf{h}_t^{cls}$ and $\mathbf{h}_v^{cls}$ through auxiliary linear probes $\mathbf{W}_{probe,t}$ and $\mathbf{W}_{probe,v}$ to obtain probability predictions $\mathbf{p}_t, \mathbf{p}_v \in \mathbb{R}^{B \times K}$ (for $K=3$ sentiment classes).
- Compute average normalized Shannon entropy:
  > $$\bar{H} = \frac{-1}{2 \log_2(K)} \sum_{k=1}^K \left( p_{t,k} \log_2(p_{t,k} + \epsilon) + p_{v,k} \log_2(p_{v,k} + \epsilon) \right) \in [0, 1]$$
- Compute cross-modal cosine similarity concordance:
  > $$C_{cross} = \frac{\mathbf{h}_t^{cls} \cdot \mathbf{h}_v^{cls}}{\Vert{}\mathbf{h}_t^{cls}\Vert{}_2 \Vert{}\mathbf{h}_v^{cls}\Vert{}_2 + \epsilon} \in [-1, 1]$$
- Compute non-parametric saliency multiplier:
  > $$\Phi = (1 - \bar{H}) \cdot \left( \frac{1 + C_{cross}}{2} \right) \in [0, 1]$$
- **Sub-Branch B (Parametric Relational Context Gate $\gamma$):**
- Construct interaction context vector $\mathbf{r} = \left[ \mathbf{z}_{att} \, \Vert \, \vert{}\mathbf{h}_t^{cls} - \mathbf{h}_v^{cls}\vert{} \, \Vert \, (\mathbf{h}_t^{cls} \odot \mathbf{h}_v^{cls}) \right] \in \mathbb{R}^{B \times 3d}$.
- Pass $\mathbf{r}$ through Linear + Sigmoid activation to obtain $\gamma = \sigma(\mathbf{W}_\gamma \mathbf{r} + b_\gamma) \in (0, 1)$.
- **Composite Calibration Factor:**
  > $$g_{cal} = \gamma \cdot \Phi \in [0, 1]$$

4. **Evidential Head & Subjective Logic Layer:**
   >

- Compute uncalibrated logits $\mathbf{v}_e = \mathbf{W}_e \mathbf{z}_{att} + \mathbf{b}_e \in \mathbb{R}^{B \times K}$.
- Apply Gated Softplus mapping to derive class evidence parameters $e_k$:
  > $$e_k = g_{cal} \cdot \text{Softplus}(v_{e,k}) = g_{cal} \cdot \ln\left(1 + \exp(v_{e,k})\right) \ge 0$$
- Construct Dirichlet distribution parameters:
  > $$\alpha_k = e_k + 1 \implies \boldsymbol{\alpha} = [\alpha_1, \dots, \alpha_K]^T$$
- Derive Subjective Logic parameters:
- Total Dirichlet Strength: $S = \sum_{k=1}^K \alpha_k = \sum_{k=1}^K e_k + K$
- Class Belief Mass: $b_k = \frac{e_k}{S}$
- Epistemic Uncertainty Mass: $u = \frac{K}{S} \in (0, 1]$
- Expected Probability Point Estimates: $\hat{p}_k = \frac{\alpha_k}{S}$
  >

5. **Custom Evidential Loss Function (`EvidentialLoss` Module):**
   Implement a dedicated loss module computing the combined Evidential Deep Learning loss:
   > $$\mathcal{L}_{EDL} = \mathcal{L}_{MSE} + \lambda_{KL} \cdot \mathcal{L}_{KL} + \lambda_{aux} \cdot (\mathcal{L}_{CE,t} + \mathcal{L}_{CE,v})$$

- **Expected Mean Squared Error Loss ($\mathcal{L}_{MSE}$):**
  > $$\mathcal{L}_{MSE} = \sum_{i=1}^B \sum_{k=1}^K \left( y_{i,k} - \hat{p}_{i,k} \right)^2 + \frac{\hat{p}_{i,k} (1 - \hat{p}_{i,k})}{S_i + 1}$$
  >
  > where $\mathbf{y}_i$ is the one-hot target vector.
- **KL Divergence Regularization Loss ($\mathcal{L}_{KL}$):**
  Regularizes non-target class evidence toward the uniform Dirichlet prior $\text{Dir}(\mathbf{1})$:
  > $$\tilde{\boldsymbol{\alpha}}_i = \mathbf{y}_i + (1 - \mathbf{y}_i) \odot \boldsymbol{\alpha}_i$$
  >
  > $$\mathcal{L}_{KL} = \text{KL}\left( \text{Dir}(\mathbf{p}_i \mid \tilde{\boldsymbol{\alpha}}_i) \parallel \text{Dir}(\mathbf{p}_i \mid \mathbf{1}) \right)$$
  >
  > Use an annealing weight $\lambda_{KL} = \min\left(1.0, \frac{\text{epoch}}{\text{max\_anneal\_epochs}}\right)$.
- **Auxiliary Cross-Entropy Loss ($\mathcal{L}_{aux}$):**
  Standard Cross-Entropy loss computed on $\mathbf{p}_t$ and $\mathbf{p}_v$ against ground-truth labels $y$ to train the unimodal auxiliary probes.
  > **Code Requirements:**
- Write modular, highly clean PyTorch code using `torch.nn.Module`.
- Include clear docstrings for classes and methods.
- Add explicit shape comments on tensor operations inside `forward()` passes (e.g., `# [B, N, d]`).
- The model's `forward()` method should accept `input_ids`, `attention_mask`, `pixel_values`, and optional ground-truth `labels`.
- Return a structured `dict` containing: `logits`, `evidence`, `alpha`, `belief`, `uncertainty`, `expected_p`, `g_cal`, `saliency_phi`, `gate_gamma`, and `loss` (if labels are provided).
- Write in file adef_gated_calibration_buffer.ipynb and use the following loaded dataset template in this file.
