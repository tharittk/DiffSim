# Research Notes: 2D Conditional Diffusion for Seismic Facies Generation

## Table of Contents
1. [Project Goal & Architecture Summary](#1-project-goal--architecture-summary)
2. [Critical Pitfalls in Current Implementation](#2-critical-pitfalls-in-current-implementation)
3. [Best Practices for Conditional Diffusion Models](#3-best-practices-for-conditional-diffusion-models)
4. [Ideas to Try](#4-ideas-to-try)
5. [Training & Data Considerations](#5-training--data-considerations)
6. [Evaluation Strategy](#6-evaluation-strategy)
7. [Deployment Considerations](#7-deployment-considerations)

---

## 1. Project Goal & Architecture Summary

**Task:** Given a real RMS amplitude map (continuous 2D image from seismic data) + sparse well facies observations (a handful of known pixel locations with categorical labels), generate plausible 2D channel facies maps with 3 classes: sand, bank, mud.

**Architecture:** Conditional DDPM/DDIM with a guided-diffusion UNet.

**Conditioning (6 channels):**
| Channel | Content | Range | Coverage |
|---------|---------|-------|----------|
| 0 | RMS amplitude | [-1, 1] continuous | Full image |
| 1 | Well presence mask | {0, 1} binary | Sparse (~3-15 pixels in 64×64) |
| 2 | Sand indicator at wells | {0, 1} binary | Sparse |
| 3 | Bank indicator at wells | {0, 1} binary | Sparse |
| 4 | Mud indicator at wells | {0, 1} binary | Sparse |
| 5 | y_t (current noisy state) | [-1, 1] continuous | Full image |

**Output:** 1-channel facies map in [-1, 1], post-processed to discrete {mud=-1, bank=0, sand=1}.

---

## 2. Critical Pitfalls in Current Implementation

### 2.1 ⛔ Attention Block Never Fires in Encoder/Decoder

**Severity: HIGH — Architecture is broken.**

The config specifies `attn_res: [16]` with `channel_mults: [1, 2, 4]`. However, the code compares the **downsampling factor** `ds` (which takes values 1, 2, 4, 8) against `attn_res`. Since `16 ∉ {1, 2, 4, 8}`, attention **never activates** in the encoder or decoder — only the middle block gets one unconditional attention layer.

```python
# In UNet.__init__()
if ds in attn_res:     # ds ∈ {1, 2, 4, 8}, attn_res = [16]  ⟹  NEVER TRUE
    layers.append(AttentionBlock(...))
```

**Impact:** The model has essentially no long-range spatial reasoning in its encoder/decoder path. For facies prediction, channels and geological bodies can span 10-40 pixels — the receptive field of convolutions alone may be insufficient.

**Fix:** Change `attn_res: [16]` to `attn_res: [4]` (which means "apply attention when `ds=4`", i.e., at 16×16 spatial resolution). Or even `attn_res: [2, 4]` to get attention at both 32×32 and 16×16.

```json
"attn_res": [4]       // attention at 16×16 resolution (ds=4)
"attn_res": [2, 4]    // attention at 32×32 AND 16×16
```

### 2.2 ⛔ Masked Loss Not Normalized by Mask Area

**Severity: HIGH — Training signal is wrong.**

The loss computation applies `mask * noise` and `mask * model_output` but uses standard `MSELoss()` which averages over **all** elements (including zeros from masking):

```python
loss = self.loss_fn(mask * noise, mask * model_output)  # MSELoss averages over ALL pixels
```

With the Flumy pipeline, the mask is ~1 everywhere except ~10 wells out of 4096 pixels (64×64). So `mask` is ~99.8% ones. The loss is **dominated by non-well regions** while the well constraint contributes almost nothing. This is actually the correct behavior for facies generation (we want to predict the full map, with wells as known anchors). However, this means the wells really function **only through the conditioning channels**, not through the masked loss.

**But consider:** If you change the conditioning to have dense conditioning (e.g., assigning well confidence that spreads from well locations), the mask contribution becomes more significant.

**Detailed analysis:** For the current sparse-well setup, the mask is almost entirely 1 (generate everywhere), so the loss is effectively unmasked — which is fine. The `(1-mask) * y_0` term in the model input gives ground truth at wells during training, and the loss ignores those locations since loss is only computed where `mask=1`. This is actually a reasonable inpainting setup.

### 2.3 ⚠️ Train-Test Discrepancy (Exposure Bias)

**Severity: MEDIUM**

During training:
```python
model_input = cat([y_cond, y_noisy * mask + (1-mask) * y_0], dim=1)
```
The model sees **ground truth `y_0`** at well locations (where `mask=0`). This is fine during training.

During inference (restoration):
```python
y_t = y_0 * (1-mask) + mask * y_t   # y_0 = known wells values
```
Here `y_0` is only the well observations — the model still gets ground truth at wells (which is correct). However, the known regions are replaced with **clean** ground truth at every step while surrounding regions are noisy. This sharp boundary between clean known values and noisy generated values can cause artifacts.

**This is a known limitation of vanilla inpainting diffusion.** The RePaint approach (Lugmayr et al., 2022) addresses this by re-noising known regions to match the noise level at each timestep.

### 2.4 ⚠️ RMS Normalization May Not Generalize to Real Data

**Severity: HIGH for deployment.**

The synthetic RMS is normalized to [-1, 1] by min-max scaling each sample independently:
```python
rms = (rms - rms.min()) / (rms.max() - rms.min()) * 2 - 1
```

**Problem:** When deploying with real seismic data, the absolute range and distribution of RMS values will differ from synthetic data. If the model has only seen synthetic RMS distributions during training, it may fail to generalize to real seismic textures.

**Considerations:**
- Real RMS has noise, acquisition footprints, amplitude anomalies, phase distortions, and frequency-dependent effects that the simple forward model doesn't produce.
- Real RMS has spatial coherence patterns at scales that differ from Flumy grid resolution.
- Solution: Domain adaptation, style transfer, or more realistic forward modeling.

### 2.5 ⚠️ Facies Output is Continuous but Task is Categorical

**Severity: MEDIUM**

The model outputs continuous values in [-1, 1], which are then discretized by nearest-neighbor to {mud=-1, bank=0, sand=1}. This is a misalignment:

- The model is trained with MSE/L1 loss on continuous targets, optimizing for regression not classification.
- Intermediate values (e.g., -0.3) don't have physical meaning but contribute to the loss.
- The discretization step is not differentiable, so the model can't learn the decision boundaries.

**Alternatives:**
- **Multi-channel categorical output:** Output 3 channels with softmax → per-class probabilities. Standard cross-entropy loss.
- **Straight-through estimator:** Keep continuous training but add a discretization-aware loss term.
- **Post-hoc thresholding:** Current approach. Simple and actually works reasonably well in practice because the diffusion process naturally sharpens to discrete values near [-1, 0, 1].

### 2.6 ⚠️ Seismic Forward Model is Oversimplified

**Severity: MEDIUM — affects domain gap.**

The synthetic RMS pipeline:
1. Assigns constant AI per facies class (sand=7000, bank=8500, mud=10000)
2. Computes reflectivity from AI contrasts
3. Convolves with a Ricker wavelet (single frequency)
4. Computes RMS over a depth window
5. Adds Gaussian noise + smoothing

**Missing from real seismic:**
- **Tuning effects**: Thin beds below seismic resolution cause constructive/destructive interference. Real RMS is highly sensitive to layer thickness.
- **Frequency-dependent attenuation (Q)**: Higher frequencies absorbed more.
- **AVO effects**: Amplitude varies with angle/offset.
- **Migration artifacts**: Imperfect focusing, sidelobes.
- **Noise character**: Real seismic noise is correlated, not white Gaussian.
- **Multiples**: Internal reverberations between layers.
- **3D effects**: Even for 2D plan view, the RMS is affected by out-of-plane reflections.

### 2.7 ⚠️ No Data Augmentation Beyond Well Re-sampling

**Severity: LOW-MEDIUM**

The FlumyDataset re-samples well locations each epoch (good), but there's no:
- Random rotation (90°, 180°, 270°) — channel patterns should be rotation-invariant
- Random horizontal/vertical flips
- Random cropping/padding
- Elastic deformation
- RMS augmentation (brightness/contrast jitter to simulate domain shift)

---

## 3. Best Practices for Conditional Diffusion Models

### 3.1 Classifier-Free Guidance (CFG)

**Key idea:** During training, randomly drop conditioning with probability `p_uncond` (e.g., 10-20%). At inference, interpolate between conditional and unconditional predictions:

```
ε_guided = ε_uncond + w * (ε_cond - ε_uncond)
```

Where `w` is the guidance scale (w=1 means standard conditioning, w>1 amplifies conditioning influence).

**Why it matters for this project:**
- With `w > 1`, the model produces facies that more strongly correlate with the RMS input.
- With `w ≈ 1`, diversity is maintained but conditioning fidelity may be weaker.
- Gives you a knob to control the **fidelity-diversity tradeoff** at inference time.

**Implementation:**
- During training: 10-20% of the time, replace conditioning channels (RMS + wells) with zeros.
- During inference: Run model twice (with and without conditioning), combine outputs.
- Cost: Doubles inference time per step, but enables powerful conditioning control.

### 3.2 v-prediction (Salimans & Ho, 2022)

Instead of predicting noise `ε` or `x_0`, predict `v`:
```
v = √(α_t) * ε - √(1 - α_t) * x_0
```

**Benefits:**
- More stable training at high and low noise levels.
- Better color/value fidelity (important for categorical outputs).
- Smoother loss landscape near t=0 and t=T.

**Current code** supports `epsilon` and `x_start` prediction types. Adding `v_prediction` requires changes to `network.py` forward/sampling methods.

### 3.3 Cosine Schedule is Correct

The current implementation uses cosine beta schedule (Nichol & Dhariwal, 2021). This is the right choice for 64×64 images — cosine preserves more signal at high noise levels compared to linear, leading to better detail preservation. ✅

### 3.4 EMA is Essential

Exponential Moving Average of model weights (`ema_decay: 0.9999`) is correctly implemented. The EMA model typically produces sharper, more consistent outputs. Always use EMA for evaluation/inference. ✅

### 3.5 Progressive Training (Optional)

For larger image sizes (128×128+):
- Start training at 64×64 with more steps, then fine-tune at higher resolution.
- This is particularly useful if you later want to increase resolution.

### 3.6 DDIM with η=0 for Deterministic Outputs

Setting `eta=0` in DDIM makes sampling deterministic given the same initial noise. This enables:
- **Latent space interpolation**: Interpolate between two noise seeds to smoothly morph between realizations.
- **Reproducibility**: Same noise → same realization.

The current code supports this. ✅

---

## 4. Ideas to Try

### 4.1 ★★★ Fix Attention Resolution (Quick Win)

Change `attn_res` from `[16]` to `[4]` (or `[2, 4]`). This enables self-attention at 16×16 (and 32×32) spatial resolution, allowing the model to capture long-range correlations between channel bodies.

**Expected impact:** Significantly better spatial coherence of generated facies, especially for long sinuous channel bodies that span the image.

### 4.2 ★★★ Implement Classifier-Free Guidance

Add conditioning dropout during training (zero-out conditioning channels with probability 0.1-0.2). At inference, use guidance scale w=2-5.

**Expected impact:** Better fidelity to RMS input while maintaining diversity across realizations.

### 4.3 ★★★ Use Cross-Attention for RMS Conditioning

Instead of channel-concatenation (cramming everything into 6 channels), use cross-attention to inject the RMS map:

```
Modify UNet to:
1. Encode RMS map with a separate lightweight encoder → spatial features
2. Use cross-attention in UNet decoder blocks: Query=UNet features, Key/Value=RMS features
3. Keep well conditioning as channel-concatenation (sparse, different modality)
```

**Why:** Channel concatenation forces the network to learn how to combine RMS with noise in early layers. Cross-attention gives the model explicit spatial correlation between generated facies and seismic data at every resolution level.

**Expected impact:** Better utilization of RMS information, more physics-informed generation.

### 4.4 ★★ Multi-Scale RMS Features

Instead of feeding raw RMS as one channel, compute features at multiple scales:
```
Channel 0: RMS (original)
Channel 1: RMS (smoothed with σ=2)
Channel 2: RMS gradient magnitude (edge detection)
Channel 3: RMS Laplacian (curvature)
```

This gives the network explicit multi-frequency information about the seismic character.

### 4.5 ★★ Larger Training Dataset

100 simulations × 10 slices = ~1000 samples (after skipping uniform slices) is quite small for a diffusion model. Consider:
- **500-1000 simulations** with 20 slices each → 10K-20K samples.
- **Varied geological scenarios**: Different `hmax`, `ng`, `isbx` ranges to cover more depositional styles.
- **Multiple image sizes**: Train at 64×64, then fine-tune at 128×128 for higher resolution.

### 4.6 ★★ RePaint-Style Inpainting

Current inference just replaces known pixels at each step:
```python
y_t = y_0 * (1-mask) + mask * y_t   # Sharp stitching
```

RePaint (Lugmayr et al., 2022) approach:
```python
for each timestep t:
    # Forward: re-add noise to known pixels to match noise level at t
    y_known_noised = sqrt(α_t) * y_0 + sqrt(1-α_t) * noise
    y_t = y_known_noised * (1-mask) + mask * y_t
    
    # Then denoise as usual
    y_t = p_sample(y_t, t, y_cond)
```

**Why:** This ensures known and unknown regions have the same noise level at each step, avoiding boundary artifacts.

### 4.7 ★★ Well Influence Radius

Instead of sparse single-pixel well locations, create a "well influence" zone that decays away from the well:

```python
# Gaussian decay from well location
for r, c in well_positions:
    influence = exp(-((rows - r)² + (cols - c)²) / (2 * σ²))
    well_confidence += influence
```

This gives the model a softer constraint that transitions gradually from "known at well" to "unknown far from well", which may produce smoother and more geologically realistic results.

### 4.8 ★ Perceptual / Structural Losses

Add auxiliary losses beyond pixel MSE:
- **SSIM loss**: Preserves structural patterns (boundaries between facies).
- **Frequency loss**: Compare Fourier spectra to ensure correct spatial statistics.
- **Facies proportion loss**: Penalize if generated sand/mud/bank proportions deviate from expected (net-to-gross).

### 4.9 ★ Domain Adaptation for Real Data

The biggest risk is the domain gap between synthetic and real RMS. Options:
- **Fine-tuning**: If some real labeled data available, fine-tune with a few hundred examples.
- **Style transfer on RMS**: Transform synthetic RMS to look like real seismic (histogram matching, neural style transfer, CycleGAN).
- **Noise augmentation during training**: Add realistic seismic noise patterns (coherent noise, acquisition footprint) to synthetic RMS during training.
- **RMS-agnostic training**: Occasionally drop the RMS channel (→ CFG) so model doesn't over-rely on synthetic-only RMS patterns.

### 4.10 ★ Ensemble of Geological Scenarios

Train with diverse Flumy parameter ranges to cover:
| Scenario | ng | isbx | hmax | Description |
|----------|-----|------|------|-------------|
| Sand-rich channels | 60-80 | 80-120 | 2-4 | Amalgamated multi-story channels |
| Mud-rich, isolated | 20-40 | 30-60 | 1-3 | Single-story, well-separated channels |
| Intermediate | 40-60 | 50-100 | 2-5 | Moderate connectivity |
| Meander belt | 50-70 | 100-150 | 3-6 | Wide, high-sinuosity channels |

This ensures the model generalizes to different fluvial styles, not just one.

---

## 5. Training & Data Considerations

### 5.1 Optimal Hyperparameters

Based on literature and current config analysis:

| Parameter | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| `attn_res` | [16] ❌ | [4] or [2, 4] | Fix: Enable attention at 16×16 or 32×32 |
| `channel_mults` | [1, 2, 4] | [1, 2, 4, 8] | Deeper network for better feature extraction |
| `epochs` | 1000 | 2000-5000 | Diffusion models need more epochs with small datasets |
| `batch_size` | 64 | 32-128 | Depends on GPU memory; 64 is fine for 64×64 |
| `lr` | 5e-5 | 1e-4 with warmup | Slightly higher with linear warmup (1000 steps) |
| `loss` | mse | mse or hybrid | MSE is standard; could add SSIM |
| `dropout` | 0.2 | 0.1 | 0.2 is aggressive; 0.1 for small datasets |
| `timesteps` | 1500 | 1000 | 1000 is sufficient for 64×64 images |
| `predict_type` | epsilon | epsilon | Standard; v-prediction could be tried |
| `beta_schedule` | cosine | cosine | Correct for this image size ✅ |

### 5.2 Training Data Quality Checks

Before training, verify data quality:
1. **Facies distribution**: What % sand/bank/mud across all training samples? Very imbalanced ratios will bias generation.
2. **RMS-quality map**: Check that RMS actually looks different for different facies. If RMS is too smoothed, it carries no information.
3. **Slice selection**: Avoid slices from very top/bottom of Flumy block (boundary effects).
4. **No duplicate slices**: Nearby z-indices can produce nearly identical plan views.

### 5.3 Training Stability

Key practices already in place:
- ✅ Cosine beta schedule (smoother than linear)
- ✅ EMA decay 0.9999
- ✅ Gradient clipping (added in recent improvements)
- ✅ ReduceLROnPlateau scheduler

Additional suggestions:
- **Learning rate warmup**: Ramp LR from 0 to target over first 1000 steps.
- **Monitor gradient norms**: Log gradient norm per step to detect instabilities.
- **Mixed precision training (FP16/BF16)**: Significantly speeds up training with minimal quality loss.

### 5.4 Dataset Size vs Model Capacity

Current setup: ~1000 training samples, ~50-70M parameters.

Rules of thumb for diffusion models:
- **Minimum**: 5K-10K samples for reasonable quality at 64×64.
- **Good**: 50K+ samples.
- **Note**: On-the-fly well augmentation effectively multiplies the dataset size since each epoch sees different well configurations. This is valuable.

If limited to <5K samples:
- Use more aggressive augmentation (flips, rotations).
- Reduce model capacity (`inner_channel: 32` instead of 64).
- Consider transfer learning from a model pre-trained on natural images.

---

## 6. Evaluation Strategy

### 6.1 Quantitative Metrics

**Per-realization metrics:**
- **Facies Proportion Error**: `|predicted_ng - true_ng|` where ng = sand fraction.
- **Connectivity**: Does the sand body form connected channels? Use connected component analysis.
- **Well Honor Rate**: Do generated facies match the known well values exactly? Should be 100%.

**Multi-realization metrics:**
- **Diversity**: Entropy of facies predictions across realizations (higher = more diverse).
- **Consistency**: Do all realizations respect the well constraints?
- **Calibration**: Does the probability of sand at each pixel (averaged across realizations) match observed frequency?

**Distribution metrics:**
- **FID (Fréchet Inception Distance)**: Compare distribution of generated vs ground truth facies maps. NOTE: Standard FID uses ImageNet features — may not be meaningful for geological images. Consider training a domain-specific feature extractor.
- **Variogram analysis**: Compare spatial correlation structure (variogram) of generated vs real facies. This is the geologically meaningful metric.
- **Multi-point statistics**: Compare channel body statistics (width, sinuosity, aspect ratio).

### 6.2 Visual Evaluation Checklist

For each test sample, visually check:
- [ ] Does the channel pattern align with the RMS intensity pattern?
- [ ] Are channel bodies continuous and geologically plausible?
- [ ] Are well observations honored (correct facies at well locations)?
- [ ] Do different realizations show meaningful variability?
- [ ] Are channel boundaries sharp (not blurry transitions)?
- [ ] Is the net-to-gross ratio approximately correct?
- [ ] Do channels have appropriate sinuosity and width?
- [ ] Are there any artifacts (checkerboard, edge effects, repetitive patterns)?

---

## 7. Deployment Considerations

### 7.1 From Training to Real Data Pipeline

```
Training:
  Flumy facies → forward modeling → synthetic RMS
  Train: (synthetic RMS, sparse wells) → facies

Deployment:
  Real seismic → extract RMS attribute
  Real wells → get well facies logs
  Infer: (real RMS, real wells) → predicted facies maps
```

### 7.2 Domain Gap Risks

| Aspect | Synthetic | Real | Risk Level |
|--------|-----------|------|------------|
| RMS texture | Smooth, clean | Noisy, artifacts | HIGH |
| RMS dynamic range | Normalized per sample | Varies by survey | MEDIUM |
| Channel patterns | Flumy-specific sinuosity | Varied depositional styles | MEDIUM |
| Scale | Fixed 64×64 pixels | Varies with survey cell size | LOW |
| Well density | 3-15 random wells | May be clustered, sparse, or dense | LOW |
| Facies boundary | Sharp (grid-based) | Gradational | MEDIUM |

### 7.3 Recommended Deployment Workflow

1. **Normalize real RMS** to [-1, 1] using the same scheme as training (min-max per sample).
2. **Validate well coordinates** map to correct image pixels.
3. **Generate 50-100 realizations** using DDIM (fast, 50 steps, ~1 min on GPU).
4. **Compute probability maps** → sand/bank/mud probability at each pixel.
5. **Compute uncertainty** → entropy map shows where model is uncertain.
6. **Visual QC** → check realizations look geologically reasonable.
7. **If quality is poor** → fine-tune model with any available labeled real data, or adjust RMS preprocessing.

### 7.4 Image Size Considerations

The model trains at 64×64. Real seismic surveys may cover much larger areas. Options:
- **Downsample** the real data to 64×64 → lose detail but match training resolution.
- **Sliding window** approach: Process overlapping 64×64 patches, stitch results → boundary artifacts.
- **Train at larger size** (128×128 or 256×256) → needs more training data and compute.
- **Super-resolution** post-processing: Generate at 64×64, then upscale with a separate model.

---

## Summary: Priority Actions

| Priority | Action | Impact | Effort |
|----------|--------|--------|--------|
| 1 | Fix `attn_res` from [16] to [4] | HIGH — enables attention | Trivial |
| 2 | Increase training data (500+ simulations) | HIGH — better generalization | Medium |
| 3 | Add data augmentation (flips, rotations) | MEDIUM — effectively 4-8x data | Low |
| 4 | Implement classifier-free guidance | HIGH — better conditioning | Medium |
| 5 | Add RMS augmentation for domain robustness | HIGH — better real-data performance | Low |
| 6 | Test RePaint-style inpainting | MEDIUM — better boundary quality | Medium |
| 7 | Cross-attention for RMS conditioning | HIGH — better architecture | High |
| 8 | Train at 128×128 | MEDIUM — better resolution | Medium |
