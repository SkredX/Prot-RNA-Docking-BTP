# Advanced Visualization

**Research-grade extensions to the Protein–RNA FFT docking pipeline.**  
Five self-contained sub-modules enriching the standard shape grid with physically
motivated multi-channel representations, statistical ion placement, conformational
flexibility, spherical harmonic decomposition, and deep-learning re-ranking — all
for one complex at a time.

---

## Contents

| File | Module | Purpose |
|------|--------|---------|
| `adv_run.py` | — | **Master entry point.** Interactive menu launcher for all five modules. |
| `adv_channel_grids.py` | 1 | Multi-channel grids: shape, electrostatics, desolvation. |
| `adv_ion_grids.py` | 2 | Explicit Mg²⁺ ion-density grids and charge screening. |
| `adv_soft_grids.py` | 3 | Gaussian-blurred soft grids for conformational flexibility. |
| `adv_spf.py` | 4 | Spherical Polar Fourier transforms and visualisation. |
| `adv_cnn_scoring.py` | 5 | 3D-CNN interface scoring, pose re-ranking, Grad-CAM. |

> **Before running anything**, copy `phase1.py`, `phase2.py`, and `phase3.py`
> from the main pipeline into this folder.

---

## Quick-Start

```powershell
# 1. Navigate into this folder
cd "D:\BTP Files\pipeline\Advanced_Visualization"

# 2. Base dependencies (same as main pipeline)
pip install numpy scipy plotly pydantic

# 3. Module 5 additionally requires PyTorch
pip install torch
# For GPU: pip install torch --index-url https://download.pytorch.org/whl/cu118

# 4. Run the master launcher
python adv_run.py
```

You will be prompted for:
- **Complex ID** (e.g. `1ASY`)
- **JSON path** (default `D:\BTP Files\PRDBv3.0\PRDBv3_info.json`)
- **PDB root** (default `D:\BTP Files\PRDBv3.0`)

Then choose which module(s) to run from the numbered menu.

---

## Module Reference

### Module 1 — Multi-Channel Grids (`adv_channel_grids.py`)

Extends the single-channel Katchalski-Katzir grid to three channels:

| Channel | Values | Description |
|---------|--------|-------------|
| 0 — Shape | `{−15, 0, +1}` | Standard KK interior / surface / exterior |
| 1 — Electrostatics | `float` kT/e | Debye-Hückel approximation to the Poisson-Boltzmann potential |
| 2 — Desolvation | `{0, +1}` | Hydration shell layer (penalty zone ~3 Å beyond vdW surface) |

**Visualisations (up to 8 tabs):** per-channel heatmaps for protein + RNA, plus
all-channel overlay showing spatial relationship of shape, potential, and shell simultaneously.

---

### Module 2 — Ion Density Grids (`adv_ion_grids.py`)

Addresses the **Magnesium Problem**: RNA's phosphate backbone carries ~−1e per residue.
Mg²⁺ ions neutralise this charge and mediate protein–RNA binding.

**Algorithm (Manning condensation heuristic):**
1. Compute RNA electrostatic potential (Module 1 engine).
2. Identify accessible surface voxels with φ < −0.5 kT/e.
3. Gaussian-smooth into a continuous probability field p(r) ∈ [0, 1].
4. Greedily find discrete Mg²⁺ site predictions (≥ 5 Å separation enforced).
5. Add +2e Debye-Hückel terms from each site → charge-screened electrostatics.

**Visualisations (3 tabs):** probability-coloured surface + site markers, raw vs.
screened comparison, volumetric ion-cloud (p > 0.4).

---

### Module 3 — Soft Grids (`adv_soft_grids.py`)

Replaces the hard −15 clash penalty with Gaussian-blurred gradients at σ = 0.5, 1.0, 2.0 Å.

```
G_soft(r; σ) = clip( G_hard ⊛ N(0, σ²),  G_hard.min(),  +1 )
```

Surface voxels are re-pinned to +1 post-blur. Effect: a ~1 Å side-chain overlap
that scores −15 in the hard grid scores ~−3 at σ=1.0 Å.

**Visualisations (up to 8 tabs):** 2D slice comparisons, voxel histograms,
3D gradient renders, depth-vs-penalty profile curves — for both protein and RNA.

---

### Module 4 — Spherical Polar Fourier Transforms (`adv_spf.py`)

Implements the SPF framework used by HEX (Ritchie & Kemp 2000) and FRODOCK
(Garzon et al. 2009) to decompose molecular shape onto a spherical harmonic basis.

**Mathematical basis:**

```
f(r, θ, φ) = Σ_{n,l,m}  c_{nlm} · R_n(r) · Y_l^m(θ, φ)
```

where `R_n(r)` are radial basis functions and `Y_l^m` are real spherical harmonics.
The rotational cross-correlation can then be evaluated for ALL rotations simultaneously
via a 1D convolution over SO(3), rather than K separate grid-rotation + FFT cycles.

**Why this matters for your pipeline:**
Your current phase4.py performs K affine_transforms (each O(N³)) before each FFT.
SPF replaces all K transforms with a single expansion + SO(3) convolution, giving
~10–100× speedup at equal angular resolution.

**Visualisations (up to 10 tabs):**
- Power spectrum heatmap P[n,l] = Σ_m |c_{nlm}|² (protein + RNA)
- Radial profiles for l = 0, 1, 2, 3 components
- Angular reconstruction quality at l_max = 2, 5, full (Mercator-projection heatmaps)
- SPF overlap kernel K[n,l] = Σ_m c^P*_{nlm} · c^R_{nlm} (complementarity fingerprint)
- Comparative protein vs. RNA power spectra with Pearson correlation
- 3D sphere rendering of outermost shell at chosen l_max

**Runtime note:** SPF expansion at l_max = 8, n_shells = 10, n_θ = 24, n_φ = 48
takes ~15–30 s on a typical laptop CPU. Reduce l_max for faster results.

**Key parameters (constructor):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `l_max` | 8 | Maximum angular order (higher = more shape detail, slower) |
| `n_shells` | 10 | Radial shell count |
| `n_theta` | 24 | Polar quadrature points (Gauss-Legendre) |
| `n_phi` | 48 | Azimuthal integration points (uniform) |

---

### Module 5 — 3D-CNN AI Scoring Function (`adv_cnn_scoring.py`)

A 3-channel 3D Convolutional Neural Network (PyTorch) that rescores the top-N
docking poses generated by the FFT engine (phase4.py).

**CNN input:** 24 × 24 × 24 Å interface box, 3 channels:
- **Channel 0 — Shape:** KK encoding of the combined protein + docked-RNA complex
- **Channel 1 — Electrostatics:** Debye-Hückel potential (protein + docked RNA)
- **Channel 2 — Hydrophobicity:** Kyte-Doolittle scale mapped onto protein surface voxels

**Architecture:**

```
Input (batch, 3, 24, 24, 24)
  ↓  Conv3d(3→32, k=3, BN, ReLU) + MaxPool  →  32 × 12³
  ↓  Conv3d(32→64, k=3, BN, ReLU) + MaxPool  →  64 × 6³
  ↓  Conv3d(64→128, k=3, BN, ReLU) + AdaptPool →  128 × 3³
  ↓  Flatten → 3,456
  ↓  FC(3456 → 512) + ReLU + Dropout(0.4)
  ↓  FC(512 → 1) + Sigmoid
Output: score ∈ [0, 1]  (1 = predicted binder)
Total parameters: ~3.5 M
```

**Three operating modes:**

| Mode | Description | Use case |
|------|-------------|----------|
| **A — Demo** | Random weights. Full pipeline preview, scores are noise-level. | First-time setup verification |
| **B — Self-supervised** | Trains on top-K (positive) vs. bottom-K (negative) FFT poses from this complex. ~25 epochs, ~3–10 min on CPU. Saves checkpoint `cnn_scorer_{ID}.pt`. | Meaningful re-ranking for this complex |
| **C — Load checkpoint** | Loads a previously saved `.pt` file. Instant inference. | Reusing trained weights across sessions |

**Visualisations (up to 5 tabs):**
1. Architecture flowchart (annotated layer boxes)
2. Input tensor channel slices — XY, XZ, YZ cross-sections of the interface box
3. Training loss curves (Mode B only)
4. Pose re-ranking scatter — FFT rank vs. CNN score, coloured by FFT score
5. Grad-CAM 3D saliency — which voxels in the binding pocket drove the CNN's decision

**Grad-CAM implementation:** Gradients are back-propagated through the last Conv3D
layer, globally average-pooled to per-channel weights, multiplied by the feature
maps, ReLU-gated, and upsampled to the input box resolution.

---

## Folder Structure (complete)

```
Advanced_Visualization/
    adv_run.py              ← master launcher
    adv_channel_grids.py    ← Module 1
    adv_ion_grids.py        ← Module 2
    adv_soft_grids.py       ← Module 3
    adv_spf.py              ← Module 4
    adv_cnn_scoring.py      ← Module 5
    phase1.py               ← COPY from main pipeline
    phase2.py               ← COPY from main pipeline
    phase3.py               ← COPY from main pipeline (Module 5 needs it)
    cnn_scorer_1ASY.pt      ← generated by Module 5 Mode B (optional)
```

---

## Testing Locally — Step-by-Step

### 1. Verify all required files are present

```powershell
cd "D:\BTP Files\pipeline\Advanced_Visualization"
dir *.py
```

Expected: `phase1.py`, `phase2.py`, `phase3.py`, `adv_run.py`,
`adv_channel_grids.py`, `adv_ion_grids.py`, `adv_soft_grids.py`,
`adv_spf.py`, `adv_cnn_scoring.py`.

### 2. Install dependencies

```powershell
pip install numpy scipy plotly pydantic
pip install torch          # for Module 5
```

### 3. Smoke-test Module 1 (fastest, ~30 s)

```powershell
python adv_run.py
# Enter: 1ASY  (or any valid complex ID)
# Accept default paths
# Select: 1
# In sub-menu: select 1 2  (shape grids for protein + RNA)
```

Two browser tabs should open with 3D voxel renders.

### 4. Test Module 4 — SPF

```powershell
python adv_spf.py
# Enter: 1ASY
# l_max: 6  (faster for testing)
# Select: 1 2 7 8  (power spectra + comparative)
```

Expansion takes ~10–20 s at l_max=6. Four tabs open.

### 5. Test Module 5 — CNN (Mode A demo, no training)

```powershell
python adv_cnn_scoring.py
# Complex: 1ASY
# Mode: A
# pkl path: (leave blank — uses synthetic poses)
# Top N: 20
# Select: 1 2 4 5  (architecture + slices + reranking + grad-cam)
```

Tensor extraction for 20 poses takes ~2–5 min on CPU.  
For faster testing, reduce top N to 5.

### 6. Full pipeline run

```powershell
python adv_run.py
# Complex: 1ASY
# Select: A   (all five modules)
```

Total runtime: ~20–40 min on CPU (dominated by Module 5 tensor extraction).
On a GPU, Module 5 runs ~5×faster.

---

## Adjusting Parameters

All physical and architectural parameters are constructor arguments:

```python
# Module 1 — wider Debye screening
builder = MultiChannelBuilder(debye_length=12.0, hydration_radius=4.0)

# Module 4 — finer angular resolution
expander = SPFExpander(l_max=12, n_shells=15, n_theta=32, n_phi=64)

# Module 5 — larger interface box
# Edit BOX_SIZE_A = 32 at the top of adv_cnn_scoring.py
# (also update BOX_VOXELS = int(BOX_SIZE_A / VOXEL_RES_A))
```

---

## Limitations and Known Approximations

| Approximation | Impact | Mitigation |
|--------------|--------|------------|
| Debye-Hückel electrostatics, not full PB solver | Overestimates potential near high-charge density | Use APBS for production; reduce `debye_length` for denser salt |
| Manning condensation heuristic for Mg²⁺ | Misses sequence-specific crystal contacts | Cross-reference with HETATM MG entries in the bound complex PDB |
| Gaussian softening applied isotropically | Cannot distinguish side-chain vs. backbone flexibility | Weight per B-factor (future extension) |
| SPF expansion uses real spherical harmonics with GL quadrature | Numerical integration error at low n_theta (< 16) | Use n_theta ≥ 24 for l_max ≥ 8 |
| CNN trained contrastively on top/bottom FFT poses | Positive examples may not be near-native; limited data | Pre-train on HADDOCK/RNACompose benchmark dataset for production use |
| Grad-CAM upsampled with trilinear interpolation | Spatial resolution limited to conv3 feature map size | Use full-resolution GradCAM++ with guided backprop for publication |

---

## References

- **Katchalski-Katzir et al. (1992)** — Molecular surface recognition. *PNAS* 89(6), 2195–2199.
- **Ritchie & Kemp (2000)** — Protein docking using spherical polar Fourier correlations. *Proteins* 39(2), 178–194.
- **Garzon et al. (2009)** — FRODOCK: a new approach for fast rotational protein–protein docking. *Bioinformatics* 25(19), 2544–2551.
- **Debye & Hückel (1923)** — Electrostatic screening in ionic solutions.
- **Manning (1978)** — Counterion condensation theory. *Q. Rev. Biophys.* 11(2), 179–246.
- **Fernández-Recio et al. (2004)** — Soft docking via Gaussian smoothing. *J. Mol. Biol.* 335(3), 843–865.
- **Cornell et al. (1995)** — AMBER99 partial charges. *JACS* 117(19), 5179–5197.
- **Kyte & Doolittle (1982)** — Hydrophobicity scale. *J. Mol. Biol.* 157(1), 105–132.
- **Selvaraju et al. (2017)** — Grad-CAM: Visual explanations from deep networks. *ICCV 2017*.
- **Bondi (1964)** — Van der Waals radii. *J. Phys. Chem.* 68(3), 441–451.
