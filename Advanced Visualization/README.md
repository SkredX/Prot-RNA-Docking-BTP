# Advanced Visualization

**Research-grade extensions to the Protein–RNA FFT docking pipeline.**  
This folder is a self-contained sub-pipeline that enriches the standard shape grid with physically motivated grid channels and provides detailed interactive visualisations — all for one complex at a time.

---

## Contents

| File | Purpose |
|------|---------|
| `adv_run.py` | **Master entry point.** Interactive menu launcher for all three modules. |
| `adv_channel_grids.py` | **Module 1** — Multi-channel grids: shape, electrostatics, desolvation. |
| `adv_ion_grids.py` | **Module 2** — Explicit Mg²⁺ ion-density grids and charge screening. |
| `adv_soft_grids.py` | **Module 3** — Gaussian-blurred soft grids for conformational flexibility. |

> **You must also copy `phase1.py` and `phase2.py` from the main pipeline into this folder before running anything.**

---

## Quick-Start

```powershell
# 1. Navigate into this folder
cd "D:\BTP Files\pipeline\Advanced_Visualization"

# 2. Ensure dependencies are installed (same as the main pipeline)
pip install numpy scipy plotly pydantic

# 3. Run the master launcher
python adv_run.py
```

You will be prompted for:
- **Complex ID** (e.g. `1ASY`)
- **JSON path** (default `D:\BTP Files\PRDBv3.0\PRDBv3_info.json`)
- **PDB root** (default `D:\BTP Files\PRDBv3.0`)

Then choose which module(s) to run from the menu.

---

## Module Reference

### Module 1 — Multi-Channel Grids (`adv_channel_grids.py`)

Extends the single-channel Katchalski-Katzir grid to three channels:

| Channel | Values | Description |
|---------|--------|-------------|
| 0 — Shape | `{−15, 0, +1}` | Standard KK interior / surface / exterior |
| 1 — Electrostatics | `float` kT/e | Debye-Hückel approximation to the Poisson-Boltzmann potential |
| 2 — Desolvation | `{0, +1}` | Hydration shell layer (penalty zone ~3 Å beyond the vdW surface) |

**Electrostatic model.** For each atom *a* with partial charge *q_a*:

```
φ_a(r) = q_a · exp(−|r − r_a| / λ) / (ε · |r − r_a|)
```

where λ = 8 Å (Debye length at ~150 mM salt) and ε = 4 (protein interior dielectric).  
Partial charges are drawn from a simplified AMBER99 lookup; fallback heuristics used for unlisted residues.

**Desolvation model.** Binary-dilation of the occupied (interior + surface) mask by *k = round(3.0 Å / resolution)* voxels, minus the occupied mask itself.  Energetic penalty is incurred when hydrophobic protein patches displace this shell.

**Visualisations opened (browser tabs):**
1. Channel 0 — shape (protein)
2. Channel 0 — shape (RNA)
3. Channel 1 — electrostatics (protein)
4. Channel 1 — electrostatics (RNA)
5. Channel 2 — desolvation shell (protein)
6. Channel 2 — desolvation shell (RNA)
7. All-channel overlay (protein)
8. All-channel overlay (RNA)

Run the module standalone:
```powershell
python adv_channel_grids.py
```

---

### Module 2 — Ion Density Grids (`adv_ion_grids.py`)

Addresses the **Magnesium Problem**: RNA's phosphate backbone carries a ~−1e charge per residue.  Mg²⁺ ions neutralise this charge and are critical mediators of protein–RNA binding.  Ignoring ion positions causes both steric and electrostatic artifacts in rigid-body docking.

**Algorithm (simplified Manning condensation heuristic):**

1. Compute the RNA electrostatic potential grid (via Module 1's engine).
2. Identify accessible voxels with φ < −0.5 kT/e (strongly negative → high Mg²⁺ affinity).
3. Gaussian-smooth the negative-potential mask to form a continuous probability field *p(r) ∈ [0, 1]*.
4. Greedily select discrete Mg²⁺ site predictions (local maxima in *p*, enforcing ≥ 5 Å separation between sites).
5. Add +2e Debye-Hückel contributions from each predicted site to produce the **charge-screened electrostatics grid**.

**Visualisations:**
1. RNA surface coloured by Mg²⁺ probability, with site markers (green diamonds).
2. Raw vs. Mg²⁺-screened electrostatics comparison (2 tabs, shared colour scale).
3. Volumetric ion cloud (all voxels with *p* > 0.4) overlaid on RNA surface.

Run standalone:
```powershell
python adv_ion_grids.py
```

---

### Module 3 — Soft Grids (`adv_soft_grids.py`)

Replaces the hard −15 clash penalty with a **Gaussian-blurred gradient** that penalises shallow steric overlaps (~1–2 Å, typical of side-chain breathing) proportionally rather than with a cliff.

**Mathematical formulation.** Given the hard grid *G_hard* with values in {−15, 0, +1}:

```
G_soft(r; σ) = clip( G_hard ⊛ N(0, σ²),  G_hard.min(),  +1 )
```

where ⊛ denotes 3D Gaussian convolution and surface voxels are optionally reset to exactly +1 post-convolution to preserve surface-contact scores.

Three σ values are generated: **0.5 Å**, **1.0 Å**, **2.0 Å**.

| σ (Å) | Effect |
|--------|--------|
| 0.5 | Minimal softening — only the outermost interior shell is affected |
| 1.0 | Moderate — ~1 Å side-chain overlap penalised at ~−3 instead of −15 |
| 2.0 | Aggressive — broad gradient, tolerates up to ~2 Å clashes |

**Visualisations:**
1. 2D cross-section slice comparison (hard vs. all σ) — protein
2. 2D cross-section slice comparison — RNA
3. Voxel-value histogram (penalty redistribution) — protein
4. Voxel-value histogram — RNA
5. 3D soft grid render (σ=1.0 Å, gradient interior) — protein
6. 3D soft grid render — RNA
7. Penalty vs. depth profile curve — protein
8. Penalty vs. depth profile curve — RNA

Run standalone:
```powershell
python adv_soft_grids.py
```

---

## Testing Locally — Step-by-Step

### 1. Folder structure

After copying the required files, your `Advanced_Visualization/` directory should look like:

```
Advanced_Visualization/
    adv_run.py              ← master launcher
    adv_channel_grids.py    ← Module 1
    adv_ion_grids.py        ← Module 2
    adv_soft_grids.py       ← Module 3
    phase1.py               ← copied from main pipeline
    phase2.py               ← copied from main pipeline
```

### 2. Verify dependencies

```powershell
python -c "import numpy, scipy, plotly, pydantic; print('All OK')"
```

### 3. Smoke-test with a known complex

Use any complex ID you know is present in the dataset (e.g. `1ASY`):

```powershell
cd "D:\BTP Files\pipeline\Advanced_Visualization"
python adv_run.py
```

Enter `1ASY` at the first prompt, accept the default paths, and select `1` (Module 1) at the menu.  
If four browser tabs open (shape protein, shape RNA, elec protein, elec RNA), everything is working.

### 4. Per-module standalone test

Each module can be run independently without `adv_run.py`:

```powershell
python adv_channel_grids.py   # Module 1
python adv_ion_grids.py       # Module 2
python adv_soft_grids.py      # Module 3
```

### 5. Adjusting parameters

All physical parameters (Debye length, hydration radius, σ values, etc.) are constructor arguments on the builder classes:

```python
# Example: finer electrostatics + wider hydration shell
builder = MultiChannelBuilder(
    resolution=1.0,
    padding=8.0,
    debye_length=6.0,      # tighter screening → more localised potential
    hydration_radius=4.5,  # wider solvent shell
)
```

Modify the `main()` function in each module file to change defaults for your experiments.

### 6. Expected memory footprint

At the default 1 Å resolution, a typical ~80-residue protein will produce a grid of ~128³ ≈ 2M voxels per channel.  All three channels together use ~25 MB RAM per molecule.  RNA structures are usually smaller.  No GPU is required.

---

## Limitations & Known Approximations

| Approximation | Impact | Mitigation |
|--------------|--------|-----------|
| Debye-Hückel electrostatics (not full PB solver) | Overestimates potential near the surface at high charge density | Reduce `debye_length` for denser salt; use APBS for production |
| Manning condensation heuristic for Mg²⁺ placement | Misses sequence-specific ion binding sites | Cross-reference with crystal-structure HETATM Mg entries |
| Gaussian softening applied isotropically | Does not distinguish backbone vs. side-chain flexibility | Use per-residue B-factor weighting (future extension) |
| Desolvation penalty is uniform (+1) across the shell | Ignores hydrophobic vs. polar distinction | Weight by atom SASA and hydrophobicity scale (future) |

---

## References

- **Katchalski-Katzir et al. (1992)** — *PNAS* 89(6), 2195–2199.
- **Debye & Hückel (1923)** — electrostatic screening in ionic solutions.
- **Manning (1978)** — counterion condensation theory for polyelectrolytes.
- **Fernández-Recio et al. (2004)** — Soft docking via Gaussian smoothing. *J. Mol. Biol.* 335(3), 843–865.
- **Cornell et al. (1995)** — AMBER99 partial charges. *JACS* 117(19), 5179–5197.
- **Bondi (1964)** — Van der Waals radii. *J. Phys. Chem.* 68(3), 441–451.
