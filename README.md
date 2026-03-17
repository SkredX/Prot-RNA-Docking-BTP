# Prot-RNA-Docking-BTP

> **Computational prediction of 3-D Protein–RNA complex structures from unbound (UU) states.**  
> Developed as a B.Tech Project (BTP) at the **Indian Institute of Technology (IIT) Kharagpur**.

This repository implements a full end-to-end computational docking pipeline — from raw PDB data management and interactive structural visualisation, through to a GPU-accelerated rigid-body docking engine — evaluated on the [PRDBv3.0](http://www.zlab.umassmed.edu/benchmark/) benchmark dataset.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Repository Layout](#2-repository-layout)
3. [Module 1 — Master Index (Kaggle)](#3-module-1--master-index-kaggle)
4. [Module 2 — Interactive 3-D Plotter (Kaggle)](#4-module-2--interactive-3-d-plotter-kaggle)
5. [Module 3 — FFT Docking Pipeline (Local)](#5-module-3--fft-docking-pipeline-local)
   - [Installation](#51-installation)
   - [Quick-start](#52-quick-start)
   - [All commands & flags](#53-all-commands--flags)
   - [Visualization flags](#54-visualization-flags)
   - [Expected output](#55-expected-output)
6. [Biological & Algorithmic Rationale](#6-biological--algorithmic-rationale)
7. [Mathematical Reference](#7-mathematical-reference)
   - [Katchalski-Katzir voxel encoding](#71-katchalski-katzir-voxel-encoding)
   - [FFT cross-correlation scoring](#72-fft-cross-correlation-scoring)
   - [Uniform SO(3) sampling](#73-uniform-so3-sampling-via-hopf-fibration)
   - [Kabsch superposition](#74-kabsch-superposition-algorithm)
   - [L-RMSD](#75-ligand-rmsd-l-rmsd)
   - [I-RMSD](#76-interface-rmsd-i-rmsd)
   - [Buried Surface Area](#77-buried-surface-area-bsa)
8. [References](#8-references)

---

## 1. What This Project Does

Protein–RNA interactions govern fundamental biological processes — splicing, translation, gene regulation, and viral replication. Determining the 3-D structure of a protein–RNA complex experimentally is expensive and slow. Computational docking predicts this structure *in silico* from the two molecules' individual unbound structures.

This project implements a **rigid-body shape complementarity search** using the following strategy:

1. Both molecules are converted into discrete 3-D voxel grids encoding their surface (+1) and impenetrable interior (−15).
2. The RNA is rotated through a mathematically uniform set of orientations sampled from SO(3).
3. For each rotation, the **Convolution Theorem** is used to evaluate the complementarity score for *every possible translation simultaneously* in O(N³ log N) time via a 3-D FFT.
4. The top-scoring (rotation, translation) pairs are written out as docked PDB files and benchmarked against the known crystal structure using CAPRI-standard metrics (L-RMSD, I-RMSD).

---

## 2. Repository Layout

```
Prot-RNA-Docking-BTP/
│
├── Master Index - Kaggle/          ← Module 1: dataset indexing for Kaggle
│   ├── generate_index.py
│   └── master_index.csv
│
├── Interactive 3D Plotter - Kaggle/ ← Module 2: browser-based structural viewer
│   └── plotter.py / notebook
│
├── Pipeline/                        ← Module 3: local FFT docking engine
│   ├── run.py                       ← single entry point
│   ├── phase1.py                    ← PDB parser
│   ├── phase2.py                    ← voxel grid builder
│   ├── phase3.py                    ← SO(3) rotation sampler
│   ├── phase4.py                    ← FFT docking engine (GPU)
│   ├── phase5.py                    ← pose export & RMSD benchmarking
│   └── README.md                    ← detailed Pipeline-specific docs
│
└── Research Papers/                 ← curated literature
```

> **New to this project?** The recommended reading order is: this file → `Pipeline/README.md` for deep implementation and mathematical details.

---

## 3. Module 1 — Master Index (Kaggle)

**Purpose:** Efficient file-path management for the PRDBv3.0 dataset hosted on Kaggle.

PRDBv3.0 contains hundreds of complexes, each in its own sub-folder with up to three PDB files (bound complex, unbound protein, unbound RNA). Searching nested directories at runtime for every case would be slow and fragile. The Master Index solves this with a pre-built CSV lookup table.

**Contents:**

| File | Role |
|------|------|
| `generate_index.py` | Walks the PRDBv3 directory tree, extracts PDB IDs and file paths, writes `master_index.csv` |
| `master_index.csv` | Flat table: `complex_id`, `complex_path`, `protein_path`, `rna_path`, `docking_case` |

**Logic:** The CSV enables O(1) path lookup by complex ID at runtime — load once with `pandas.read_csv()`, index on `complex_id`, and retrieve any file path in a single dictionary lookup. This is especially important on Kaggle where directory traversal is slow and the working directory is read-only.

**Usage on Kaggle:**

```python
import pandas as pd

index = pd.read_csv('/kaggle/input/prot-rna-index/master_index.csv', index_col='complex_id')

# Retrieve paths for a specific complex
row = index.loc['1ASY']
print(row['complex_path'])   # path to 1ASY.pdb
print(row['protein_path'])   # path to unbound protein PDB
print(row['rna_path'])       # path to unbound RNA PDB

# Filter to UU cases only
uu_cases = index[index['docking_case'] == 'UU']
print(f'{len(uu_cases)} UU cases available')
```

---

## 4. Module 2 — Interactive 3-D Plotter (Kaggle)

**Purpose:** Browser-based structural visualisation and sanity-checking before running docking.

Running a docking algorithm on malformed or mis-classified PDB files wastes hours of compute time. This module provides an interactive 3-D viewer to visually inspect any complex in the dataset directly in a Kaggle notebook or browser.

**What it renders:**

| View | Atoms shown | Colour |
|------|-------------|--------|
| Bound complex | All heavy atoms | Protein = blue, RNA = orange |
| Unbound protein | All heavy atoms | Blue |
| Unbound RNA | All heavy atoms | Orange |
| Voxel grid (Phase 2 preview) | Surface / interior voxels | Surface = red, Interior = blue |

**What to look for:**

- Are protein and RNA chains correctly separated (not all classified as "unknown")?
- Does the bound complex show the two molecules in physical contact?
- Are the unbound structures in reasonable conformations without gaps in the backbone?
- Does the voxel grid preview show a continuous, well-defined molecular surface with no isolated floating voxels?

**Usage on Kaggle:**

```python
from plotter import visualize_structure, visualize_grid
from phase1 import parse_pdb
from phase2 import GridBuilder

# Visualise bound complex
struct = parse_pdb(index.loc['1ASY']['complex_path'], pdb_id='1ASY')
visualize_structure(struct, title='1ASY — Bound Complex')

# Visualise voxel grid for the unbound protein
protein_struct = parse_pdb(index.loc['1ASY']['protein_path'], pdb_id='1ASY_pro')
builder = GridBuilder(resolution=1.0, padding=8.0)
grid = builder.build(protein_struct, mol_type='protein')
visualize_grid(grid)
```

All plots are rendered via **Plotly** and open in the browser or inline in a Jupyter/Kaggle notebook. Points are automatically downsampled to 150,000 if the grid is very large, preventing browser crashes.

---

## 5. Module 3 — FFT Docking Pipeline (Local)

This is the computational core of the project. It runs locally (not on Kaggle) because it requires a GPU and substantial memory for the FFT operations.

The pipeline is controlled entirely through a single entry point, `run.py`. You never call the individual phase files directly.

### 5.1 Installation

```bash
# Core dependencies
pip install numpy scipy torch pydantic plotly

# GPU acceleration — replace cu118 with your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Verify GPU is visible
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **Before running anything:** navigate into the `Pipeline/` directory first.
> Python resolves scripts relative to your *current working directory*, not the script's location.
>
> ```powershell
> cd "D:\BTP Files\Prot-RNA-Docking-BTP\Pipeline"
> python run.py --complex 1ASY
> ```

### 5.2 Quick-start

```bash
# Dock one complex
python run.py --complex 1ASY

# Dock one complex and open all visualizations (5 browser tabs)
python run.py --complex 1ASY --viz

# Dock multiple complexes
python run.py --complex 1ASY 1AV6 1B23

# Dock every UU case in the dataset
python run.py --all

# Higher accuracy (finer rotation sampling, save top 10 poses)
python run.py --complex 1ASY --step 15.0 --top_n 10
```

### 5.3 All commands & flags

| Flag | Default | Description |
|------|---------|-------------|
| `--complex ID…` | — | One or more complex IDs. Mutually exclusive with `--all`. |
| `--all` | — | Process every UU case in `PRDBv3_info.json`. Mutually exclusive with `--complex`. |
| `--viz [TYPE…]` | off | Open visualizations. No argument = all three types. See §5.4. |
| `--step` | `30.0` | Angular step in degrees for SO(3) sampling. Halving it increases rotation count ~8×. |
| `--resolution` | `1.0` | Voxel edge length in Å. Smaller = finer grid = slower. |
| `--top_n` | `5` | Number of top-ranked poses to export per complex. |
| `--json` | `D:\BTP…json` | Path to `PRDBv3_info.json`. |
| `--pdb_root` | `D:\BTP…` | Root folder containing one sub-folder per complex ID. |
| `--output` | `generated_PDBS` | Output directory for ranked PDB files. |
| `--results` | `results.pkl` | Pickle file for raw `DockingResult` objects. |

**Accuracy vs. speed trade-off for `--step`:**

| `--step` | Rotations | Typical run time (GPU) |
|----------|-----------|------------------------|
| 30° | ~512 | ~2–5 min per complex |
| 15° | ~4,096 | ~15–40 min per complex |
| 10° | ~13,824 | ~1–2 hr per complex |
| 6° | ~64,000 | ~6–12 hr per complex |

### 5.4 Visualization flags

`--viz` accepts zero or more space-separated type names. **`--viz` alone enables all three.**

| Command | What opens |
|---------|-----------|
| `python run.py --complex 1ASY --viz` | All 5 tabs: atom scatter + protein grid + RNA grid + rotation axes + angle histogram |
| `python run.py --complex 1ASY --viz structure` | 1 tab — 3-D atom scatter of the bound complex |
| `python run.py --complex 1ASY --viz grid` | 2 tabs — protein and RNA voxel grids (surface = red, interior = blue) |
| `python run.py --complex 1ASY --viz rotations` | 2 tabs — SO(3) axis distribution + angle histogram |
| `python run.py --complex 1ASY --viz structure grid` | 3 tabs |
| `python run.py --complex 1ASY --viz grid rotations` | 4 tabs |
| `python run.py --all --viz rotations` | Rotation plots open **once only**; all complexes then dock silently |

> `rotations` opens exactly once regardless of how many complexes you process — the SO(3) set is identical for all complexes at the same `--step`.  
> `structure` and `grid` open per complex. Avoid with `--all` on large datasets.

### 5.5 Expected output

**`results.pkl`** — reload and inspect docking results:

```python
import pickle
with open('results.pkl', 'rb') as f:
    results = pickle.load(f)

top = results['1ASY'][0]
print(top.score)               # complementarity score (higher = better)
print(top.rotation_matrix)     # np.ndarray (3×3)
print(top.translation_vector)  # np.ndarray (3,)  in Ångström
```

**`generated_PDBS/`** — ranked PDB files, one sub-folder per complex per rank:

```
generated_PDBS/
    1ASY/
        rank1/  protein.pdb  rna.pdb
        rank2/  protein.pdb  rna.pdb
        ...
```

The protein is unchanged. The RNA has the docking transform applied:

```
coord_docked = R @ (coord_unbound − center) + center + t
```

The console prints a benchmark table for each complex on completion:

```
  Rank   Score      L-RMSD (Å)   I-RMSD (Å)   Output
  1      1847.00       12.34         8.91       generated_PDBS/1ASY/rank1
  2      1731.50       18.02        11.43       generated_PDBS/1ASY/rank2
```

---

## 6. Biological & Algorithmic Rationale

### Why shape complementarity?

Macromolecules associate primarily through non-covalent interactions (hydrogen bonds, electrostatic forces, van der Waals contacts) across their solvent-accessible surfaces. Simultaneously, no two atoms can occupy the same space — steric clashing is physically impossible (Pauli exclusion). A successful docking prediction must therefore satisfy two constraints:

1. **Maximise surface contact** — bring the two accessible surfaces into proximity.
2. **Minimise steric clash** — ensure the solid interiors do not overlap.

These constraints map directly onto the Katchalski-Katzir voxel encoding (§7.1): surface voxels score +1, interior voxels score −15. The cross-correlation of the two grids over all possible translations produces a score that rewards surface–surface contact and heavily penalises any interior overlap.

### Why FFT?

For a voxel grid of size N × N × N, evaluating the complementarity score at every possible translation directly requires O(N³) operations *per translation*, and there are O(N³) possible translations — giving O(N⁶) total. For N = 128, that is 4.4 × 10¹² operations *per rotation*. The Convolution Theorem reduces this to O(N³ log N) ≈ 2.8 × 10⁸ — a ~16,000× speedup for a single rotation, and a ~10⁶× speedup over the full naive search.

### Why quaternions for rotation?

Euler angle parameterisations of SO(3) suffer from gimbal lock — at certain configurations, two of the three angle axes become aligned and a degree of freedom is lost. Uniform sampling in Euler angle space also over-samples orientations near the coordinate poles, introducing systematic bias. Quaternions on S³ avoid both problems and allow the Hopf fibration to generate a provably uniform distribution over all 3-D orientations (§7.3).

---

## 7. Mathematical Reference

### 7.1 Katchalski-Katzir Voxel Encoding

The continuous molecular structure is discretised into a 3-D grid G of dimensions Nₓ × Nᵧ × N_z. Each voxel at position **r** is assigned a value from {−15, 0, +1}:

```
G(r) = −15   if  min_a ‖r − r_a‖ < R_a − δ       (interior)
G(r) = +1    if  min_a ‖r − r_a‖ < R_a            (surface)
G(r) =  0    otherwise                             (solvent)
```

- `r_a` — atomic centre of atom `a`
- `R_a` — Van der Waals radius of atom `a` (Bondi 1964 + AMBER99)
- `δ = 1.4 Å` — water probe radius (Lee & Richards 1971), defines surface layer thickness

The −15 interior penalty ensures a **single steric clash completely dominates any number of surface contacts**, enforcing a hard steric exclusion constraint. Grid dimensions are always powers of two for FFT efficiency.

The shape complementarity score for translational offset **t** between protein grid P and RNA grid R is:

```
S(t) = Σ_r  P(r) · R(r − t)
```

| Voxel pair | Contribution | Meaning |
|---|---|---|
| surface × surface | +1 × +1 = **+1** | Rewarded surface contact |
| surface × interior | +1 × −15 = **−15** | Penalised clash |
| interior × interior | −15 × −15 = **+225** | Artefact suppressed by grid design |
| solvent × anything | 0 | No contribution |

---

### 7.2 FFT Cross-Correlation Scoring

Direct evaluation of S(**t**) for all N³ translation vectors requires **O(N⁶)** operations. The Convolution Theorem reduces this to **O(N³ log N)**:

```
S(t)  =  Σ_r P(r) · R(r − t)  =  [P ★ R](t)       (spatial cross-correlation)

S     =  F⁻¹( F(P) · conj(F(R)) )                   (frequency domain equivalent)
```

where `F` is the 3-D discrete Fourier transform (DFT) and `conj` is the complex conjugate. This follows directly from the **cross-correlation theorem**: the Fourier transform of the cross-correlation of two functions equals the pointwise product of the first function's transform with the complex conjugate of the second.

**Implementation in `phase4.py`:**

```python
# Computed once before the rotation loop — protein never moves
fft_pro  = torch.fft.fftn(pro_tensor)

# Per rotation:
fft_rna  = torch.fft.fftn(rna_tensor_rotated)
corr     = torch.fft.ifftn(fft_pro * torch.conj(fft_rna)).real

best_score = corr.max()
best_voxel = corr.argmax()    # unravelled to (ix, iy, iz) → translation in Å
```

**Complexity comparison** (N = 128, K = 512 rotations):

| Method | Operations | Time (approx.) |
|--------|-----------|----------------|
| Naive direct search | K × N⁶ ≈ 2.3 × 10¹⁵ | Years |
| FFT (this pipeline) | K × N³ log N ≈ 1.4 × 10¹¹ | Minutes |

---

### 7.3 Uniform SO(3) Sampling via Hopf Fibration

Every rotation R ∈ SO(3) corresponds to a pair of antipodal unit quaternions ±**q** = ±(w, x, y, z) on the 3-sphere S³ ⊂ ℝ⁴. The relationship between a quaternion and an axis-angle rotation by angle α about unit axis **n̂** is:

```
q  =  ( cos(α/2),  n̂ · sin(α/2) )
```

Naive uniform sampling in Euler angle space concentrates samples near the poles of the sphere, introducing systematic orientation bias. The **Hopf fibration** solves this by decomposing S³ into a product of S¹ fibres over S², enabling exact uniform sampling via three angles (θ, φ, ψ):

```
w = cos(θ) · cos(ψ/2)          θ ∈ [0, π/2]
x = cos(θ) · sin(ψ/2)          φ ∈ [0, 2π)
y = sin(θ) · cos(φ + ψ/2)      ψ ∈ [0, 2π)
z = sin(θ) · sin(φ + ψ/2)
```

Midpoint sampling (`θᵢ = (π/2)(i + 0.5) / N_θ`) with equal spacing in φ and ψ gives near-uniform coverage. Rotation count scales as **O(step⁻³)**:

| `--step` | Rotations sampled |
|----------|------------------|
| 30° | ~512 |
| 15° | ~4,096 |
| 10° | ~13,824 |
| 6° | ~64,000 |

Each quaternion is converted to a 3×3 rotation matrix (Shoemake 1985):

```
R = [ [1−2(y²+z²),   2(xy−zw),    2(xz+yw)  ],
      [  2(xy+zw),  1−2(x²+z²),   2(yz−xw)  ],
      [  2(xz−yw),   2(yz+xw),   1−2(x²+y²) ] ]
```

The RNA grid is then rotated around its geometric centre using `scipy.ndimage.affine_transform` with trilinear interpolation, and voxel values are re-snapped to {−15, 0, +1} after interpolation.

---

### 7.4 Kabsch Superposition Algorithm

Finds the optimal rigid-body superposition (rotation R + translation **t**) that minimises RMSD between two corresponding point sets P (mobile) and Q (reference). Used in `phase5.py` to align the predicted protein onto the reference before computing L-RMSD.

```
1.  Centre both sets:        P_c = P − mean(P),    Q_c = Q − mean(Q)
2.  Covariance matrix:       H   = P_cᵀ Q_c
3.  SVD:                     H   = U S Vᵀ
4.  Reflection correction:   d   = sign(det(VᵀUᵀ)),    D = diag(1, 1, d)
5.  Optimal rotation:        R   = Vᵀ D Uᵀ
6.  Optimal translation:     t   = mean(Q) − R · mean(P)
7.  Apply:                   P_aligned = (R · Pᵀ)ᵀ + t
```

Step 4 is critical: the determinant correction ensures R is a **proper rotation** (det R = +1). Without it, the algorithm may return an improper rotation (reflection + rotation, det = −1) for mirror-symmetric point sets.

---

### 7.5 Ligand RMSD (L-RMSD)

Standard CAPRI benchmark metric. Measures global RNA positioning accuracy after protein superposition.

**Procedure:**
1. Extract Cα atoms from predicted unbound protein and reference bound protein; match by `(chain_id, res_seq)`.
2. Compute Kabsch superposition: `R_sup, t_sup = kabsch(pred_Cα, ref_Cα)`.
3. Apply to docked RNA C4' coordinates: `pred_C4'_sup = R_sup · pred_C4' + t_sup`.
4. Compute RMSD against reference C4' atoms:

```
L-RMSD = sqrt( (1/N) · Σᵢ ‖pred_C4'_sup(i) − ref_C4'(i)‖² )
```

N = number of matched C4' atoms. C4' is the ribose 4-prime carbon, the standard RNA backbone representative atom in CAPRI assessments.

**Quality thresholds (CAPRI):**

| Category | L-RMSD |
|----------|--------|
| High | < 1 Å |
| Medium | < 5 Å |
| Acceptable | < 10 Å |
| Incorrect | ≥ 10 Å |

---

### 7.6 Interface RMSD (I-RMSD)

More sensitive to binding-site geometry than L-RMSD. The superposition is performed on the **interface atoms themselves** rather than the whole protein.

**Interface definition:** residues with any heavy atom within **10 Å** of any heavy atom of the partner molecule in the reference crystal structure.

**Procedure:**
1. Identify interface protein residues and interface RNA residues using the 10 Å cutoff on the reference complex.
2. Extract Cα (protein interface) and C4' (RNA interface) atoms from both predicted and reference structures.
3. Stack into combined arrays: `pred_int = [pred_Cα_interface | pred_C4'_interface]`.
4. Kabsch superposition of `pred_int` onto `ref_int`.
5. Compute RMSD of the superimposed interface atoms:

```
I-RMSD = sqrt( (1/M) · Σⱼ ‖pred_int_sup(j) − ref_int(j)‖² )
```

M = total matched interface backbone atoms (Cα + C4').

**Quality thresholds (CAPRI):**

| Category | I-RMSD |
|----------|--------|
| High | < 1 Å |
| Medium | < 2 Å |
| Acceptable | < 4 Å |
| Incorrect | ≥ 4 Å |

---

### 7.7 Buried Surface Area (BSA)

BSA quantifies the solvent-accessible surface area (SASA) shielded from solvent when the two molecules form a complex. It is a proxy for binding interface size and correlates strongly with binding affinity across protein–RNA complexes.

```
BSA = SASA(protein_free) + SASA(RNA_free) − SASA(complex)
```

SASA is computed by rolling a probe sphere (r = 1.4 Å, the water molecule radius) over the Van der Waals surface of the molecule. Typical protein–RNA interfaces bury **1,000–3,000 Å²**.

BSA is not computed during the FFT docking search — doing so would require a full solvent-accessibility calculation for every one of the millions of candidate poses, which is prohibitively expensive. The voxel scoring function is its discrete analogue: maximising the count of surface-contact (+1 × +1) voxel pairs is equivalent to maximising the buried surface area between the two surface layers.

To compute precise BSA post-hoc on the final top poses using [FreeSASA](https://freesasa.github.io/):

```python
import freesasa

sasa_complex = freesasa.calc(freesasa.Structure('generated_PDBS/1ASY/rank1/complex.pdb')).totalArea()
sasa_protein = freesasa.calc(freesasa.Structure('generated_PDBS/1ASY/rank1/protein.pdb')).totalArea()
sasa_rna     = freesasa.calc(freesasa.Structure('generated_PDBS/1ASY/rank1/rna.pdb')).totalArea()

bsa = sasa_protein + sasa_rna - sasa_complex
print(f'BSA = {bsa:.1f} Å²')
```

---

## 8. References

- **Katchalski-Katzir et al. (1992)** — Molecular surface recognition: determination of geometric fit between proteins and their ligands by correlation techniques. *PNAS* 89(6), 2195–2199.
- **Shoemake, K. (1985)** — Animating rotation with quaternion curves. *SIGGRAPH '85 Proceedings*.
- **Kabsch, W. (1976)** — A solution for the best rotation to relate two sets of vectors. *Acta Crystallographica* A32, 922–923.
- **Lee & Richards (1971)** — The interpretation of protein structures: estimation of static accessibility. *Journal of Molecular Biology* 55(3), 379–400.
- **Bondi, A. (1964)** — Van der Waals Volumes and Radii. *Journal of Physical Chemistry* 68(3), 441–451.
- **Vajda et al. (2020)** — New additions to the ClusPro server motivated by CAPRI. *PLOS Computational Biology*.
- **PRDBv3** — Protein–RNA Docking Benchmark v3.0.

---

*Developed at IIT Kharagpur as a B.Tech Project. For Pipeline-specific documentation, installation instructions, and extended mathematical derivations, see [`Pipeline/README.md`](Pipeline/README.md).*
