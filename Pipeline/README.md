# Prot-RNA-Docking-BTP

**Rigid-body FFT-based Protein–RNA docking pipeline**  
Global translational + rotational search using shape complementarity, GPU-accelerated via PyTorch.  
Evaluated on the [PRDBv3](http://www.zlab.umassmed.edu/benchmark/) Unbound–Unbound (UU) benchmark dataset.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Prerequisites & Installation](#3-prerequisites--installation)
4. [Running the Pipeline](#4-running-the-pipeline)
   - [Quick-start examples](#41-quick-start-examples)
   - [--viz flag reference](#42---viz-flag-reference)
   - [Full flag reference](#43-full-flag-reference)
   - [Expected console output](#44-expected-console-output)
5. [Output Structure](#5-output-structure)
6. [Mathematical Reference](#6-mathematical-reference)
   - [Katchalski-Katzir voxel encoding](#61-katchalski-katzir-voxel-encoding)
   - [FFT cross-correlation scoring](#62-fft-cross-correlation-scoring)
   - [Uniform SO(3) sampling](#63-uniform-so3-sampling-via-hopf-fibration)
   - [Kabsch superposition](#64-kabsch-superposition-algorithm)
   - [L-RMSD](#65-ligand-rmsd-l-rmsd)
   - [I-RMSD](#66-interface-rmsd-i-rmsd)
   - [Buried Surface Area](#67-buried-surface-area-bsa)
7. [References](#7-references)

---

## 1. Project Overview

This pipeline predicts the 3-D complex structure of a protein and an RNA molecule starting from their individually crystallised, **unbound (UU)** states. The approach is based on **shape complementarity**: the surface of the RNA should nest into the surface of the protein without their solid interiors clashing.

Because the search space is six-dimensional (3 rotational + 3 translational degrees of freedom), brute-force evaluation is infeasible. Two key algorithmic choices make it tractable:

- **Convolution Theorem** — evaluating the complementarity score for *all possible translations simultaneously* via a single 3-D FFT pair, reducing the translational search from O(N⁶) to O(N³ log N).
- **Uniform SO(3) sampling** — rotating the RNA through a mathematically uniform, bias-free set of orientations using the Hopf fibration instead of Euler angles.

GPU acceleration via PyTorch is used for all FFT operations.

---

## 2. Repository Structure

All pipeline logic lives in six Python files in a single flat directory.

| Phase | File | Responsibility |
|-------|------|----------------|
| 1 | `phase1.py` | Parse PDB files, classify chains as protein/RNA, load UU cases |
| 2 | `phase2.py` | Voxelise atom coordinates into 3-D shape grids |
| 3 | `phase3.py` | Generate uniform SO(3) rotation matrix set |
| 4 | `phase4.py` | GPU-accelerated FFT cross-correlation docking |
| 5 | `phase5.py` | Write docked PDB files, compute L-RMSD / I-RMSD |
| — | `run.py` | Interactive entry point — orchestrates all phases |

```
project_root/
    run.py
    phase1.py
    phase2.py
    phase3.py
    phase4.py
    phase5.py
    results.pkl          ← generated on run
    generated_PDBS/      ← generated on run
        1ASY/
            rank1/  protein.pdb  rna.pdb
            rank2/  protein.pdb  rna.pdb
            ...
```

**Data directory (separate):**

```
D:\BTP Files\PRDBv3.0\
    PRDBv3_info.json
    1ASY\
        1ASY.pdb          ← bound complex
        <U_pro_ID>.pdb    ← unbound protein
        <U_RNA_ID>.pdb    ← unbound RNA
    1AV6\ ...
```

---

## 3. Prerequisites & Installation

**Python 3.9+** is required.

```bash
pip install numpy scipy torch pydantic plotly

# For GPU acceleration — match the CUDA version to your driver:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Verify GPU visibility:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Default data paths are hardcoded in `run.py` and can always be overridden:

```python
DEFAULT_JSON     = r"D:\BTP Files\PRDBv3.0\PRDBv3_info.json"
DEFAULT_PDB_ROOT = r"D:\BTP Files\PRDBv3.0"
```

> **Important:** Always `cd` into the project directory before running `run.py`.
> Python resolves the script path relative to your *current working directory*.
> Running `python run.py` from `C:\Users\YourName` will raise `No such file or directory`.
>
> ```powershell
> cd "D:\BTP Files\pipeline"
> python run.py --complex 1ASY
> ```

---

## 4. Running the Pipeline

`run.py` is the **only script you need to call**. It drives all five phases end-to-end.

### 4.1 Quick-start examples

```bash
# Dock a single complex — no visualizations
python run.py --complex 1ASY

# Dock a single complex — show ALL visualizations (5 browser tabs)
python run.py --complex 1ASY --viz

# Dock several complexes at once
python run.py --complex 1ASY 1AV6 1B23

# Dock every UU case in the dataset
python run.py --all

# Finer control over docking parameters
python run.py --complex 1ASY --step 15.0 --resolution 1.0 --top_n 10

# Override data paths
python run.py --complex 1ASY \
    --json     "E:\MyData\PRDBv3_info.json" \
    --pdb_root "E:\MyData"
```

### 4.2 `--viz` flag reference

`--viz` accepts zero or more space-separated type names.  
**`--viz` alone (no names) enables all three types.**

| Command | Browser tabs opened |
|---------|-------------------|
| `python run.py --complex 1ASY --viz` | All 5: structure + protein grid + RNA grid + rotation axes + angle histogram |
| `python run.py --complex 1ASY --viz structure` | 1 tab — 3-D atom scatter of the bound complex (protein/RNA coloured separately) |
| `python run.py --complex 1ASY --viz grid` | 2 tabs — protein voxel grid and RNA voxel grid (surface = red, interior = blue) |
| `python run.py --complex 1ASY --viz rotations` | 2 tabs — SO(3) rotation-axis sphere + rotation-angle histogram |
| `python run.py --complex 1ASY --viz structure grid` | 3 tabs — atom scatter + both voxel grids |
| `python run.py --complex 1ASY --viz grid rotations` | 4 tabs — voxel grids + rotation coverage plots |
| `python run.py --complex 1ASY --viz structure rotations` | 3 tabs — atom scatter + rotation plots |
| `python run.py --complex 1ASY` | No tabs — visualizations fully disabled |
| `python run.py --all --viz rotations` | Rotation plots open **once** only; all complexes dock silently |

> **Tip:** `rotations` opens exactly once regardless of how many complexes you process, because the SO(3) set is identical for all complexes at the same `--step`.  
> `structure` and `grid` open tabs **per complex** — avoid combining with `--all` on large datasets.

### 4.3 Full flag reference

| Flag | Default | Description |
|------|---------|-------------|
| `--complex ID…` | — | One or more complex IDs. Mutually exclusive with `--all`. |
| `--all` | — | Run on every UU case in the JSON. Mutually exclusive with `--complex`. |
| `--viz [TYPE…]` | off | Visualizations. No argument = all three. Types: `structure` `grid` `rotations`. |
| `--step` | `30.0` | Angular step in degrees for SO(3) sampling. Halving it increases rotation count ~8×. |
| `--resolution` | `1.0` | Voxel edge length in Å. Smaller = finer grid = slower. |
| `--top_n` | `5` | Number of top-ranked poses to export per complex. |
| `--json` | `D:\BTP…json` | Absolute path to `PRDBv3_info.json`. |
| `--pdb_root` | `D:\BTP…` | Root folder containing one sub-folder per complex ID. |
| `--output` | `generated_PDBS` | Root output directory for ranked PDB files. |
| `--results` | `results.pkl` | Pickle file for raw `DockingResult` objects. |

### 4.4 Expected console output

```
══════════════════════════════════════════════════════════════════
  Protein-RNA FFT Docking Pipeline
══════════════════════════════════════════════════════════════════
  Mode        : ['1ASY']
  Angular step: 30.0°   Resolution: 1.0 Å   Top-N: 5
  Visualize   : off  (pass --viz to enable)

══ Phase 1 — Loading & parsing PDB structures ══
  ... Phase 1 summary table ...

══ Phase 3 — Building SO(3) rotation set ══
SO3Sampler: generating rotations at 30.0° step … 512 rotations generated.

── Complex 1/1: 1ASY — Phases 2 & 4 ──
  [Hardware] FFTDocker using device: CUDA
  Common Grid Shape : (128, 128, 256)
  Building fixed protein grid...
  Building native RNA grid (once)...
  Evaluating 512 rotations...
  [1ASY] Docking complete in 142.37 seconds.
  Top Score: 1847.00

══ Phase 5 — 1ASY ══
  Rank   Score      L-RMSD (Å)   I-RMSD (Å)   Output
  1      1847.00       12.34         8.91       generated_PDBS/1ASY/rank1

══ Global Benchmark Summary ══
  Complex      Rank   Score      L-RMSD (Å)   I-RMSD (Å)
  1ASY         1      1847.00       12.34         8.91

  Results pickle saved → results.pkl
```

---

## 5. Output Structure

**`results.pkl`** — Python pickle containing a dict of `{complex_id: List[DockingResult]}`:

```python
import pickle
with open('results.pkl', 'rb') as f:
    results = pickle.load(f)

top = results['1ASY'][0]
print(top.score)              # float
print(top.rotation_matrix)    # np.ndarray (3×3)
print(top.translation_vector) # np.ndarray (3,) in Å
```

**`generated_PDBS/`** — PDB files for each ranked pose. Protein coordinates are unchanged (fixed receptor). RNA coordinates have the docking transform applied:

```
coord_docked = R @ (coord_unbound − center) + center + t
```

where `R` is the rotation matrix, `t` the translation vector, and `center` the geometric centre of the unbound RNA.

---

## 6. Mathematical Reference

### 6.1 Katchalski-Katzir Voxel Encoding

The continuous molecular structure is discretised into a 3-D grid G of dimensions Nₓ × Nᵧ × N_z. Each voxel at position **r** is assigned a value from {−15, 0, +1}:

```
G(r) = −15   if  min_a ‖r − r_a‖ < R_a − δ       (interior)
G(r) = +1    if  min_a ‖r − r_a‖ < R_a            (surface)
G(r) =  0    otherwise                             (solvent)
```

- `r_a` — atomic centre of atom `a`
- `R_a` — Van der Waals radius of atom `a` (Bondi 1964 + AMBER99)
- `δ = 1.4 Å` — water probe radius (Lee & Richards 1971), defines surface layer thickness

The −15 interior penalty is chosen so that a **single steric clash completely dominates any number of surface contacts**, enforcing a hard exclusion constraint. Grid dimensions are always rounded up to the next power of two for FFT efficiency.

The shape complementarity score for translational offset **t** is:

```
S(t) = Σ_r  P(r) · R(r − t)
```

Positive contributions arise where surface voxels align with surface voxels (+1 × +1 = +1). Negative contributions arise where a surface voxel aligns with an interior voxel (−15 × +1 = −15), penalising clashes.

---

### 6.2 FFT Cross-Correlation Scoring

Direct evaluation of S(**t**) for all N³ translations requires **O(N⁶)** operations. The Convolution Theorem reduces this to **O(N³ log N)**:

```
S(t) = Σ_r P(r) · R(r − t)  =  [P ★ R](t)      (cross-correlation)

S  =  F⁻¹( F(P) · conj(F(R)) )
```

where `F` is the 3-D discrete Fourier transform and `conj` is the complex conjugate.

**Implementation:**

```python
fft_pro  = torch.fft.fftn(pro_tensor)            # computed once before the loop
fft_rna  = torch.fft.fftn(rna_tensor_rotated)    # per rotation
corr     = torch.fft.ifftn(fft_pro * torch.conj(fft_rna)).real
best_score = corr.max()
best_voxel = corr.argmax()                        # unravelled to (ix, iy, iz)
```

The protein FFT is computed **once** and reused across all rotations, since the protein grid never changes. For N = 128 and K = 512 rotations this represents a ~10⁶× speedup over naive enumeration.

---

### 6.3 Uniform SO(3) Sampling via Hopf Fibration

Every rotation R ∈ SO(3) corresponds to a unit quaternion **q** = (w, x, y, z) ∈ S³. Naive uniform sampling in Euler angle space causes polar over-sampling ("gimbal lock"). The **Hopf fibration** decomposes S³ into a product of S¹ fibres over S², enabling uniform sampling via three independent angles (θ, φ, ψ):

```
w = cos(θ) · cos(ψ/2)
x = cos(θ) · sin(ψ/2)
y = sin(θ) · cos(φ + ψ/2)
z = sin(θ) · sin(φ + ψ/2)

θ ∈ [0, π/2],   φ ∈ [0, 2π),   ψ ∈ [0, 2π)
```

Midpoint sampling at `θᵢ = (π/2)(i + 0.5) / N_θ` with equal spacing in φ and ψ gives near-uniform SO(3) coverage. Rotation count scales as **O(step⁻³)**:

| Angular step | Rotation count |
|---|---|
| 30° | ~512 |
| 15° | ~4,096 |
| 10° | ~13,824 |
| 6° | ~64,000 |

Each quaternion is converted to a 3×3 rotation matrix via the Shoemake (1985) formula:

```
R = [[1−2(y²+z²),   2(xy−zw),    2(xz+yw)  ],
     [  2(xy+zw),  1−2(x²+z²),   2(yz−xw)  ],
     [  2(xz−yw),   2(yz+xw),   1−2(x²+y²) ]]
```

---

### 6.4 Kabsch Superposition Algorithm

Finds the optimal rigid-body superposition (rotation + translation) minimising RMSD between two corresponding point sets P (mobile) and Q (reference). Used in Phase 5 to align the predicted protein onto the reference before computing L-RMSD.

1. Centre both sets: `P_c = P − mean(P)`,  `Q_c = Q − mean(Q)`
2. Covariance matrix: `H = P_cᵀ Q_c`
3. SVD: `H = U S Vᵀ`
4. Reflection correction: `d = sign(det(VᵀUᵀ))`,  `D = diag(1, 1, d)`
5. Optimal rotation: `R = Vᵀ D Uᵀ`
6. Optimal translation: `t = mean(Q) − R · mean(P)`
7. Apply: `P_aligned = (R · Pᵀ)ᵀ + t`

The determinant correction in step 4 ensures R is a **proper rotation** (det = +1), not a reflection.

---

### 6.5 Ligand RMSD (L-RMSD)

Standard CAPRI metric. Measures how well the predicted RNA position matches the reference, after optimally superimposing the predicted complex onto the reference protein.

**Steps:**
1. Extract Cα atoms from predicted unbound protein and reference bound protein.
2. Match residues by `(chain_id, res_seq)`.
3. Compute Kabsch superposition: `R_sup, t_sup = kabsch(pred_Cα, ref_Cα)`.
4. Apply to docked RNA C4' coordinates: `pred_C4'_sup = R_sup · pred_C4' + t_sup`.
5. Compute RMSD against reference C4' atoms.

```
L-RMSD = sqrt( (1/N) · Σᵢ ‖pred_C4'_sup(i) − ref_C4'(i)‖² )
```

N = number of matched C4' atoms. C4' is the ribose sugar carbon used as the RNA backbone representative atom.

**CAPRI quality thresholds:** < 10 Å = acceptable · < 5 Å = medium · < 1 Å = high

---

### 6.6 Interface RMSD (I-RMSD)

More sensitive to binding-site accuracy than L-RMSD. Interface residues are those with any heavy atom within **10 Å** of any heavy atom of the partner in the reference complex.

**Steps:**
1. Identify interface protein residues (any atom within 10 Å of RNA in the reference complex).
2. Identify interface RNA residues (any atom within 10 Å of protein in the reference complex).
3. Extract Cα of interface protein residues and C4' of interface RNA residues (predicted + reference).
4. Stack into combined arrays `pred_int` and `ref_int`.
5. Kabsch superposition of `pred_int` onto `ref_int`.
6. Compute RMSD of superimposed interface atoms.

```
I-RMSD = sqrt( (1/M) · Σⱼ ‖pred_int_sup(j) − ref_int(j)‖² )
```

M = total matched interface backbone atoms (Cα + C4'). Unlike L-RMSD, the superposition is performed on the **interface atoms themselves**, making it more sensitive to local binding-site geometry.

**CAPRI quality thresholds:** < 4 Å = acceptable · < 2 Å = medium · < 1 Å = high

---

### 6.7 Buried Surface Area (BSA)

BSA quantifies solvent-accessible surface area (SASA) that becomes shielded when the two molecules associate. It is a proxy for interface size and correlates with binding affinity.

```
BSA = SASA(protein_free) + SASA(RNA_free) − SASA(complex)
```

SASA is calculated by rolling a probe sphere (r = 1.4 Å) over the Van der Waals surface. BSA is **not computed during the FFT search** (too expensive per-pose), but the voxel scoring function is its discrete analogue: maximising surface-contact (+1 × +1) terms is equivalent to maximising buried surface area.

To compute precise BSA on the final top poses:

```python
import freesasa
sasa_c = freesasa.calc(freesasa.Structure('rank1/complex.pdb')).totalArea()
sasa_p = freesasa.calc(freesasa.Structure('rank1/protein.pdb')).totalArea()
sasa_r = freesasa.calc(freesasa.Structure('rank1/rna.pdb')).totalArea()
bsa = sasa_p + sasa_r - sasa_c
print(f'BSA = {bsa:.1f} Å²')   # typical range: 1000–3000 Å²
```

---

## 7. References

- **Katchalski-Katzir et al. (1992)** — Molecular surface recognition: determination of geometric fit between proteins and their ligands by correlation techniques. *PNAS* 89(6), 2195–2199.
- **Shoemake, K. (1985)** — Animating rotation with quaternion curves. *SIGGRAPH '85*.
- **Kabsch, W. (1976)** — A solution for the best rotation to relate two sets of vectors. *Acta Crystallographica* A32, 922–923.
- **Lee & Richards (1971)** — The interpretation of protein structures: estimation of static accessibility. *Journal of Molecular Biology* 55(3), 379–400.
- **Bondi, A. (1964)** — Van der Waals Volumes and Radii. *Journal of Physical Chemistry* 68(3), 441–451.
- **PRDBv3** — Protein–RNA Docking Benchmark v3.0.
