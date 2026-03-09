# Prot-RNA-Docking-BTP

Welcome to the **Prot-RNA-Docking-BTP** repository. This project focuses on the computational prediction of 3D Protein-RNA complex structures starting from their unbound (UU) states. 

Macromolecular docking is a massive computationally expensive search problem. This repository breaks down the pipeline into data curation, interactive visualization, and a mathematically rigorous rigid-body docking engine using Fast Fourier Transforms (FFT).

---

## Repository Structure & Navigation

This repository is organized into distinct modules separating data management, visualization, the core docking algorithm, and literature.

### 1. `Master Index - Kaggle/`
**Purpose:** Data Management and Indexing.
Working with large structural databases (like PRDBv3) requires efficient file tracking. 
* **Contents:** Contains the `.csv` master index of all PDB files natively hosted in the Kaggle environment, alongside the scripts used to generate this index.
* **Logic:** Instead of searching through nested directories during runtime, the master index provides $O(1)$ lookup times for bound complexes, unbound proteins, and unbound RNA file paths, streamlining batch processing on Kaggle.

### 2. `Interactive 3D Plotter - Kaggle/`
**Purpose:** Structural Visualization and Sanity Checking.
* **Contents:** Kaggle-native Python scripts/notebooks utilizing the Master Index to render user-interactive 3D plots of the molecules.
* **Logic:** Before running complex docking algorithms, it is crucial to visually verify the integrity of the PDB files. This tool parses the atomic coordinates and renders 3D scatter/surface plots of the unbound RNA, unbound protein, and the native bound complex. It allows researchers to visually inspect binding pockets and structural conformations directly in the browser.

### 3. `Phase 1/` (Core FFT Docking Engine)
**Purpose:** Rigid-Body Shape Complementarity Search.
This folder contains the local Python pipeline for the actual docking prediction. 
* **`PDBparser.py`**: Extracts and cleans $x, y, z$ atomic coordinates from the PDB files.
* **`grid3d.py`**: Discretizes the continuous molecular coordinates into a 3D Cartesian voxel grid. Voxels are assigned values based on the Katchalski-Katzir representation to reward surface overlap ($+1$) and heavily penalize interior steric clashes ($-15$).
* **`RotSampler.py`**: Generates a uniform set of rotation matrices covering the $SO(3)$ space using quaternions $q = (w, x, y, z)$ derived from Hopf coordinates. This prevents the "gimbal lock" or pole-oversampling issues common in basic Euler angle sampling.
* **`FFT.py`**: The computational heart of the phase. It rotates the RNA, places it in a shared grid with the static protein, and uses the Convolution Theorem to evaluate all possible translation vectors $\mathbf{t}$ simultaneously:
  $$S = \mathcal{F}^{-1}(\mathcal{F}(P) \cdot \overline{\mathcal{F}(R_{rot})})$$
  This reduces the search complexity for a grid of size $N$ from $O(N^6)$ to $O(N^3 \log N)$.

### 4. `Research papers/`
**Purpose:** Theoretical Foundation.
* **Contents:** A curated collection of PDFs and reference materials that form the biological and mathematical backbone of this project (e.g., Katchalski-Katzir algorithm, uniform $SO(3)$ sampling metrics, Bondi VDW radii definitions).

---

## Biological & Mathematical Rationale

The central dogma of this docking algorithm relies on **shape complementarity**. Biologically, proteins and RNA interact across their solvent-accessible surfaces via non-covalent bonds. Physically, their solid interiors cannot overlap due to Pauli exclusion forces (steric clashes). 

By digitizing the molecules into discrete 3D grids, we mathematically simulate these physical laws. The FFT acts as a rapid correlation engine, sliding the RNA grid across the Protein grid to find the exact spatial translation where the surfaces interlock perfectly, minimizing clashes.

---

## Quickstart Guide

1. **Explore the Data:** If you are using Kaggle, start with the `Master Index - Kaggle/` to set up your file paths.
2. **Visualize:** Run the `Interactive 3D Plotter` to inspect your specific Protein-RNA pairs.
3. **Dock (Local):** Navigate to `Phase 1/` on your local machine and run the FFT search:
   ```bash
   python FFT.py --step 15.0 --resolution 1.0
   ```
(Ensure you have installed numpy, scipy, and plotly before running).

Developed as part of a B.Tech Project (BTP) submitted at Indian Institute of Technology (IIT) Kharagpur.
